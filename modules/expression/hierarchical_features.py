import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio


AXIS_NAMES = ("accent", "intensity", "breathiness")
LANGUAGE_IDS = {"unknown": 0, "en": 1, "ko": 2, "ja": 3}
COMPONENT_NAMES = (
    "accent_energy_delta_db",
    "accent_energy_slope_db",
    "accent_f0_rise_st",
    "accent_emphasis_delta_db",
    "intensity_energy_db",
    "intensity_emphasis_db",
    "breath_cpp_db",
    "breath_hnr_db",
    "breath_h1_h2_db",
    "breath_hf_flatness",
)

AXIS_COMPONENT_WEIGHTS = {
    # Singing accent is an onset attack, not a melodic pitch-rise event.
    "accent": (0.40, 0.35, 0.00, 0.25),
    "intensity": (0.50, 0.50),
    "breathiness": (0.40, 0.35, 0.15, 0.10),
}

M4_CONSONANTS = {
    "b",
    "p",
    "m",
    "f",
    "d",
    "t",
    "n",
    "l",
    "g",
    "k",
    "h",
    "j",
    "q",
    "x",
    "zh",
    "ch",
    "sh",
    "r",
    "z",
    "c",
    "s",
    "y",
    "w",
}
NUS_VOWELS = {
    "aa",
    "ae",
    "ah",
    "ao",
    "aw",
    "ax",
    "axr",
    "ay",
    "eh",
    "er",
    "ey",
    "ih",
    "ix",
    "iy",
    "ow",
    "oy",
    "uh",
    "uw",
    "ux",
}
JAPANESE_VOWELS = {
    "a",
    "aː",
    "e",
    "eː",
    "i",
    "i̥",
    "o",
    "oː",
    "u",
    "ɨ",
    "ɨː",
    "ɨ̥",
    "ɯ",
    "ɯ̥",
}
KOREAN_VOWELS = {
    "a",
    "ae",
    "e",
    "eo",
    "eu",
    "i",
    "o",
    "oe",
    "u",
    "ui",
    "wa",
    "wae",
    "we",
    "wi",
    "wo",
    "ya",
    "yae",
    "ye",
    "yeo",
    "yo",
    "yu",
    "eː",
    "iː",
    "oː",
    "uː",
    "ɐ",
    "ɛ",
    "ɛː",
    "ɨ",
    "ɨː",
    "ʌ",
    "ʌː",
}
SILENCE_LABELS = {
    "",
    "sil",
    "sp",
    "spn",
    "pau",
    "br",
    "<sp>",
    "<ap>",
    "none",
}


@dataclass
class FrameAlignment:
    onset_times: list
    active_mask: torch.Tensor
    stable_vowel_mask: torch.Tensor
    vowel_onset_times: list = field(default_factory=list)
    accent_events: list = field(default_factory=list)


@dataclass
class AccentEvent:
    onset_time: float
    vowel_start_time: float
    vowel_end_time: float
    lexical_stress: float = 0.5
    phrase_initial: float = 0.0

    @property
    def duration(self):
        return max(float(self.vowel_end_time) - float(self.vowel_start_time), 0.0)


def _normalise_label(label):
    return label.strip().lower().rstrip("0123456789")


def _english_stress(label):
    label = str(label).strip()
    if not label or label[-1] not in "012":
        return 0.5
    return {"0": 0.30, "1": 1.0, "2": 0.70}[label[-1]]


def _interval_mask(frame_times, intervals, trim_seconds=0.0):
    mask = torch.zeros_like(frame_times, dtype=torch.bool)
    for start, end in intervals:
        duration = max(end - start, 0.0)
        trim = min(trim_seconds, duration * 0.2)
        lo = start + trim
        hi = end - trim
        if hi > lo:
            mask |= (frame_times >= lo) & (frame_times <= hi)
    return mask


def read_m4_syllable_onset_times(textgrid_path):
    return read_m4_onset_landmarks(textgrid_path)["syllable"]


def read_m4_onset_landmarks(textgrid_path):
    from praatio import textgrid

    grid = textgrid.openTextgrid(
        str(textgrid_path),
        includeEmptyIntervals=True,
        duplicateNamesMode="rename",
    )
    tier_names = list(grid.tierNames)
    if len(tier_names) < 2:
        raise ValueError(f"Expected two TextGrid tiers: {textgrid_path}")

    syllable_entries = grid.getTier(tier_names[0]).entries
    phone_entries = grid.getTier(tier_names[1]).entries
    syllable = [
        float(entry.start)
        for entry in syllable_entries
        if _normalise_label(entry.label) not in SILENCE_LABELS
    ]
    vowel = [
        float(entry.start)
        for entry in phone_entries
        if (
            _normalise_label(entry.label) not in SILENCE_LABELS
            and _normalise_label(entry.label) not in M4_CONSONANTS
        )
    ]
    return {"syllable": syllable, "vowel": vowel}


def parse_m4_textgrid(textgrid_path, frame_times, onset_mode="vowel"):
    from praatio import textgrid

    grid = textgrid.openTextgrid(
        str(textgrid_path),
        includeEmptyIntervals=True,
        duplicateNamesMode="rename",
    )
    tier_names = list(grid.tierNames)
    if len(tier_names) < 2:
        raise ValueError(f"Expected two TextGrid tiers: {textgrid_path}")

    syllable_entries = grid.getTier(tier_names[0]).entries
    phone_entries = grid.getTier(tier_names[1]).entries

    active_intervals = []
    syllable_onsets = []
    for entry in syllable_entries:
        label = _normalise_label(entry.label)
        if label not in SILENCE_LABELS:
            active_intervals.append((entry.start, entry.end))
            syllable_onsets.append(float(entry.start))

    vowel_intervals = []
    vowel_onsets = []
    for entry in phone_entries:
        label = _normalise_label(entry.label)
        if label not in SILENCE_LABELS and label not in M4_CONSONANTS:
            vowel_intervals.append((entry.start, entry.end))
            vowel_onsets.append(float(entry.start))

    if onset_mode == "syllable":
        onset_times = syllable_onsets
    elif onset_mode == "vowel":
        onset_times = vowel_onsets
    else:
        raise ValueError(f"Unknown M4Singer onset mode: {onset_mode}")

    return FrameAlignment(
        onset_times=onset_times,
        active_mask=_interval_mask(frame_times, active_intervals),
        stable_vowel_mask=_interval_mask(frame_times, vowel_intervals, trim_seconds=0.035),
        vowel_onset_times=vowel_onsets,
    )


def event_gate_from_onset_times(
    frame_times,
    onset_times,
    sigma_seconds=0.055,
):
    gate = torch.zeros_like(frame_times)
    if frame_times.numel() < 2:
        return gate
    frame_step = float((frame_times[1] - frame_times[0]).detach().cpu())
    sigma_frames = max(sigma_seconds / max(frame_step, 1e-6), 1.0)
    positions = torch.arange(
        frame_times.numel(),
        device=frame_times.device,
        dtype=frame_times.dtype,
    )
    for onset_time in onset_times:
        center = int(torch.argmin((frame_times - float(onset_time)).abs()))
        pulse = torch.exp(-0.5 * ((positions - center) / sigma_frames).square())
        gate = torch.maximum(gate, pulse)
    return gate


def _smooth_interval_gate(frame_times, start_time, end_time, edge_seconds=0.060):
    gate = torch.zeros_like(frame_times)
    start_time = float(start_time)
    end_time = float(end_time)
    if end_time <= start_time or frame_times.numel() == 0:
        return gate
    inside = (frame_times >= start_time) & (frame_times <= end_time)
    if not inside.any():
        return gate
    duration = end_time - start_time
    edge = max(min(float(edge_seconds), duration * 0.25), 1e-4)
    rise = ((frame_times - start_time) / edge).clamp(0.0, 1.0)
    fall = ((end_time - frame_times) / edge).clamp(0.0, 1.0)
    gate[inside] = torch.minimum(rise, fall)[inside]
    return torch.sin(0.5 * math.pi * gate).square()


def _event_acoustic_salience(log_mel, frame_times, event):
    if log_mel is None or log_mel.numel() == 0:
        return 0.5
    log_power = 2.0 * log_mel.float()
    energy_db = (10.0 / math.log(10.0)) * torch.logsumexp(log_power, dim=0)
    event_mask = (
        (frame_times >= float(event.vowel_start_time))
        & (frame_times <= float(event.vowel_end_time))
    )
    local_mask = (
        (frame_times >= float(event.onset_time) - 0.35)
        & (frame_times <= float(event.vowel_end_time) + 0.35)
    )
    if not event_mask.any() or local_mask.sum() < 3:
        return 0.5
    local = energy_db[local_mask]
    robust_scale = (local.quantile(0.75) - local.quantile(0.25)).clamp_min(1.5)
    relative = (energy_db[event_mask].mean() - local.median()) / robust_scale
    return float(torch.sigmoid(relative).clamp(0.15, 0.95).detach().cpu())


def build_accent_event_tensors(frame_times, accent_events, language, log_mel=None):
    """Build attack, nucleus, and duration-aware sustain gates.

    The frame context stores performed salience, normalized vowel duration,
    phrase-initial status, and whether the event began before the current crop.
    """
    component_gates = torch.zeros(
        3,
        frame_times.numel(),
        device=frame_times.device,
        dtype=frame_times.dtype,
    )
    event_context = torch.zeros(
        4,
        frame_times.numel(),
        device=frame_times.device,
        dtype=frame_times.dtype,
    )
    language = str(language or "unknown").lower()
    frame_step = (
        float((frame_times[1] - frame_times[0]).abs())
        if frame_times.numel() > 1
        else 0.01
    )

    def landmark_gate(time_value, sigma_seconds):
        if (
            float(time_value) < float(frame_times[0]) - frame_step
            or float(time_value) > float(frame_times[-1]) + frame_step
        ):
            return torch.zeros_like(frame_times)
        return event_gate_from_onset_times(
            frame_times,
            [time_value],
            sigma_seconds=sigma_seconds,
        )

    for event in accent_events:
        attack = landmark_gate(event.onset_time, 0.050)
        nucleus = landmark_gate(event.vowel_start_time, 0.045)
        sustain = _smooth_interval_gate(
            frame_times,
            event.vowel_start_time,
            event.vowel_end_time,
        )
        acoustic = _event_acoustic_salience(log_mel, frame_times, event)
        if language == "en":
            salience = 0.70 * float(event.lexical_stress) + 0.30 * acoustic
        else:
            salience = 0.75 * acoustic + 0.25 * float(event.phrase_initial)
        salience = max(0.15, min(salience, 1.0))
        duration_value = math.log1p(event.duration / 0.08) / math.log1p(2.0 / 0.08)
        duration_value = max(0.0, min(duration_value, 1.0))
        event_geometry = torch.maximum(torch.maximum(attack, nucleus), sustain)
        context_values = (salience, duration_value, event.phrase_initial, event.onset_time < 0.0)

        component_gates[0] = torch.maximum(component_gates[0], attack)
        component_gates[1] = torch.maximum(component_gates[1], nucleus)
        component_gates[2] = torch.maximum(component_gates[2], sustain)
        for index, value in enumerate(context_values):
            event_context[index] = torch.maximum(
                event_context[index],
                event_geometry * float(value),
            )
    return component_gates, event_context


def accent_events_from_payload(payload, duration_seconds):
    serialized = payload.get("accent_events")
    if isinstance(serialized, list) and serialized:
        return [
            AccentEvent(
                onset_time=float(item["onset_time_seconds"]),
                vowel_start_time=float(item["vowel_start_time_seconds"]),
                vowel_end_time=float(item["vowel_end_time_seconds"]),
                lexical_stress=float(item.get("lexical_stress", 0.5)),
                phrase_initial=float(item.get("phrase_initial", 0.0)),
            )
            for item in serialized
        ]

    onsets = [
        float(value)
        for value in payload.get(
            "onset_times_seconds",
            payload.get("onset_times", []),
        )
    ]
    vowels = [
        float(value)
        for value in payload.get(
            "vowel_onset_times_seconds",
            payload.get("vowel_onset_times", onsets),
        )
    ]
    events = []
    for index, vowel_start in enumerate(vowels):
        preceding = [onset for onset in onsets if onset <= vowel_start + 1e-4]
        onset = preceding[-1] if preceding else vowel_start
        next_vowel = vowels[index + 1] if index + 1 < len(vowels) else duration_seconds
        vowel_end = min(max(vowel_start + 0.12, next_vowel - 0.02), duration_seconds)
        events.append(AccentEvent(onset, vowel_start, vowel_end))
    return events


def replace_accent_gate_with_onsets(
    gates,
    onset_times,
    duration_seconds,
    sigma_seconds=0.055,
):
    return replace_accent_gate_with_landmarks(
        gates,
        syllable_onset_times=onset_times,
        vowel_onset_times=None,
        duration_seconds=duration_seconds,
        syllable_sigma_seconds=sigma_seconds,
    )


def replace_accent_gate_with_landmarks(
    gates,
    syllable_onset_times,
    vowel_onset_times,
    duration_seconds,
    syllable_sigma_seconds=0.055,
    vowel_sigma_seconds=0.045,
    vowel_weight=0.75,
):
    if gates.dim() not in (2, 3) or gates.size(-2) != len(AXIS_NAMES):
        raise ValueError(f"Expected [3,T] or [B,3,T] gates, got {gates.shape}")
    frames = gates.size(-1)
    frame_times = (
        torch.arange(frames, device=gates.device, dtype=torch.float32) + 0.5
    ) * (float(duration_seconds) / max(frames, 1))
    syllable_gate = event_gate_from_onset_times(
        frame_times,
        syllable_onset_times,
        sigma_seconds=syllable_sigma_seconds,
    ).to(dtype=gates.dtype)
    accent = syllable_gate
    if vowel_onset_times:
        vowel_gate = event_gate_from_onset_times(
            frame_times,
            vowel_onset_times,
            sigma_seconds=vowel_sigma_seconds,
        ).to(dtype=gates.dtype)
        accent = torch.maximum(accent, float(vowel_weight) * vowel_gate)
    output = gates.clone()
    if gates.dim() == 2:
        output[0] = accent * (gates[1] > 0).to(dtype=gates.dtype)
    else:
        output[:, 0] = accent[None] * (gates[:, 1] > 0).to(dtype=gates.dtype)
    return output


def parse_nus_alignment(annotation_path, segment_start, segment_end, frame_times):
    phone_entries = []
    with Path(annotation_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            start, end = float(parts[0]), float(parts[1])
            if end < segment_start or start > segment_end:
                continue
            phone_entries.append(
                (
                    max(start, segment_start) - segment_start,
                    min(end, segment_end) - segment_start,
                    _normalise_label(parts[2]),
                )
            )

    active_intervals = []
    vowel_intervals = []
    onset_times = []
    for start, end, label in phone_entries:
        if label not in SILENCE_LABELS:
            active_intervals.append((start, end))
        if label in NUS_VOWELS:
            vowel_intervals.append((start, end))
            onset_times.append(float(start))

    return FrameAlignment(
        onset_times=onset_times,
        active_mask=_interval_mask(frame_times, active_intervals),
        stable_vowel_mask=_interval_mask(frame_times, vowel_intervals, trim_seconds=0.025),
        vowel_onset_times=onset_times,
    )


def _relative_interval(start, end, segment_start, segment_end):
    if end <= segment_start or start >= segment_end:
        return None
    return (
        max(start, segment_start) - segment_start,
        min(end, segment_end) - segment_start,
    )


def _is_vowel(label, language):
    label = _normalise_label(label)
    if language == "en":
        return label in NUS_VOWELS
    if language == "ko":
        return label in KOREAN_VOWELS
    if language == "ja":
        return label in JAPANESE_VOWELS
    return label in NUS_VOWELS or label in JAPANESE_VOWELS


def parse_gtsinger_alignment(
    annotation_path,
    segment_start,
    segment_end,
    frame_times,
    language,
):
    entries = json.loads(Path(annotation_path).read_text(encoding="utf-8"))
    active_intervals = []
    vowel_intervals = []
    vowel_onsets = []
    word_onsets = []
    accent_events = []
    previous_voiced_end = None
    previous_was_pause = True

    for entry in entries:
        word = _normalise_label(str(entry.get("word", "")))
        word_start = float(entry.get("start_time", 0.0))
        word_end = float(entry.get("end_time", word_start))
        is_sounding = word not in SILENCE_LABELS
        phrase_initial = (
            previous_was_pause
            or previous_voiced_end is None
            or word_start - previous_voiced_end >= 0.18
        )
        if is_sounding:
            relative = _relative_interval(
                word_start,
                word_end,
                segment_start,
                segment_end,
            )
            if relative is not None and segment_start <= word_start < segment_end:
                word_onsets.append(word_start - segment_start)

        raw_labels = [str(label) for label in entry.get("ph", [])]
        phone_starts = [float(value) for value in entry.get("ph_start", [])]
        phone_ends = [float(value) for value in entry.get("ph_end", [])]
        last_vowel_index = -1
        for phone_index, (raw_label, start, end) in enumerate(
            zip(raw_labels, phone_starts, phone_ends)
        ):
            label = _normalise_label(raw_label)
            relative = _relative_interval(
                start,
                end,
                segment_start,
                segment_end,
            )
            if relative is None or label in SILENCE_LABELS:
                continue
            active_intervals.append(relative)
            if _is_vowel(label, language):
                vowel_intervals.append(relative)
                if segment_start <= start < segment_end:
                    vowel_onsets.append(start - segment_start)
                syllable_phone_index = last_vowel_index + 1
                syllable_onset = (
                    phone_starts[syllable_phone_index]
                    if syllable_phone_index < len(phone_starts)
                    else start
                )
                if last_vowel_index < 0:
                    syllable_onset = word_start
                if end > segment_start and syllable_onset < segment_end:
                    accent_events.append(
                        AccentEvent(
                            onset_time=syllable_onset - segment_start,
                            vowel_start_time=start - segment_start,
                            vowel_end_time=end - segment_start,
                            lexical_stress=(
                                _english_stress(raw_label)
                                if language == "en"
                                else 0.5
                            ),
                            phrase_initial=float(
                                phrase_initial and last_vowel_index < 0
                            ),
                        )
                    )
                last_vowel_index = phone_index
        if is_sounding:
            previous_voiced_end = word_end
            previous_was_pause = False
        else:
            previous_was_pause = True

    return FrameAlignment(
        onset_times=word_onsets or vowel_onsets,
        active_mask=_interval_mask(frame_times, active_intervals),
        stable_vowel_mask=_interval_mask(
            frame_times,
            vowel_intervals,
            trim_seconds=0.025,
        ),
        vowel_onset_times=vowel_onsets,
        accent_events=accent_events,
    )


def fallback_alignment(frame_times, f0, energy_db):
    active = f0 > 0
    if active.any():
        energy_threshold = torch.quantile(energy_db[active], 0.15)
        active &= energy_db >= energy_threshold
    else:
        energy_threshold = torch.quantile(energy_db, 0.60)
        active = energy_db >= energy_threshold

    energy_delta = F.pad(energy_db[1:] - energy_db[:-1], (1, 0))
    local_max = energy_delta == F.max_pool1d(
        energy_delta[None, None],
        kernel_size=9,
        stride=1,
        padding=4,
    )[0, 0]
    threshold = torch.quantile(energy_delta[active], 0.80) if active.any() else 0.0
    onset_mask = local_max & active & (energy_delta >= threshold)
    onset_times = frame_times[onset_mask].detach().cpu().tolist()

    stable = active & (energy_delta.abs() <= torch.quantile(energy_delta.abs(), 0.70))
    return FrameAlignment(onset_times, active, stable)


def _gaussian_smooth(values, sigma_frames):
    if sigma_frames <= 0:
        return values
    radius = max(int(math.ceil(3.0 * sigma_frames)), 1)
    positions = torch.arange(-radius, radius + 1, device=values.device, dtype=values.dtype)
    kernel = torch.exp(-0.5 * (positions / sigma_frames).square())
    kernel = kernel / kernel.sum()
    padded = F.pad(
        values[None, None],
        (radius, radius),
        mode="replicate",
    )
    return F.conv1d(
        padded,
        kernel[None, None],
    )[0, 0]


def compute_log_mel_and_power(
    waveform,
    sample_rate=44100,
    n_fft=2048,
    hop_length=512,
    win_length=2048,
    n_mels=128,
    f_min=0.0,
    f_max=None,
):
    waveform = waveform.float()
    pad = int((n_fft - hop_length) / 2)
    padded = F.pad(waveform[None, None], (pad, pad), mode="reflect")[0, 0]
    window = torch.hann_window(win_length, device=waveform.device, dtype=waveform.dtype)
    spectrum = torch.stft(
        padded,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=False,
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    magnitude = spectrum.abs().clamp_min(1e-7)
    power = magnitude.square()
    mel_scale = torchaudio.transforms.MelScale(
        n_mels=n_mels,
        sample_rate=sample_rate,
        f_min=f_min,
        f_max=f_max,
        n_stft=n_fft // 2 + 1,
    ).to(device=waveform.device, dtype=waveform.dtype)
    log_mel = torch.log(mel_scale(power.sqrt()).clamp_min(1e-5))
    return log_mel, magnitude, power


def align_f0_to_frames(f0_10ms, frame_times):
    if not torch.is_tensor(f0_10ms):
        f0_10ms = torch.as_tensor(f0_10ms)
    f0_10ms = f0_10ms.to(device=frame_times.device, dtype=torch.float32)
    indices = torch.round(frame_times / 0.01).long().clamp(0, max(f0_10ms.numel() - 1, 0))
    if f0_10ms.numel() == 0:
        return torch.zeros_like(frame_times)
    return f0_10ms[indices]


def _spectral_emphasis(power, frequencies, f0):
    total = power.sum(dim=0).clamp_min(1e-10)
    cutoff = torch.where(f0 > 0, 1.5 * f0, torch.full_like(f0, 300.0))
    low_mask = frequencies[:, None] <= cutoff[None]
    low = (power * low_mask).sum(dim=0).clamp_min(1e-10)
    return 10.0 * torch.log10(total / low)


def _cepstral_and_harmonic_features(waveform, magnitude, power, f0, sample_rate, n_fft):
    log_magnitude = torch.log(magnitude.clamp_min(1e-7))
    cepstrum = torch.fft.irfft(log_magnitude, n=n_fft, dim=0)
    autocorrelation = torch.fft.irfft(power, n=n_fft, dim=0)

    num_frames = magnitude.size(1)
    frame_indices = torch.arange(num_frames, device=waveform.device)
    valid_f0 = f0 > 0
    lags = torch.round(sample_rate / f0.clamp_min(1.0)).long().clamp(1, n_fft - 1)

    acf_ratio = autocorrelation[lags, frame_indices] / autocorrelation[0].clamp_min(1e-8)
    acf_ratio = acf_ratio.clamp(1e-4, 1.0 - 1e-4)
    hnr = 10.0 * torch.log10(acf_ratio / (1.0 - acf_ratio))

    q_min = max(int(sample_rate / 1000.0), 1)
    q_max = min(int(sample_rate / 60.0), n_fft - 1)
    q_axis = torch.arange(q_min, q_max + 1, device=waveform.device, dtype=torch.float32)
    q_centered = q_axis - q_axis.mean()
    denom = q_centered.square().sum().clamp_min(1.0)
    cep_region = cepstrum[q_min : q_max + 1]
    slope = (cep_region * q_centered[:, None]).sum(dim=0) / denom
    intercept = cep_region.mean(dim=0)
    baseline_at_lag = intercept + slope * (lags.float() - q_axis.mean())
    cpp = (cepstrum[lags, frame_indices] - baseline_at_lag) * (20.0 / math.log(10.0))

    bin_hz = sample_rate / n_fft
    h1_bins = torch.round(f0 / bin_hz).long().clamp(1, magnitude.size(0) - 1)
    h2_bins = torch.round(2.0 * f0 / bin_hz).long().clamp(1, magnitude.size(0) - 1)
    h1 = magnitude[h1_bins, frame_indices].clamp_min(1e-7)
    h2 = magnitude[h2_bins, frame_indices].clamp_min(1e-7)
    h1_h2 = 20.0 * torch.log10(h1 / h2)

    frequencies = torch.linspace(
        0.0,
        sample_rate / 2.0,
        magnitude.size(0),
        device=waveform.device,
    )
    hf = power[(frequencies >= 2500.0) & (frequencies <= 8000.0)].clamp_min(1e-10)
    hf_flatness = torch.exp(torch.log(hf).mean(dim=0)) / hf.mean(dim=0).clamp_min(1e-10)

    invalid_value = torch.zeros_like(hnr)
    return (
        torch.where(valid_f0, cpp, invalid_value),
        torch.where(valid_f0, hnr, invalid_value),
        torch.where(valid_f0, h1_h2, invalid_value),
        torch.where(valid_f0, hf_flatness, invalid_value),
    )


def _masked_mean(values, mask):
    selected = values[mask]
    if selected.numel() == 0:
        return values.new_tensor(0.0)
    return selected.mean()


def _event_component_contours(
    frame_times,
    onset_times,
    energy_db,
    emphasis_db,
    f0,
    sigma_seconds=0.055,
):
    num_frames = frame_times.numel()
    contours = energy_db.new_zeros(4, num_frames)
    event_gate = energy_db.new_zeros(num_frames)
    event_mask = torch.zeros(num_frames, device=energy_db.device, dtype=torch.bool)
    if num_frames < 2:
        return contours, event_mask, event_gate

    frame_step = float((frame_times[1] - frame_times[0]).detach().cpu())
    sigma_frames = max(sigma_seconds / max(frame_step, 1e-6), 1.0)
    energy_slope = F.pad(energy_db[1:] - energy_db[:-1], (1, 0))

    for onset_time in onset_times:
        center = int(torch.argmin((frame_times - onset_time).abs()))
        pre = (frame_times >= onset_time - 0.150) & (frame_times < onset_time)
        post = (frame_times >= onset_time) & (frame_times <= onset_time + 0.120)
        local = (frame_times >= onset_time - 0.030) & (frame_times <= onset_time + 0.100)
        if not pre.any() or not post.any():
            continue

        energy_delta = _masked_mean(energy_db, post) - _masked_mean(energy_db, pre)
        positive_slope = energy_slope[local].max() if local.any() else energy_slope[center]
        emphasis_delta = _masked_mean(emphasis_db, post) - _masked_mean(emphasis_db, pre)

        pre_f0 = f0[pre & (f0 > 0)]
        post_f0 = f0[post & (f0 > 0)]
        if pre_f0.numel() and post_f0.numel():
            f0_rise = 12.0 * torch.log2(
                post_f0.median().clamp_min(1.0) / pre_f0.median().clamp_min(1.0)
            )
        else:
            f0_rise = energy_db.new_tensor(0.0)

        distance = torch.arange(num_frames, device=energy_db.device, dtype=energy_db.dtype) - center
        pulse = torch.exp(-0.5 * (distance / sigma_frames).square())
        event_gate = torch.maximum(event_gate, pulse)
        contours[0] += energy_delta * pulse
        contours[1] += positive_slope * pulse
        contours[2] += f0_rise * pulse
        contours[3] += emphasis_delta * pulse
        event_mask |= pulse >= math.exp(-2.0)

    return contours, event_mask, event_gate


def extract_hierarchical_components(
    waveform,
    sample_rate,
    f0_10ms,
    alignment_builder=None,
    n_fft=2048,
    hop_length=512,
    win_length=2048,
    n_mels=128,
):
    if waveform.dim() != 1:
        raise ValueError(f"Expected mono waveform [samples], got {tuple(waveform.shape)}")
    if sample_rate != 44100:
        waveform = torchaudio.functional.resample(
            waveform[None],
            sample_rate,
            44100,
        )[0]
        sample_rate = 44100

    log_mel, magnitude, power = compute_log_mel_and_power(
        waveform,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
    )
    num_frames = log_mel.size(1)
    frame_times = (
        torch.arange(num_frames, device=waveform.device, dtype=torch.float32) * hop_length
        + win_length / 2
    ) / sample_rate
    f0 = align_f0_to_frames(f0_10ms, frame_times)
    energy_db = 10.0 * torch.log10(power.sum(dim=0).clamp_min(1e-10))
    frequencies = torch.linspace(
        0.0,
        sample_rate / 2.0,
        power.size(0),
        device=waveform.device,
    )
    emphasis_db = _spectral_emphasis(power, frequencies, f0)

    alignment = alignment_builder(frame_times) if alignment_builder is not None else None
    if alignment is None:
        alignment = fallback_alignment(frame_times, f0, energy_db)
    active_mask = alignment.active_mask.to(waveform.device) & (f0 > 0)
    stable_mask = alignment.stable_vowel_mask.to(waveform.device) & active_mask

    accent_components, accent_mask, accent_gate = _event_component_contours(
        frame_times,
        alignment.onset_times,
        energy_db,
        emphasis_db,
        f0,
    )
    accent_mask &= active_mask

    sigma_frames = 0.200 * sample_rate / hop_length
    slow_energy = _gaussian_smooth(energy_db, sigma_frames)
    slow_emphasis = _gaussian_smooth(emphasis_db, sigma_frames)
    cpp, hnr, h1_h2, hf_flatness = _cepstral_and_harmonic_features(
        waveform,
        magnitude,
        power,
        f0,
        sample_rate,
        n_fft,
    )
    medium_sigma = 0.060 * sample_rate / hop_length
    cpp = _gaussian_smooth(cpp, medium_sigma)
    hnr = _gaussian_smooth(hnr, medium_sigma)
    h1_h2 = _gaussian_smooth(h1_h2, medium_sigma)
    hf_flatness = _gaussian_smooth(hf_flatness, medium_sigma)

    components = torch.stack(
        [
            accent_components[0],
            accent_components[1],
            accent_components[2],
            accent_components[3],
            slow_energy,
            slow_emphasis,
            cpp,
            hnr,
            h1_h2,
            hf_flatness,
        ],
        dim=0,
    )
    masks = torch.stack([accent_mask, active_mask, stable_mask], dim=0)
    structure_gates = torch.stack(
        [
            accent_gate * active_mask.to(dtype=accent_gate.dtype),
            active_mask.to(dtype=accent_gate.dtype),
            stable_mask.to(dtype=accent_gate.dtype),
        ],
        dim=0,
    )
    return {
        "log_mel": log_mel,
        "components": components,
        "masks": masks,
        "structure_gates": structure_gates,
        "f0": f0,
        "frame_times": frame_times,
        "alignment": alignment,
    }


def extract_structure_gates(
    waveform,
    sample_rate,
    f0_10ms,
    target_length,
    n_fft=2048,
    hop_length=512,
    win_length=2048,
):
    """Build annotation-free accent/active/stable gates for inference."""

    if waveform.dim() != 1:
        waveform = waveform.reshape(-1)
    if sample_rate != 44100:
        waveform = torchaudio.functional.resample(
            waveform[None],
            sample_rate,
            44100,
        )[0]
        sample_rate = 44100
    log_mel, _, power = compute_log_mel_and_power(
        waveform,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=128,
    )
    frame_times = (
        torch.arange(log_mel.size(1), device=waveform.device, dtype=torch.float32)
        * hop_length
        + win_length / 2
    ) / sample_rate
    f0 = align_f0_to_frames(f0_10ms, frame_times)
    energy_db = 10.0 * torch.log10(power.sum(dim=0).clamp_min(1e-10))
    alignment = fallback_alignment(frame_times, f0, energy_db)

    accent = torch.zeros_like(energy_db)
    sigma_frames = max(0.055 * sample_rate / hop_length, 1.0)
    for onset_time in alignment.onset_times:
        center = int(torch.argmin((frame_times - onset_time).abs()))
        distance = (
            torch.arange(
                energy_db.numel(),
                device=energy_db.device,
                dtype=energy_db.dtype,
            )
            - center
        )
        accent = torch.maximum(
            accent,
            torch.exp(-0.5 * (distance / sigma_frames).square()),
        )
    gates = torch.stack(
        [
            accent * alignment.active_mask.to(dtype=accent.dtype),
            alignment.active_mask.to(dtype=accent.dtype),
            alignment.stable_vowel_mask.to(dtype=accent.dtype),
        ],
        dim=0,
    )
    if gates.size(-1) != target_length:
        gates = F.interpolate(
            gates[None],
            size=target_length,
            mode="nearest",
        )[0]
    return gates


def apply_accent_gate_mode(gates, mode="positive", shift_frames=4):
    if mode == "positive":
        return gates
    if mode != "biphasic":
        raise ValueError(f"Unknown accent gate mode: {mode}")
    if gates.size(-2) != len(AXIS_NAMES):
        raise ValueError(f"Expected [...,3,T] gates, got {tuple(gates.shape)}")

    shift = max(int(shift_frames), 1)
    accent = gates[..., 0, :]
    if accent.size(-1) <= shift:
        return gates
    post_lobe = F.pad(accent[..., :-shift], (shift, 0))
    pre_lobe = F.pad(accent[..., shift:], (0, shift))
    biphasic = post_lobe - pre_lobe
    scale = biphasic.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
    output = gates.clone()
    output[..., 0, :] = biphasic / scale
    return output


def _robust_z(values, median, scale):
    return ((values - median) / max(scale, 1e-6)).clamp(-5.0, 5.0)


def compose_raw_axes(components, f0, stats, residualize_breathiness=True):
    if components.dim() == 2:
        components = components.unsqueeze(0)
        f0 = f0.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    medians = components.new_tensor(
        [stats["components"][name]["median"] for name in COMPONENT_NAMES]
    )[None, :, None]
    scales = components.new_tensor(
        [stats["components"][name]["scale"] for name in COMPONENT_NAMES]
    )[None, :, None]
    z = ((components - medians) / scales.clamp_min(1e-6)).clamp(-5.0, 5.0)

    configured_weights = stats.get("axis_component_weights", AXIS_COMPONENT_WEIGHTS)
    accent_weights = components.new_tensor(configured_weights["accent"])[None, :, None]
    intensity_weights = components.new_tensor(configured_weights["intensity"])[None, :, None]
    breath_weights = components.new_tensor(configured_weights["breathiness"])[None, :, None]

    accent = (z[:, 0:4] * accent_weights).sum(dim=1)
    intensity = (z[:, 4:6] * intensity_weights).sum(dim=1)
    breath_parts = torch.stack([-z[:, 6], -z[:, 7], z[:, 8], z[:, 9]], dim=1)
    breathiness = (breath_parts * breath_weights).sum(dim=1)

    if residualize_breathiness:
        regression = stats.get("breathiness_residual", {})
        coefficients = components.new_tensor(
            regression.get("coefficients", [0.0, 0.0])
        )
        log_f0 = torch.log2(f0.clamp_min(1.0))
        predictors = regression.get("predictors", [])
        if predictors == ["intensity_emphasis", "log_f0"]:
            breathiness = (
                breathiness
                - coefficients[0]
                - coefficients[1] * z[:, 5]
                - coefficients[2] * log_f0
            )
        elif coefficients.numel() == 2:
            breathiness = breathiness - coefficients[0] - coefficients[1] * log_f0
        else:
            # Backward compatibility for early smoke statistics only.
            breathiness = (
                breathiness
                - coefficients[0]
                - coefficients[1] * intensity
                - coefficients[2] * log_f0
            )

    raw_axes = torch.stack([accent, intensity, breathiness], dim=1)
    return raw_axes.squeeze(0) if squeeze else raw_axes


def compose_axis_targets(components, masks, f0, stats):
    if components.dim() == 2:
        components = components.unsqueeze(0)
        masks = masks.unsqueeze(0)
        f0 = f0.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False

    raw_axes = compose_raw_axes(components, f0, stats)
    axis_medians = components.new_tensor(
        [stats["axes"][name]["median"] for name in AXIS_NAMES]
    )[None, :, None]
    axis_scales = components.new_tensor(
        [stats["axes"][name]["scale"] for name in AXIS_NAMES]
    )[None, :, None]
    targets = ((raw_axes - axis_medians) / axis_scales.clamp_min(1e-6)).clamp(-3.0, 3.0)
    targets = targets * masks.to(dtype=targets.dtype)
    return targets.squeeze(0) if squeeze else targets
