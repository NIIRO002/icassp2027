import os
import glob
import json

import numpy as np

os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import shutil
import warnings
import argparse
import torch
import yaml

warnings.simplefilter('ignore')

# load packages
import random

from modules.commons import *
import time

import torchaudio
from modules.commons import str2bool
from modules.audio import mel_spectrogram
from modules.expression import (
    ExpressionEncoder,
    ExpressionFiLMAdapter,
    ExpressionResidualAdapter,
    ScalarIntensityAdapter,
    SymmetricIntensityAdapter,
    SymmetricVocalEffortAdapter,
    apply_expression_controls,
    extract_expression_features,
    format_expression_stats,
    load_expression_control,
    load_onset_times,
    apply_onset_accent,
    summarize_expression_features,
)
from modules.expression.temporal_context_adapter import (
    TemporalContextExpressionAdapter,
)
from modules.expression.asymmetric_onset_effort_adapter import (
    AsymmetricOnsetEffortAdapter,
    AsymmetricOnsetExpressionAdapter,
)
from modules.expression.range_conditioned_onset_adapter import (
    RangeConditionedOnsetAdapter,
    RangeConditionedExpressionAdapter,
    pitch_range_features,
)
from modules.expression.language_duration_accent_adapter import (
    LanguageDurationAccentAdapter,
    LanguageDurationExpressionAdapter,
)
from modules.expression.hierarchical_features import (
    LANGUAGE_IDS,
    accent_events_from_payload,
    apply_accent_gate_mode,
    build_accent_event_tensors,
    extract_structure_gates,
    replace_accent_gate_with_onsets,
    replace_accent_gate_with_landmarks,
)
from modules.expression.multiscale_adapter import (
    FormulaVibratoF0Adapter,
    VocalRegisterAdapter,
)

from hf_utils import load_custom_model_from_hf


mel_scale_cache = {}
window_cache = {}


def mel_spectrogram_torch(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    key = (sampling_rate, n_fft, num_mels, fmin, fmax, y.device)
    if key not in mel_scale_cache:
        mel_scale_cache[key] = torchaudio.transforms.MelScale(
            n_mels=num_mels,
            sample_rate=sampling_rate,
            f_min=fmin,
            f_max=fmax,
            n_stft=n_fft // 2 + 1,
        ).to(y.device)
    if (win_size, y.device) not in window_cache:
        window_cache[(win_size, y.device)] = torch.hann_window(win_size, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
        mode="reflect",
    ).squeeze(1)
    spec = torch.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_size,
        win_length=win_size,
        window=window_cache[(win_size, y.device)],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    ).abs()
    mel = mel_scale_cache[key](spec)
    return torch.log(torch.clamp(mel, min=1e-5))


def find_cached_file(filename):
    for root in ("checkpoints", os.path.join("checkpoints", "hf_cache")):
        matches = glob.glob(os.path.join(root, "**", filename), recursive=True)
        if matches:
            return matches[0]
    return None


def cached_or_download(repo_id, filename, local_files_only=True):
    cached = find_cached_file(filename)
    if cached:
        return cached
    if local_files_only:
        raise FileNotFoundError(
            f"{filename} was not found in local checkpoints. Run inference once online or pass an explicit path."
        )
    return load_custom_model_from_hf(repo_id, filename, None)


# Load model and configuration
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

fp16 = False
def load_models(args):
    global fp16
    fp16 = args.fp16
    print("Loading SEED-VC checkpoint/config...", flush=True)
    if not args.f0_condition:
        if args.checkpoint is None:
            dit_checkpoint_path = cached_or_download(
                "Plachta/Seed-VC",
                "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
                local_files_only=args.local_files_only,
            )
            dit_config_path = cached_or_download(
                "Plachta/Seed-VC",
                "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
                local_files_only=args.local_files_only,
            )
        else:
            dit_checkpoint_path = args.checkpoint
            dit_config_path = args.config
        f0_fn = None
    else:
        if args.checkpoint is None:
            dit_checkpoint_path = cached_or_download(
                "Plachta/Seed-VC",
                "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
                local_files_only=args.local_files_only,
            )
            dit_config_path = cached_or_download(
                "Plachta/Seed-VC",
                "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
                local_files_only=args.local_files_only,
            )
        else:
            dit_checkpoint_path = args.checkpoint
            dit_config_path = args.config
        if args.skip_f0:
            print("Skipping RMVPE F0 extractor.", flush=True)
            f0_fn = None
        else:
            # f0 extractor
            from modules.rmvpe import RMVPE

            print("Loading RMVPE F0 extractor...", flush=True)
            model_path = args.rmvpe_checkpoint or cached_or_download(
                "lj1995/VoiceConversionWebUI",
                "rmvpe.pt",
                local_files_only=args.local_files_only,
            )
            f0_extractor = RMVPE(model_path, is_half=False, device=device)
            f0_fn = f0_extractor.infer_from_audio

    config = yaml.safe_load(open(dit_config_path, "r"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = 'DiT'
    checkpoint_state = torch.load(dit_checkpoint_path, map_location="cpu")
    cfm_state = checkpoint_state.get("net", {}).get("cfm", {})
    has_wavenet_head = any(key.endswith("estimator.conv1.weight") for key in cfm_state)
    if args.auto_headfix and has_wavenet_head and model_params.DiT.final_layer_type != "wavenet":
        print("Checkpoint uses wavenet final head; overriding config final_layer_type=wavenet.", flush=True)
        model_params.DiT.final_layer_type = "wavenet"
    model = build_model(model_params, stage="DiT")
    hop_length = config["preprocess_params"]["spect_params"]["hop_length"]
    sr = config["preprocess_params"]["sr"]

    # Load checkpoints
    print("Building/loading frozen SEED-VC backbone...", flush=True)
    model, _, _, _ = load_checkpoint(
        model,
        None,
        dit_checkpoint_path,
        load_only_params=True,
        ignore_modules=[],
        is_distributed=False,
    )
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    # Load additional modules
    from modules.campplus.DTDNN import CAMPPlus

    print("Loading CAMPPlus speaker encoder...", flush=True)
    campplus_ckpt_path = args.campplus_checkpoint or cached_or_download(
        "funasr/campplus",
        "campplus_cn_common.bin",
        local_files_only=args.local_files_only,
    )
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_model.load_state_dict(torch.load(campplus_ckpt_path, map_location="cpu"))
    campplus_model.eval()
    campplus_model.to(device)
    print("CAMPPlus loaded.", flush=True)

    vocoder_type = model_params.vocoder.type

    if vocoder_type == 'bigvgan':
        from modules.bigvgan import bigvgan
        bigvgan_name = args.bigvgan_model or model_params.vocoder.name
        print("Loading BigVGAN vocoder...", flush=True)
        bigvgan_model = bigvgan.BigVGAN.from_pretrained(
            bigvgan_name,
            use_cuda_kernel=False,
            local_files_only=args.local_files_only,
        )
        # remove weight norm in the model and set to eval mode
        print("Removing BigVGAN weight norm...", flush=True)
        bigvgan_model.remove_weight_norm()
        bigvgan_model = bigvgan_model.eval().to(device)
        print("BigVGAN loaded.", flush=True)
        vocoder_fn = bigvgan_model
    elif vocoder_type == 'hifigan':
        from modules.hifigan.generator import HiFTGenerator
        from modules.hifigan.f0_predictor import ConvRNNF0Predictor
        hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
        hift_gen = HiFTGenerator(**hift_config['hift'], f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
        hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
        hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
        hift_gen.eval()
        hift_gen.to(device)
        vocoder_fn = hift_gen
    elif vocoder_type == "vocos":
        vocos_config = yaml.safe_load(open(model_params.vocoder.vocos.config, 'r'))
        vocos_path = model_params.vocoder.vocos.path
        vocos_model_params = recursive_munch(vocos_config['model_params'])
        vocos = build_model(vocos_model_params, stage='mel_vocos')
        vocos_checkpoint_path = vocos_path
        vocos, _, _, _ = load_checkpoint(vocos, None, vocos_checkpoint_path,
                                         load_only_params=True, ignore_modules=[], is_distributed=False)
        _ = [vocos[key].eval().to(device) for key in vocos]
        _ = [vocos[key].to(device) for key in vocos]
        total_params = sum(sum(p.numel() for p in vocos[key].parameters() if p.requires_grad) for key in vocos.keys())
        print(f"Vocoder model total parameters: {total_params / 1_000_000:.2f}M")
        vocoder_fn = vocos.decoder
    else:
        raise ValueError(f"Unknown vocoder type: {vocoder_type}")

    speech_tokenizer_type = model_params.speech_tokenizer.type
    if speech_tokenizer_type == 'whisper':
        # whisper
        print("Loading Whisper/content encoder...", flush=True)
        from transformers import AutoFeatureExtractor, WhisperModel
        whisper_name = args.whisper_model or model_params.speech_tokenizer.name
        whisper_model = WhisperModel.from_pretrained(
            whisper_name,
            torch_dtype=torch.float16,
            local_files_only=args.local_files_only,
        ).to(device)
        del whisper_model.decoder
        whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(
            whisper_name,
            local_files_only=args.local_files_only,
        )

        def semantic_fn(waves_16k):
            ori_inputs = whisper_feature_extractor([waves_16k.squeeze(0).cpu().numpy()],
                                                   return_tensors="pt",
                                                   return_attention_mask=True,
                                                   sampling_rate=16000)
            ori_input_features = whisper_model._mask_input_features(
                ori_inputs.input_features, attention_mask=ori_inputs.attention_mask).to(device)
            with torch.no_grad():
                ori_outputs = whisper_model.encoder(
                    ori_input_features.to(whisper_model.encoder.dtype),
                    head_mask=None,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
            S_ori = ori_outputs.last_hidden_state.to(torch.float32)
            S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
            return S_ori
    elif speech_tokenizer_type == 'cnhubert':
        from transformers import (
            Wav2Vec2FeatureExtractor,
            HubertModel,
        )
        hubert_model_name = config['model_params']['speech_tokenizer']['name']
        hubert_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(hubert_model_name)
        hubert_model = HubertModel.from_pretrained(hubert_model_name)
        hubert_model = hubert_model.to(device)
        hubert_model = hubert_model.eval()
        hubert_model = hubert_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy()
                for bib in range(len(waves_16k))
            ]
            ori_inputs = hubert_feature_extractor(ori_waves_16k_input_list,
                                                  return_tensors="pt",
                                                  return_attention_mask=True,
                                                  padding=True,
                                                  sampling_rate=16000).to(device)
            with torch.no_grad():
                ori_outputs = hubert_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    elif speech_tokenizer_type == 'xlsr':
        from transformers import (
            Wav2Vec2FeatureExtractor,
            Wav2Vec2Model,
        )
        model_name = config['model_params']['speech_tokenizer']['name']
        output_layer = config['model_params']['speech_tokenizer']['output_layer']
        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
        wav2vec_model.encoder.layers = wav2vec_model.encoder.layers[:output_layer]
        wav2vec_model = wav2vec_model.to(device)
        wav2vec_model = wav2vec_model.eval()
        wav2vec_model = wav2vec_model.half()

        def semantic_fn(waves_16k):
            ori_waves_16k_input_list = [
                waves_16k[bib].cpu().numpy()
                for bib in range(len(waves_16k))
            ]
            ori_inputs = wav2vec_feature_extractor(ori_waves_16k_input_list,
                                                   return_tensors="pt",
                                                   return_attention_mask=True,
                                                   padding=True,
                                                   sampling_rate=16000).to(device)
            with torch.no_grad():
                ori_outputs = wav2vec_model(
                    ori_inputs.input_values.half(),
                )
            S_ori = ori_outputs.last_hidden_state.float()
            return S_ori
    else:
        raise ValueError(f"Unknown speech tokenizer type: {speech_tokenizer_type}")
    # Generate mel spectrograms
    print("Preparing mel function...", flush=True)
    mel_fn_args = {
        "n_fft": config['preprocess_params']['spect_params']['n_fft'],
        "win_size": config['preprocess_params']['spect_params']['win_length'],
        "hop_size": config['preprocess_params']['spect_params']['hop_length'],
        "num_mels": config['preprocess_params']['spect_params']['n_mels'],
        "sampling_rate": sr,
        "fmin": config['preprocess_params']['spect_params'].get('fmin', 0),
        "fmax": None if config['preprocess_params']['spect_params'].get('fmax', "None") == "None" else 8000,
        "center": False
    }
    to_mel = lambda x: mel_spectrogram(x, **mel_fn_args)
    print("Mel function ready.", flush=True)

    return (
        model,
        semantic_fn,
        f0_fn,
        vocoder_fn,
        campplus_model,
        to_mel,
        mel_fn_args,
    )

def adjust_f0_semitones(f0_sequence, n_semitones):
    factor = 2 ** (n_semitones / 12)
    return f0_sequence * factor

def crossfade(chunk1, chunk2, overlap):
    fade_out = np.cos(np.linspace(0, np.pi / 2, overlap)) ** 2
    fade_in = np.cos(np.linspace(np.pi / 2, 0, overlap)) ** 2
    if len(chunk2) < overlap:
        chunk2[:overlap] = chunk2[:overlap] * fade_in[:len(chunk2)] + (chunk1[-overlap:] * fade_out)[:len(chunk2)]
    else:
        chunk2[:overlap] = chunk2[:overlap] * fade_in + chunk1[-overlap:] * fade_out
    return chunk2


def load_audio_mono(path, sr):
    print(f"Loading audio: {path}", flush=True)
    wav, original_sr = torchaudio.load(path)
    wav = wav.mean(dim=0)
    if original_sr != sr:
        wav = torchaudio.functional.resample(wav[None], original_sr, sr).squeeze(0)
    print(f"Loaded audio shape={tuple(wav.shape)} sr={sr}", flush=True)
    return wav.cpu().numpy()


@torch.no_grad()
def main(args, loaded_models=None):
    if (
        args.accent_control_mode in ("dsp", "effort")
        and abs(args.accent_control) > 1.0
    ):
        raise ValueError(
            "DSP/effort accent control uses the normalized range [-1, 1]"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    if loaded_models is None:
        loaded_models = load_models(args)
    model, semantic_fn, f0_fn, vocoder_fn, campplus_model, mel_fn, mel_fn_args = loaded_models
    sr = mel_fn_args['sampling_rate']
    f0_condition = args.f0_condition
    auto_f0_adjust = args.auto_f0_adjust
    pitch_shift = args.semi_tone_shift

    source = args.source
    target_name = args.target
    diffusion_steps = args.diffusion_steps
    length_adjust = args.length_adjust
    inference_cfg_rate = args.inference_cfg_rate
    source_audio = load_audio_mono(source, sr)
    ref_audio = load_audio_mono(target_name, sr)

    sr = 22050 if not f0_condition else 44100
    hop_length = 256 if not f0_condition else 512
    max_context_window = sr // hop_length * 30
    overlap_frame_len = 16
    overlap_wave_len = overlap_frame_len * hop_length

    # Process audio
    source_audio = torch.tensor(source_audio).unsqueeze(0).float().to(device)
    ref_audio = torch.tensor(ref_audio[:sr * 25]).unsqueeze(0).float().to(device)
    accent_onset_payload = None
    accent_onset_times = None
    if args.accent_onset_json:
        accent_onset_payload, accent_onset_times = load_onset_times(
            args.accent_onset_json
        )

    time_vc_start = time.time()
    # Resample
    print("Resampling source/reference...", flush=True)
    converted_waves_16k = torchaudio.functional.resample(source_audio, sr, 16000)
    # if source audio less than 30 seconds, whisper can handle in one forward
    print("Extracting source content features...", flush=True)
    if converted_waves_16k.size(-1) <= 16000 * 30:
        S_alt = semantic_fn(converted_waves_16k)
    else:
        overlapping_time = 5  # 5 seconds
        S_alt_list = []
        buffer = None
        traversed_time = 0
        while traversed_time < converted_waves_16k.size(-1):
            if buffer is None:  # first chunk
                chunk = converted_waves_16k[:, traversed_time:traversed_time + 16000 * 30]
            else:
                chunk = torch.cat(
                    [buffer, converted_waves_16k[:, traversed_time:traversed_time + 16000 * (30 - overlapping_time)]],
                    dim=-1)
            S_alt = semantic_fn(chunk)
            if traversed_time == 0:
                S_alt_list.append(S_alt)
            else:
                S_alt_list.append(S_alt[:, 50 * overlapping_time:])
            buffer = chunk[:, -16000 * overlapping_time:]
            traversed_time += 30 * 16000 if traversed_time == 0 else chunk.size(-1) - 16000 * overlapping_time
        S_alt = torch.cat(S_alt_list, dim=1)

    ori_waves_16k = torchaudio.functional.resample(ref_audio, sr, 16000)
    print("Extracting reference content features...", flush=True)
    S_ori = semantic_fn(ori_waves_16k)

    print("Computing mel spectrograms...", flush=True)
    mel = mel_fn(source_audio.to(device).float())
    mel2 = mel_fn(ref_audio.to(device).float())

    print("Extracting reference speaker embedding...", flush=True)
    target_lengths = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
    target2_lengths = torch.LongTensor([mel2.size(2)]).to(mel2.device)

    feat2 = torchaudio.compliance.kaldi.fbank(ori_waves_16k,
                                              num_mel_bins=80,
                                              dither=0,
                                              sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = campplus_model(feat2.unsqueeze(0))

    if f0_condition and f0_fn is not None:
        F0_ori = f0_fn(ori_waves_16k[0], thred=0.03)
        F0_alt = f0_fn(converted_waves_16k[0], thred=0.03)

        F0_ori = torch.from_numpy(F0_ori).to(device)[None]
        F0_alt = torch.from_numpy(F0_alt).to(device)[None]

        voiced_F0_ori = F0_ori[F0_ori > 1]
        voiced_F0_alt = F0_alt[F0_alt > 1]

        log_f0_alt = torch.log(F0_alt + 1e-5)
        voiced_log_f0_ori = torch.log(voiced_F0_ori + 1e-5)
        voiced_log_f0_alt = torch.log(voiced_F0_alt + 1e-5)
        median_log_f0_ori = torch.median(voiced_log_f0_ori)
        median_log_f0_alt = torch.median(voiced_log_f0_alt)

        # shift alt log f0 level to ori log f0 level
        shifted_log_f0_alt = log_f0_alt.clone()
        if auto_f0_adjust:
            shifted_log_f0_alt[F0_alt > 1] = log_f0_alt[F0_alt > 1] - median_log_f0_alt + median_log_f0_ori
        shifted_f0_alt = torch.exp(shifted_log_f0_alt)
        if pitch_shift != 0:
            shifted_f0_alt[F0_alt > 1] = adjust_f0_semitones(shifted_f0_alt[F0_alt > 1], pitch_shift)
    else:
        F0_ori = None
        F0_alt = None
        shifted_f0_alt = None

    vibrato_details = None
    if args.use_vibrato_f0_adapter:
        if shifted_f0_alt is None:
            raise ValueError(
                "Vibrato control requires --f0-condition true"
            )
        if not args.vibrato_f0_checkpoint:
            raise ValueError(
                "--vibrato-f0-checkpoint is required when vibrato control is enabled"
            )
        vibrato_state = torch.load(args.vibrato_f0_checkpoint, map_location="cpu")
        vibrato_adapter = FormulaVibratoF0Adapter(
            frame_rate=100.0,
            rate_hz=vibrato_state.get("rate_hz", 5.0),
            positive_depth_cent=(
                args.vibrato_input_depth_cent
                if args.vibrato_input_depth_cent is not None
                else vibrato_state.get("positive_depth_cent", 30.0)
            ),
            negative_suppression_gain=vibrato_state.get(
                "negative_suppression_gain",
                1.0,
            ),
        ).to(device)
        vibrato_adapter.eval()
        vibrato_gate = (
            torch.ones_like(shifted_f0_alt)
            if args.vibrato_temporal_gate_mode == "global"
            else None
        )
        shifted_f0_alt, vibrato_details = vibrato_adapter(
            shifted_f0_alt,
            args.vibrato_control,
            gate=vibrato_gate,
            return_details=True,
        )
        print(
            "Vibrato F0 control: "
            f"control={args.vibrato_control:.3f} "
            f"rate={float(vibrato_details['rate_hz']):.2f}Hz "
            f"depth={float(vibrato_details['depth_cent']):.1f}cent "
            f"gate={float(vibrato_details['gate'].mean()):.3f} "
            f"gate_mode={args.vibrato_temporal_gate_mode}",
            flush=True,
        )

    # Length regulation
    print("Running length regulator...", flush=True)
    cond, _, codes, commitment_loss, codebook_loss = model.length_regulator(S_alt, ylens=target_lengths,
                                                                                       n_quantizers=3,
                                                                                       f0=shifted_f0_alt)
    prompt_condition, _, codes, commitment_loss, codebook_loss = model.length_regulator(S_ori,
                                                                                       ylens=target2_lengths,
                                                                                       n_quantizers=3,
                                                                                       f0=F0_ori)
    if args.use_expression:
        print("Applying expression adapter...", flush=True)
        expression_state = None
        checkpoint_adapter_type = args.adapter_type
        checkpoint_residual_scale = args.residual_scale
        if args.expression_checkpoint:
            expression_state = torch.load(args.expression_checkpoint, map_location="cpu")
            checkpoint_adapter_type = expression_state.get("adapter_type", checkpoint_adapter_type)
            checkpoint_residual_scale = expression_state.get("residual_scale", checkpoint_residual_scale)
        expression_encoder = ExpressionEncoder(
            in_dim=8,
            hidden_dim=args.expression_hidden_dim,
            out_dim=cond.size(-1),
        ).to(device)
        if checkpoint_adapter_type == "film":
            expression_adapter = ExpressionFiLMAdapter(
                cond_dim=cond.size(-1),
                expr_dim=cond.size(-1),
                bottleneck=args.expression_bottleneck,
            ).to(device)
        else:
            expression_adapter = ExpressionResidualAdapter(
                cond_dim=cond.size(-1),
                expr_dim=cond.size(-1),
                bottleneck=args.expression_bottleneck,
                residual_scale=checkpoint_residual_scale,
            ).to(device)
        if expression_state:
            expression_encoder.load_state_dict(expression_state["expression_encoder"])
            expression_adapter.load_state_dict(expression_state["expression_adapter"])
        elif args.expression_debug:
            print("Expression checkpoint: none; adapter is identity-initialized.")
        expression_encoder.eval()
        expression_adapter.eval()

        if args.expression_source == "target":
            expr_audio = ref_audio
            expr_f0 = F0_ori
        else:
            expr_audio = source_audio
            expr_f0 = shifted_f0_alt

        expr_features = extract_expression_features(
            expr_audio,
            sr=sr,
            hop_length=hop_length,
            f0=expr_f0,
            target_len=cond.size(1),
        ).to(device=device, dtype=cond.dtype)
        expression_control = load_expression_control(
            control=args.expression_control,
            control_json=args.expression_control_json,
        )
        cli_control = {
            key: value
            for key, value in {
                "energy": args.energy,
                "vibrato_rate": args.vibrato_rate,
                "vibrato_depth": args.vibrato_depth,
                "breathiness": args.breathiness,
                "brightness": args.brightness,
                "onset_strength": args.onset_strength,
            }.items()
            if value is not None
        }
        if cli_control:
            if expression_control is None:
                expression_control = {"global": cli_control}
            else:
                expression_control = dict(expression_control)
                expression_control.setdefault("global", {}).update(cli_control)
        expr_features = apply_expression_controls(
            expr_features,
            control=expression_control,
            strength=args.expression_control_strength,
            sr=sr,
            hop_length=hop_length,
        )
        expr_emb = expression_encoder(expr_features)
        base_cond = cond
        cond = expression_adapter(cond, expr_emb, strength=args.expression_strength)

        if args.expression_debug:
            print(f"Expression source: {args.expression_source}")
            print(f"Expression adapter type: {checkpoint_adapter_type}")
            print(f"Expression residual scale: {checkpoint_residual_scale}")
            print("Expression features: energy, energy_slope, f0_slope, vibrato_rate, vibrato_depth, breathiness, brightness, onset_strength")
            print(f"Expression control: {expression_control}")
            print(f"cond: {tuple(cond.shape)}")
            print(f"expr_features: {tuple(expr_features.shape)}")
            print(f"expr_emb: {tuple(expr_emb.shape)}")
            delta_abs = (cond - base_cond).detach().float().abs()
            print(
                f"adapter_delta abs_mean={float(delta_abs.mean().cpu()):.6f} "
                f"abs_max={float(delta_abs.max().cpu()):.6f}",
                flush=True,
            )
            print(
                "expr_stats "
                + format_expression_stats(summarize_expression_features(expr_features)),
                flush=True,
            )

    if args.use_intensity_adapter:
        if not args.intensity_checkpoint:
            raise ValueError("--intensity-checkpoint is required when --use-intensity-adapter true")
        print("Applying scalar intensity adapter...", flush=True)
        intensity_state = torch.load(args.intensity_checkpoint, map_location="cpu")
        intensity_adapter_type = intensity_state.get("adapter_type", "scalar_intensity")
        if intensity_adapter_type == "symmetric_linear":
            intensity_adapter = SymmetricIntensityAdapter(
                cond_dim=intensity_state.get("cond_dim", cond.size(-1)),
                residual_scale=intensity_state.get("residual_scale", args.intensity_residual_scale),
            ).to(device)
        else:
            intensity_adapter = ScalarIntensityAdapter(
                cond_dim=intensity_state.get("cond_dim", cond.size(-1)),
                hidden_dim=intensity_state.get("hidden_dim", args.intensity_hidden_dim),
                residual_scale=intensity_state.get("residual_scale", args.intensity_residual_scale),
            ).to(device)
        intensity_adapter.load_state_dict(intensity_state["intensity_adapter"])
        intensity_adapter.eval()
        base_cond = cond
        cond = intensity_adapter(cond, args.intensity_control, strength=args.intensity_strength)
        if args.expression_debug:
            delta_abs = (cond - base_cond).detach().float().abs()
            print(f"Intensity adapter type: {intensity_adapter_type}")
            print(f"Intensity control: {args.intensity_control:+.3f}")
            print(
                f"intensity_delta abs_mean={float(delta_abs.mean().cpu()):.6f} "
                f"abs_max={float(delta_abs.max().cpu()):.6f}",
                flush=True,
            )

    if args.use_effort_adapter:
        if not args.effort_checkpoint:
            raise ValueError("--effort-checkpoint is required when --use-effort-adapter true")
        print("Applying symmetric vocal-effort adapter...", flush=True)
        effort_state = torch.load(args.effort_checkpoint, map_location="cpu")
        effort_adapter = SymmetricVocalEffortAdapter(
            cond_dim=effort_state.get("cond_dim", cond.size(-1)),
            bottleneck=effort_state.get("bottleneck", args.effort_bottleneck),
            residual_scale=effort_state.get("residual_scale", args.effort_residual_scale),
        ).to(device)
        effort_adapter.load_state_dict(effort_state["effort_adapter"])
        effort_adapter.eval()
        base_cond = cond
        cond = effort_adapter(cond, args.effort_control, strength=args.effort_strength)
        if args.expression_debug:
            delta_abs = (cond - base_cond).detach().float().abs()
            print(f"Effort control: {args.effort_control:+.3f}")
            print(
                f"effort_delta abs_mean={float(delta_abs.mean().cpu()):.6f} "
                f"abs_max={float(delta_abs.max().cpu()):.6f}",
                flush=True,
            )

    structure_gates = None
    if args.use_hierarchical_adapter:
        if not args.hierarchical_checkpoint:
            raise ValueError(
                "--hierarchical-checkpoint is required when "
                "--use-hierarchical-adapter true"
            )
        print("Applying hierarchical expression adapter...", flush=True)
        hierarchical_state = torch.load(
            args.hierarchical_checkpoint,
            map_location="cpu",
        )
        adapter_type = hierarchical_state.get("adapter_type")
        is_temporal_context = adapter_type == "temporal_context_expression"
        if not is_temporal_context:
            raise ValueError("Unsupported expression-controller checkpoint")
        if is_temporal_context and args.accent_control != 0.0:
            raise ValueError(
                "The released checkpoint does not expose local emphasis; "
                "--accent-control must be 0."
            )
        global_adapter_kwargs = dict(
            cond_dim=hierarchical_state.get("cond_dim", cond.size(-1)),
            bottleneck=hierarchical_state.get(
                "bottleneck",
                args.hierarchical_bottleneck,
            ),
            residual_scale=hierarchical_state.get(
                "residual_scale",
                args.hierarchical_residual_scale,
            ),
            axis_residual_scales=hierarchical_state.get(
                "axis_residual_scales"
            ),
            combined_residual_scale=hierarchical_state.get(
                "combined_residual_scale"
            ),
            style_dim=(
                hierarchical_state.get("style_dim", style2.size(-1))
                if hierarchical_state.get("style_conditioned", False)
                else None
            ),
        )
        if is_temporal_context:
            global_adapter_kwargs["breathiness_kernel_size"] = (
                hierarchical_state.get("breathiness_kernel_size", 3)
            )
        global_adapter = TemporalContextExpressionAdapter(**global_adapter_kwargs)
        is_asymmetric_onset = hierarchical_state.get("adapter_type") == (
            "asymmetric_onset_effort"
        )
        is_range_conditioned = hierarchical_state.get("adapter_type") == (
            "range_conditioned_onset"
        )
        is_language_duration = hierarchical_state.get("adapter_type") == (
            "language_duration_accent"
        )
        if is_asymmetric_onset:
            onset_adapter = AsymmetricOnsetEffortAdapter(
                cond_dim=hierarchical_state.get("cond_dim", cond.size(-1)),
                bottleneck=hierarchical_state.get("onset_effort_bottleneck", 64),
                component_scales=hierarchical_state.get(
                    "onset_effort_component_scales",
                    (0.08, 0.06, 0.05),
                ),
                combined_residual_scale=hierarchical_state.get(
                    "onset_effort_combined_residual_scale",
                    0.16,
                ),
                control_gain=hierarchical_state.get(
                    "onset_effort_control_gain",
                    2.5,
                ),
                style_dim=(
                    hierarchical_state.get("style_dim", style2.size(-1))
                    if hierarchical_state.get("style_conditioned", False)
                    else None
                ),
            )
            hierarchical_adapter = AsymmetricOnsetExpressionAdapter(
                global_adapter,
                onset_adapter,
            ).to(device)
        elif is_range_conditioned:
            onset_adapter = RangeConditionedOnsetAdapter(
                cond_dim=hierarchical_state.get("cond_dim", cond.size(-1)),
                bottleneck=hierarchical_state.get("onset_effort_bottleneck", 64),
                component_scales=hierarchical_state.get(
                    "range_conditioned_component_scales",
                    (0.08, 0.045),
                ),
                combined_residual_scale=hierarchical_state.get(
                    "range_conditioned_combined_residual_scale",
                    0.12,
                ),
                control_gain=hierarchical_state.get("range_conditioned_control_gain", 2.5),
                style_dim=(
                    hierarchical_state.get("style_dim", style2.size(-1))
                    if hierarchical_state.get("style_conditioned", False)
                    else None
                ),
                range_dim=hierarchical_state.get("range_conditioned_range_dim", 3),
                range_hidden_dim=hierarchical_state.get(
                    "range_conditioned_range_hidden_dim",
                    16,
                ),
                spectral_gain_floor=hierarchical_state.get(
                    "range_conditioned_spectral_gain_floor",
                    0.08,
                ),
                spectral_gain_ceiling=hierarchical_state.get(
                    "range_conditioned_spectral_gain_ceiling",
                    0.75,
                ),
                range_decay=hierarchical_state.get(
                    "range_conditioned_range_decay",
                    1.5,
                ),
                spectral_correction_max=hierarchical_state.get(
                    "range_conditioned_spectral_correction_max",
                    0.08,
                ),
            )
            hierarchical_adapter = RangeConditionedExpressionAdapter(
                global_adapter,
                onset_adapter,
            ).to(device)
        elif is_language_duration:
            accent_adapter = LanguageDurationAccentAdapter(
                cond_dim=hierarchical_state.get("cond_dim", cond.size(-1)),
                bottleneck=hierarchical_state.get("onset_effort_bottleneck", 64),
                component_scales=hierarchical_state.get(
                    "language_duration_component_scales",
                    (0.08, 0.05, 0.04),
                ),
                combined_residual_scale=hierarchical_state.get(
                    "language_duration_combined_residual_scale",
                    0.14,
                ),
                control_gain=hierarchical_state.get("language_duration_control_gain", 2.5),
                style_dim=(
                    hierarchical_state.get("style_dim", style2.size(-1))
                    if hierarchical_state.get("style_conditioned", False)
                    else None
                ),
                event_context_dim=hierarchical_state.get(
                    "language_duration_event_context_dim",
                    4,
                ),
                language_count=hierarchical_state.get("language_duration_language_count", 4),
                language_embedding_dim=hierarchical_state.get(
                    "language_duration_language_embedding_dim",
                    8,
                ),
                context_hidden_dim=hierarchical_state.get(
                    "language_duration_context_hidden_dim",
                    24,
                ),
                range_dim=hierarchical_state.get("range_conditioned_range_dim", 3),
                range_hidden_dim=hierarchical_state.get("range_conditioned_range_hidden_dim", 16),
                spectral_gain_floor=hierarchical_state.get(
                    "range_conditioned_spectral_gain_floor",
                    0.10,
                ),
                spectral_gain_ceiling=hierarchical_state.get(
                    "range_conditioned_spectral_gain_ceiling",
                    0.75,
                ),
                range_decay=hierarchical_state.get("range_conditioned_range_decay", 1.5),
                spectral_correction_max=hierarchical_state.get(
                    "range_conditioned_spectral_correction_max",
                    0.08,
                ),
                context_gain_max=hierarchical_state.get(
                    "language_duration_context_gain_max",
                    0.35,
                ),
                negative_output_gain=(
                    args.language_duration_negative_output_gain
                    if args.language_duration_negative_output_gain is not None
                    else hierarchical_state.get("language_duration_negative_output_gain", 1.0)
                ),
                positive_output_gain=(
                    args.language_duration_positive_output_gain
                    if args.language_duration_positive_output_gain is not None
                    else hierarchical_state.get("language_duration_positive_output_gain", 1.0)
                ),
            )
            hierarchical_adapter = LanguageDurationExpressionAdapter(
                global_adapter,
                accent_adapter,
            ).to(device)
        else:
            hierarchical_adapter = global_adapter.to(device)
        hierarchical_adapter.load_state_dict(
            hierarchical_state["hierarchical_adapter"]
        )
        hierarchical_adapter.eval()
        source_f0_for_gates = (
            shifted_f0_alt[0]
            if shifted_f0_alt is not None
            else torch.zeros(
                max(int(source_audio.size(-1) / sr / 0.01), 1),
                device=device,
            )
        )
        structure_gates = extract_structure_gates(
            source_audio[0],
            sr,
            source_f0_for_gates,
            target_length=cond.size(1),
        )[None].to(device=device, dtype=cond.dtype)
        if accent_onset_times is not None:
            vowel_onset_times = accent_onset_payload.get(
                "vowel_onset_times_seconds",
                accent_onset_payload.get("vowel_onset_times"),
            )
            landmark_mode = hierarchical_state.get(
                "accent_landmark_mode",
                "syllable",
            )
            if landmark_mode == "dual" and isinstance(vowel_onset_times, list):
                structure_gates = replace_accent_gate_with_landmarks(
                    structure_gates,
                    syllable_onset_times=accent_onset_times,
                    vowel_onset_times=vowel_onset_times,
                    duration_seconds=source_audio.size(-1) / sr,
                    vowel_weight=hierarchical_state.get(
                        "vowel_landmark_weight",
                        0.75,
                    ),
                )
            else:
                structure_gates = replace_accent_gate_with_onsets(
                    structure_gates,
                    accent_onset_times,
                    duration_seconds=source_audio.size(-1) / sr,
                )
            print(
                f"Using {len(accent_onset_times)} exact accent onsets from "
                f"{args.accent_onset_json} mode={landmark_mode}",
                flush=True,
            )
        structure_gates = apply_accent_gate_mode(
            structure_gates,
            mode=hierarchical_state.get("accent_gate_mode", "positive"),
            shift_frames=hierarchical_state.get(
                "accent_gate_shift_frames",
                4,
            ),
        )
        component_gates = structure_gates[:, :1].expand(-1, 3, -1).clone()
        event_context = None
        language_ids = None
        if is_language_duration:
            duration_seconds = source_audio.size(-1) / sr
            payload = accent_onset_payload
            if payload is None:
                accent_gate = structure_gates[0, 0].abs()
                pooled = torch.nn.functional.max_pool1d(
                    accent_gate[None, None],
                    kernel_size=7,
                    stride=1,
                    padding=3,
                )[0, 0]
                peak_indices = torch.where(
                    (accent_gate >= pooled - 1e-5) & (accent_gate > 0.65)
                )[0]
                inferred_onsets = [
                    float(index) * duration_seconds / max(cond.size(1), 1)
                    for index in peak_indices.detach().cpu().tolist()
                ]
                payload = {
                    "language": args.source_language,
                    "onset_times_seconds": inferred_onsets,
                    "vowel_onset_times_seconds": inferred_onsets,
                }
                print(
                    "The language-duration controller is using acoustic fallback events; exact alignment "
                    "JSON is recommended for language-aware accent.",
                    flush=True,
                )
            language = str(
                payload.get("language", args.source_language) or "unknown"
            ).lower()
            frame_times = (
                torch.arange(
                    cond.size(1),
                    device=device,
                    dtype=torch.float32,
                )
                + 0.5
            ) * (duration_seconds / max(cond.size(1), 1))
            accent_events = accent_events_from_payload(payload, duration_seconds)
            source_event_mel = torch.nn.functional.interpolate(
                mel.float(),
                size=cond.size(1),
                mode="linear",
                align_corners=False,
            )[0]
            event_gates, event_context_value = build_accent_event_tensors(
                frame_times,
                accent_events,
                language,
                log_mel=source_event_mel,
            )
            active = (structure_gates[0, 1] > 0).to(event_gates)
            component_gates = (event_gates * active[None]).to(cond)[None]
            event_context = event_context_value.to(cond)[None]
            structure_gates[:, 0] = component_gates.amax(dim=1)
            language_ids = torch.tensor(
                [LANGUAGE_IDS.get(language, LANGUAGE_IDS["unknown"])],
                device=device,
                dtype=torch.long,
            )
            enabled_components = set(args.language_duration_accent_components)
            component_mask = cond.new_tensor(
                [
                    name in enabled_components
                    for name in LanguageDurationAccentAdapter.component_names
                ]
            )
            component_gates = component_gates * component_mask[None, :, None]
            print(
                "Language-duration components: "
                + ",".join(sorted(enabled_components))
                + f" language={language} events={len(accent_events)}",
                flush=True,
            )
        if (is_asymmetric_onset or is_range_conditioned) and accent_onset_times is not None:
            syllable_gates = replace_accent_gate_with_onsets(
                structure_gates,
                accent_onset_times,
                duration_seconds=source_audio.size(-1) / sr,
            )
            if isinstance(vowel_onset_times, list):
                vowel_gates = replace_accent_gate_with_onsets(
                    structure_gates,
                    vowel_onset_times,
                    duration_seconds=source_audio.size(-1) / sr,
                )
            else:
                vowel_gates = syllable_gates
            shift = max(
                int(hierarchical_state.get("harmonic_gate_shift_frames", 4)),
                0,
            )
            vowel_gate = vowel_gates[:, 0]
            harmonic_gate = (
                torch.nn.functional.pad(vowel_gate[:, :-shift], (shift, 0))
                if shift > 0 and vowel_gate.size(-1) > shift
                else vowel_gate.clone()
            )
            component_gates = torch.stack(
                (syllable_gates[:, 0], vowel_gate, harmonic_gate),
                dim=1,
            )
        if is_asymmetric_onset:
            enabled_components = set(
                getattr(
                    args,
                    "asymmetric_onset_accent_components",
                    AsymmetricOnsetEffortAdapter.component_names,
                )
            )
            component_mask = cond.new_tensor(
                [
                    name in enabled_components
                    for name in AsymmetricOnsetEffortAdapter.component_names
                ]
            )
            component_gates = component_gates * component_mask[None, :, None]
            print(
                "Asymmetric onset components: "
                + ",".join(sorted(enabled_components)),
                flush=True,
            )
        elif is_range_conditioned:
            component_gates = component_gates[:, :2]
            enabled_components = set(args.range_conditioned_accent_components)
            component_mask = cond.new_tensor(
                [
                    name in enabled_components
                    for name in RangeConditionedOnsetAdapter.component_names
                ]
            )
            component_gates = component_gates * component_mask[None, :, None]
            print(
                "Range-conditioned onset components: "
                + ",".join(sorted(enabled_components)),
                flush=True,
            )
        if (
            args.accent_temporal_gate_mode == "global"
            and (is_asymmetric_onset or is_range_conditioned or is_language_duration)
        ):
            enabled = component_gates.abs().amax(dim=-1, keepdim=True) > 0
            active = (structure_gates[:, 1:2] > 0).to(component_gates)
            component_gates = active.expand_as(component_gates) * enabled.to(
                component_gates
            )
        if is_asymmetric_onset or is_range_conditioned or is_language_duration:
            print(
                f"accent_temporal_gate={args.accent_temporal_gate_mode}",
                flush=True,
            )
        range_features = None
        if is_range_conditioned or is_language_duration:
            source_range_f0 = F0_alt if F0_alt is not None else shifted_f0_alt
            reference_range_f0 = F0_ori
            if source_range_f0 is None:
                source_range_f0 = cond.new_zeros(1, cond.size(1))
            if reference_range_f0 is None:
                reference_range_f0 = cond.new_zeros(1, cond.size(1))
            range_features = pitch_range_features(
                source_range_f0,
                reference_range_f0,
            ).to(device=device, dtype=cond.dtype)
        hierarchical_controls = cond.new_tensor(
            [
                [
                    (
                        args.accent_control
                        if args.accent_control_mode == "latent"
                        else 0.0
                    ),
                    args.hierarchical_intensity_control,
                    args.hierarchical_breathiness_control,
                ]
            ]
        )
        base_cond = cond
        hierarchical_kwargs = {
            "style": (
                style2
                if hierarchical_state.get("style_conditioned", False)
                else None
            ),
            "strength": args.hierarchical_strength,
            "return_details": True,
        }
        if is_temporal_context:
            hierarchical_kwargs["context_gates"] = structure_gates
        if is_asymmetric_onset or is_range_conditioned:
            hierarchical_kwargs["component_gates"] = component_gates
        if is_range_conditioned:
            hierarchical_kwargs["range_features"] = range_features
        elif is_language_duration:
            hierarchical_kwargs["component_gates"] = component_gates
            hierarchical_kwargs["event_context"] = event_context
            hierarchical_kwargs["language_ids"] = language_ids
            hierarchical_kwargs["range_features"] = range_features
        cond, hierarchical_details = hierarchical_adapter(
            cond,
            hierarchical_controls,
            structure_gates,
            **hierarchical_kwargs,
        )
        if (
            not (is_asymmetric_onset or is_range_conditioned)
            and args.accent_control_mode == "effort"
            and args.accent_control != 0.0
        ):
            if accent_onset_times is None:
                raise ValueError(
                    "--accent-onset-json is required for effort accent control"
                )
            onset_effort_gates = torch.zeros_like(structure_gates)
            onset_effort_gates[:, 1] = structure_gates[:, 0]
            onset_effort_controls = cond.new_tensor(
                [
                    [
                        0.0,
                        args.accent_control * args.onset_effort_internal_scale,
                        0.0,
                    ]
                ]
            )
            cond, onset_effort_details = hierarchical_adapter(
                cond,
                onset_effort_controls,
                onset_effort_gates,
                style=(
                    style2
                    if hierarchical_state.get("style_conditioned", False)
                    else None
                ),
                strength=args.hierarchical_strength,
                return_details=True,
            )
            hierarchical_details["onset_effort"] = onset_effort_details
        if args.expression_debug:
            delta_abs = (cond - base_cond).detach().float().abs()
            gate_coverage = structure_gates.detach().float().mean(dim=-1)[0]
            print(
                "Hierarchical controls: "
                f"latent_accent={float(hierarchical_controls[0, 0]):+.3f} "
                f"intensity={args.hierarchical_intensity_control:+.3f} "
                f"breathiness={args.hierarchical_breathiness_control:+.3f}",
                flush=True,
            )
            print(
                "Hierarchical gate coverage: "
                f"accent={float(gate_coverage[0]):.3f} "
                f"intensity={float(gate_coverage[1]):.3f} "
                f"breathiness={float(gate_coverage[2]):.3f}",
                flush=True,
            )
            print(
                f"hierarchical_delta abs_mean={float(delta_abs.mean().cpu()):.6f} "
                f"abs_max={float(delta_abs.max().cpu()):.6f}",
                flush=True,
            )
            if args.accent_control_mode == "effort":
                print(
                    "Onset effort accent: "
                    f"normalized={args.accent_control:+.3f} "
                    f"internal={args.accent_control * args.onset_effort_internal_scale:+.3f}",
                    flush=True,
                )
            if is_range_conditioned:
                print(
                    "Pitch-range conditioning: "
                    f"source={float(range_features[0, 0]):+.3f} "
                    f"reference={float(range_features[0, 1]):+.3f} "
                    f"difference={float(range_features[0, 2]):+.3f} "
                    f"spectral_gain={float(hierarchical_details['onset']['spectral_gain'][0]):.3f}",
                    flush=True,
                )
            elif is_language_duration:
                accent_details = hierarchical_details["accent"]
                print(
                    "Language-duration conditioning: "
                    f"language_id={int(language_ids[0])} "
                    f"attack={float(component_gates[:, 0].mean()):.3f} "
                    f"nucleus={float(component_gates[:, 1].mean()):.3f} "
                    f"sustain={float(component_gates[:, 2].mean()):.3f} "
                    f"spectral_gain={float(accent_details['spectral_gain'][0]):.3f} "
                    f"polarity_gain={float(accent_details['polarity_gain'][0]):.2f}",
                    flush=True,
                )

    if args.use_vocal_register_adapter:
        if not args.vocal_register_checkpoint:
            raise ValueError(
                "--vocal-register-checkpoint is required when register control is enabled"
            )
        register_state = torch.load(
            args.vocal_register_checkpoint,
            map_location="cpu",
        )
        register_adapter = VocalRegisterAdapter(
            cond_dim=register_state.get("cond_dim", cond.size(-1)),
            bottleneck=register_state.get(
                "bottleneck",
                args.vocal_register_bottleneck,
            ),
            residual_scale=register_state.get(
                "residual_scale",
                args.vocal_register_residual_scale,
            ),
            control_gain=register_state.get("control_gain", 2.0),
        ).to(device)
        register_adapter.load_state_dict(
            register_state["vocal_register_adapter"]
        )
        register_adapter.eval()
        if structure_gates is None:
            source_f0_for_gates = (
                shifted_f0_alt[0]
                if shifted_f0_alt is not None
                else torch.zeros(
                    max(int(source_audio.size(-1) / sr / 0.01), 1),
                    device=device,
                )
            )
            structure_gates = extract_structure_gates(
                source_audio[0],
                sr,
                source_f0_for_gates,
                target_length=cond.size(1),
            )[None].to(device=device, dtype=cond.dtype)
        base_cond = cond
        cond, register_details = register_adapter(
            cond,
            args.vocal_register_control,
            gate=structure_gates[:, 2],
            strength=args.vocal_register_strength,
            return_details=True,
        )
        if args.expression_debug:
            delta = (cond - base_cond).detach().float().abs()
            print(
                "Vocal register control: "
                f"value={args.vocal_register_control:+.3f} "
                f"gate={float(structure_gates[:, 2].mean()):.3f} "
                f"delta_mean={float(delta.mean()):.6f} "
                f"delta_max={float(delta.max()):.6f}",
                flush=True,
            )

    # Adapter construction may consume RNG state. Reset immediately before
    # diffusion so neutral and controlled runs share identical initial noise.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    max_source_window = max_context_window - mel2.size(2)
    # split source condition (cond) into chunks
    processed_frames = 0
    generated_wave_chunks = []
    # generate chunk by chunk and stream the output
    while processed_frames < cond.size(1):
        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]
        is_last_chunk = processed_frames + max_source_window >= cond.size(1)
        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)
        print(f"Running diffusion chunk frames={processed_frames}:{processed_frames + chunk_cond.size(1)}...", flush=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):
            # Voice Conversion
            vc_target = model.cfm.inference(cat_condition,
                                                       torch.LongTensor([cat_condition.size(1)]).to(mel2.device),
                                                       mel2, style2, None, diffusion_steps,
                                                       inference_cfg_rate=inference_cfg_rate)
            vc_target = vc_target[:, :, mel2.size(-1):]
        print("Running vocoder...", flush=True)
        vc_wave = vocoder_fn(vc_target.float()).squeeze()
        vc_wave = vc_wave[None, :]
        if processed_frames == 0:
            if is_last_chunk:
                output_wave = vc_wave[0].cpu().numpy()
                generated_wave_chunks.append(output_wave)
                break
            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
        elif is_last_chunk:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            processed_frames += vc_target.size(2) - overlap_frame_len
            break
        else:
            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(),
                                    overlap_wave_len)
            generated_wave_chunks.append(output_wave)
            previous_chunk = vc_wave[0, -overlap_wave_len:]
            processed_frames += vc_target.size(2) - overlap_frame_len
    vc_wave = torch.tensor(np.concatenate(generated_wave_chunks))[None, :].float()
    if args.accent_control_mode == "dsp" and args.accent_control != 0.0:
        if accent_onset_times is None:
            raise ValueError(
                "--accent-onset-json is required for non-zero DSP accent control"
            )
        duration_scale = vc_wave.size(-1) / max(source_audio.size(-1), 1)
        scaled_onsets = [value * duration_scale for value in accent_onset_times]
        vc_wave, accent_details = apply_onset_accent(
            vc_wave,
            sr,
            scaled_onsets,
            args.accent_control,
            max_gain_db=args.dsp_accent_max_gain_db,
            attack_seconds=args.dsp_accent_attack_seconds,
            fade_seconds=args.dsp_accent_fade_seconds,
            preserve_rms=args.dsp_accent_preserve_rms,
            peak_limit=args.dsp_accent_peak_limit,
            return_details=True,
        )
        print(
            "DSP accent: "
            f"control={accent_details['control']:+.3f} "
            f"gain_db={accent_details['gain_db']:+.2f} "
            f"coverage={accent_details['activity_coverage']:.3f} "
            f"limited={accent_details['limited']}",
            flush=True,
        )
    time_vc_end = time.time()
    print(f"RTF: {(time_vc_end - time_vc_start) / vc_wave.size(-1) * sr}")

    source_name = os.path.basename(source).split(".")[0]
    target_name = os.path.basename(target_name).split(".")[0]
    os.makedirs(args.output, exist_ok=True)
    torchaudio.save(os.path.join(args.output, f"vc_{source_name}_{target_name}_{length_adjust}_{diffusion_steps}_{inference_cfg_rate}.wav"), vc_wave.cpu(), sr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="./examples/source/source_s1.wav")
    parser.add_argument("--target", type=str, default="./examples/reference/s1p1.wav")
    parser.add_argument("--output", type=str, default="./reconstructed")
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--length-adjust", type=float, default=1.0)
    parser.add_argument("--inference-cfg-rate", type=float, default=0.7)
    parser.add_argument("--f0-condition", type=str2bool, default=False)
    parser.add_argument("--skip-f0", type=str2bool, default=False)
    parser.add_argument("--auto-f0-adjust", type=str2bool, default=False)
    parser.add_argument("--semi-tone-shift", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, help="Path to the checkpoint file", default=None)
    parser.add_argument("--config", type=str, help="Path to the config file", default=None)
    parser.add_argument(
        "--rmvpe-checkpoint",
        type=str,
        default=None,
        help="Optional explicit rmvpe.pt path (useful for offline inference).",
    )
    parser.add_argument(
        "--campplus-checkpoint",
        type=str,
        default=None,
        help="Optional explicit campplus_cn_common.bin path (useful for offline inference).",
    )
    parser.add_argument(
        "--bigvgan-model",
        type=str,
        default=None,
        help="Optional local BigVGAN model directory (useful for offline inference).",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=None,
        help="Optional local Whisper model directory (useful for offline inference).",
    )
    parser.add_argument("--fp16", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--batch-jobs",
        type=str,
        default=None,
        help="Optional JSON list of per-item overrides; the frozen backbone is loaded once.",
    )
    parser.add_argument("--deterministic", type=str2bool, default=False)
    parser.add_argument("--local-files-only", type=str2bool, default=True)
    parser.add_argument("--auto-headfix", type=str2bool, default=False)
    parser.add_argument("--use-expression", type=str2bool, default=False)
    parser.add_argument("--expression-source", type=str, choices=["source", "target"], default="source")
    parser.add_argument("--expression-strength", type=float, default=1.0)
    parser.add_argument("--expression-control", type=str, default=None)
    parser.add_argument("--expression-control-json", type=str, default=None)
    parser.add_argument("--expression-control-strength", type=float, default=1.0)
    parser.add_argument("--expression-checkpoint", type=str, default=None)
    parser.add_argument("--adapter-type", type=str, choices=["residual", "film"], default="residual")
    parser.add_argument("--expression-hidden-dim", type=int, default=128)
    parser.add_argument("--expression-bottleneck", type=int, default=64)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--energy", type=float, default=None)
    parser.add_argument("--vibrato-rate", type=float, default=None)
    parser.add_argument("--vibrato-depth", type=float, default=None)
    parser.add_argument("--breathiness", type=float, default=None)
    parser.add_argument("--brightness", type=float, default=None)
    parser.add_argument("--onset-strength", type=float, default=None)
    parser.add_argument("--expression-debug", type=str2bool, default=False)
    parser.add_argument("--use-intensity-adapter", type=str2bool, default=False)
    parser.add_argument("--intensity-checkpoint", type=str, default=None)
    parser.add_argument("--intensity-control", type=float, default=0.0)
    parser.add_argument("--intensity-strength", type=float, default=1.0)
    parser.add_argument("--intensity-hidden-dim", type=int, default=64)
    parser.add_argument("--intensity-residual-scale", type=float, default=0.05)
    parser.add_argument("--use-effort-adapter", type=str2bool, default=False)
    parser.add_argument("--effort-checkpoint", type=str, default=None)
    parser.add_argument("--effort-control", type=float, default=0.0)
    parser.add_argument("--effort-strength", type=float, default=1.0)
    parser.add_argument("--effort-bottleneck", type=int, default=64)
    parser.add_argument("--effort-residual-scale", type=float, default=0.1)
    parser.add_argument("--use-hierarchical-adapter", type=str2bool, default=False)
    parser.add_argument("--hierarchical-checkpoint", type=str, default=None)
    parser.add_argument("--accent-control", type=float, default=0.0)
    parser.add_argument(
        "--accent-control-mode",
        type=str,
        choices=["latent", "dsp", "effort", "off"],
        default="latent",
    )
    parser.add_argument(
        "--accent-temporal-gate-mode",
        choices=("event", "global"),
        default="event",
    )
    parser.add_argument("--onset-effort-internal-scale", type=float, default=5.0)
    parser.add_argument(
        "--asymmetric-onset-accent-components",
        nargs="+",
        choices=AsymmetricOnsetEffortAdapter.component_names,
        default=list(AsymmetricOnsetEffortAdapter.component_names),
    )
    parser.add_argument(
        "--range-conditioned-accent-components",
        nargs="+",
        choices=RangeConditionedOnsetAdapter.component_names,
        default=list(RangeConditionedOnsetAdapter.component_names),
    )
    parser.add_argument(
        "--language-duration-accent-components",
        nargs="+",
        choices=LanguageDurationAccentAdapter.component_names,
        default=list(LanguageDurationAccentAdapter.component_names),
    )
    parser.add_argument(
        "--source-language",
        choices=("unknown", "en", "ko", "ja"),
        default="unknown",
    )
    parser.add_argument("--language-duration-negative-output-gain", type=float, default=None)
    parser.add_argument("--language-duration-positive-output-gain", type=float, default=None)
    parser.add_argument("--dsp-accent-max-gain-db", type=float, default=3.0)
    parser.add_argument("--dsp-accent-attack-seconds", type=float, default=0.07)
    parser.add_argument("--dsp-accent-fade-seconds", type=float, default=0.008)
    parser.add_argument("--dsp-accent-preserve-rms", type=str2bool, default=True)
    parser.add_argument("--dsp-accent-peak-limit", type=float, default=0.999)
    parser.add_argument("--hierarchical-intensity-control", type=float, default=0.0)
    parser.add_argument("--hierarchical-breathiness-control", type=float, default=0.0)
    parser.add_argument("--hierarchical-strength", type=float, default=1.0)
    parser.add_argument("--hierarchical-bottleneck", type=int, default=64)
    parser.add_argument("--hierarchical-residual-scale", type=float, default=0.15)
    parser.add_argument("--accent-onset-json", type=str, default=None)
    parser.add_argument("--use-vocal-register-adapter", type=str2bool, default=False)
    parser.add_argument("--vocal-register-checkpoint", type=str, default=None)
    parser.add_argument("--vocal-register-control", type=float, default=0.0)
    parser.add_argument("--vocal-register-strength", type=float, default=1.0)
    parser.add_argument("--vocal-register-bottleneck", type=int, default=64)
    parser.add_argument("--vocal-register-residual-scale", type=float, default=0.2)
    parser.add_argument("--use-vibrato-f0-adapter", type=str2bool, default=False)
    parser.add_argument("--vibrato-f0-checkpoint", type=str, default=None)
    parser.add_argument("--vibrato-control", type=float, default=0.0)
    parser.add_argument("--vibrato-input-depth-cent", type=float, default=None)
    parser.add_argument(
        "--vibrato-temporal-gate-mode",
        choices=("sustain", "global"),
        default="sustain",
        help="Use inferred note sustains or all voiced frames for vibrato control.",
    )
    args = parser.parse_args()
    if args.batch_jobs:
        with open(args.batch_jobs, "r", encoding="utf-8") as handle:
            batch_jobs = json.load(handle)
        if not isinstance(batch_jobs, list) or not batch_jobs:
            raise ValueError("--batch-jobs must contain a non-empty JSON list")
        loaded_models = load_models(args)
        print(f"[batch] jobs={len(batch_jobs)}", flush=True)
        for index, overrides in enumerate(batch_jobs, start=1):
            if not isinstance(overrides, dict):
                raise TypeError(f"Batch job {index} is not an object")
            job_args = argparse.Namespace(**vars(args))
            for key, value in overrides.items():
                if not hasattr(job_args, key):
                    raise ValueError(f"Unknown batch argument: {key}")
                setattr(job_args, key, value)
            print(f"[batch] {index}/{len(batch_jobs)} output={job_args.output}", flush=True)
            main(job_args, loaded_models=loaded_models)
    else:
        main(args)
