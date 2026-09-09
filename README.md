# Expression-Specific Temporal Support for Controllable Singing Voice Conversion

This branch is the inference-only ICASSP 2027 research release accompanying
*Expression-Specific Temporal Support for Controllable Singing Voice Conversion*. It
adds lightweight control to the frozen 44.1 kHz SEED-VC singing-voice-conversion
path. The repository provides one fixed release checkpoint: training seed 2027
at step 6000, selected by a predeclared convention rather than test-set score.

The released controls are:

- Breathiness: a 3-frame local controller applied on stable voiced frames.
- Intensity: local conditioning plus the mean condition of each contiguous
  active singing region, applied on that active region.

Vibrato is a separate positive-only analytic F0 controller. It applies a 5 Hz,
60-cent sinusoid inside inferred note-sustain regions. Zero control is an exact
bypass at the controller/F0 level.

## What is included

- `inference_expr.py`: SEED-VC inference with the final controller hook.
- `modules/expression/temporal_context_adapter.py`: expression-specific
  temporal controller.
- `checkpoints/expression_temporal_controller_seed2027.pth`: controller-only
  weights (383,680 parameters; no SEED-VC weights and no optimizer state).
- `checkpoints/frozen_formula_vibrato_f0_adapter.pth`: analytic Vibrato
  parameters.
- `configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml`: frozen
  SEED-VC configuration.
- `configs/expression_temporal_release.yaml`: release manifest and defaults.

The repository deliberately excludes datasets, evaluation WAVs, experiment
trees, listening-study data, and third-party pretrained weights.

## Environment

Python 3.10 and an NVIDIA CUDA GPU are recommended. The validation environment
used PyTorch 2.4.0+cu124, CUDA 12.4, and Python 3.10.14.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements-release.txt
```

On Windows PowerShell, activate with `.venv\\Scripts\\Activate.ps1` and use
the same installation commands without the line-continuation backslash.

## Required pretrained resources

This release does not redistribute the frozen backbone. The first online run with
`--local-files-only false` obtains the resources used by SEED-VC:

- `Plachta/Seed-VC`:
  `DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth`
- `openai/whisper-small`
- `lj1995/VoiceConversionWebUI`: `rmvpe.pt`
- CAMPPlus: `campplus_cn_common.bin`
- `nvidia/bigvgan_v2_44khz_128band_512x`

Review each upstream model card and license before use. After caching, pass
`--local-files-only true` for offline inference.

## Inference

Provide a monophonic singing source and a 1--30 second reference voice WAV.
Controls are in `[-1, 1]` for Breathiness and Intensity. Vibrato is
positive-only in `[0, 1]`. These are the only expression controls exposed by
the release.

```bash
bash examples/infer_expression_control.sh source.wav reference.wav outputs \
  0.5 0.0 0.0
```

The final three arguments are Breathiness, Intensity, and Vibrato. For a
neutral controller pass `0 0 0`. The example uses diffusion seed 2027, 30
steps, CFG 0.7, source F0 conditioning, and no automatic F0 shift.

For an offline cached run, edit the example to use `--local-files-only true`.
You may pass explicit local resources with `--rmvpe-checkpoint`,
`--campplus-checkpoint`, `--bigvgan-model`, and `--whisper-model`; the last two
accept local pretrained-model directories.
Verify downloaded and bundled artifacts against `SHA256SUMS.txt` where a hash
is listed.

## Scope and limitations

The objective results establish controllability and target-relative
selectivity on the documented multilingual validation design. They do not
establish perceptual superiority, universal best temporal support, or reduced
absolute off-axis acoustic change. Listening evaluation is not included in
this release. See `MODEL_CARD.md` for intended use, limitations, and voice-use
safety notes.

## License and attribution

This derivative code release follows the repository's GNU GPL v3 license; see
`LICENSE`. Third-party code and model resources retain their own licenses and
are not relicensed here. See `THIRD_PARTY_NOTICES.md`.
