#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 6 ]; then
  echo "usage: $0 SOURCE.wav REFERENCE.wav OUTPUT_DIR BREATHINESS INTENSITY VIBRATO" >&2
  exit 2
fi

python inference_expr.py \
  --source "$1" \
  --target "$2" \
  --output "$3" \
  --f0-condition true \
  --auto-f0-adjust false \
  --diffusion-steps 30 \
  --inference-cfg-rate 0.7 \
  --seed 2027 \
  --deterministic true \
  --fp16 true \
  --local-files-only false \
  --use-hierarchical-adapter true \
  --hierarchical-checkpoint checkpoints/expression_temporal_controller_seed2027.pth \
  --hierarchical-breathiness-control "$4" \
  --hierarchical-intensity-control "$5" \
  --use-vibrato-f0-adapter true \
  --vibrato-f0-checkpoint checkpoints/frozen_formula_vibrato_f0_adapter.pth \
  --vibrato-temporal-gate-mode sustain \
  --vibrato-control "$6"
