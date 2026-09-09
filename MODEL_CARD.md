# Expression-Specific Temporal Support for Controllable Singing Voice Conversion

## Model card

## Model details

The released model is a 383,680-parameter controller attached to a frozen
SEED-VC singing-voice-conversion backbone. The learned controls are Breathiness
and Intensity. The release checkpoint is seed 2027 at fixed step 6000 and
contains neither SEED-VC weights nor optimizer state.

The learned controller changes the aligned 768-dimensional conditioning. The
analytic Vibrato controller instead changes source F0 inside inferred note
sustains. A zero control multiplies each residual by zero before addition.

## Training data

The learned branches used 4,379 training clips: 3,379 GTSinger clips and 1,000
NUS-48E clips, covering English, Korean, and Japanese. The split was
speaker-disjoint from the final held-out test identities.
Dataset audio is not distributed here and remains subject to its source terms.

## Evaluation summary

The objective validation used 12 preselected source-reference pairs and 972
generated WAVs across the released supports and one-axis support interventions.
The released supports showed monotonic target control, while the interventions
produced axis-dependent trade-offs. A separate 12-pair Vibrato localization
study showed that sustain localization reduced off-sustain F0 modification
while retaining 96.96% of the all-voiced response.

These are objective measurements. No human-listening conclusion is bundled
with this release.

## Intended use

- Research on controllable singing voice conversion.
- Consent-based creation and analysis of singing-voice transformations.
- Reproduction of the released controller's inference path.

## Limitations

- The study covers limited corpora, languages, identities, and control levels.
- Temporal-support results are not universal optimality claims.
- Leakage is target-relative; it must not be interpreted as proof that raw
  off-axis acoustic changes decrease.
- Breathiness and Intensity metrics are acoustic proxies, not perceptual ground
  truth.
- Vibrato is additive, positive-only, fixed-rate, and uses heuristic note
  boundaries; explicit glissando exclusion is not implemented.
- The frozen upstream models retain their own limitations and biases.

## Responsible use

Use only voices and recordings for which you have permission. Do not use the
model for impersonation, fraud, harassment, evasion of attribution, or
non-consensual voice cloning. Clearly disclose synthetic or converted audio
where appropriate and comply with applicable law and dataset/model licenses.

## Reproducibility

Use the pinned controller hash in `SHA256SUMS.txt`, the included backbone
configuration, seed 2027, 30 diffusion steps, CFG 0.7, F0 conditioning on, and
automatic F0 adjustment off.
