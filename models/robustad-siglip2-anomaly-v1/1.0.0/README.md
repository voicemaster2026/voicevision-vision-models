# RobustAD SigLIP2 anomaly candidate 1.0.0

Three subset-specific linear heads classify an image as normal or anomalous for
MetalParts, PCB, or PiledBags. The frozen image encoder is the pinned public
Google SigLIP2 Base Patch16 224 model; its weights are fetched from Hugging Face
by the inference script and are not duplicated in this repository.

## Provenance and weights

- Trained checkpoint: `model.pt`, 21,757 bytes, SHA-256
  `4bd7a63c47ae3829b145288175f11c14b0304104a4aaad0851916470c284fcac`.
- Dataset: AmazonScience/RobustAD revision
  `82c642b2d18be60f2dab104f8dd4dea1f4ccfacb`, licensed CC BY 4.0. Attribution:
  AmazonScience, “RobustAD: A Realworld Anomaly Detection Dataset for
  Robustness.” The dataset card notes that the formal paper citation is
  forthcoming. No source images are redistributed here.
- Base model: `google/siglip2-base-patch16-224` revision
  `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, Apache-2.0.
- `dataset-source-manifest.json` contains the pinned dataset's file paths,
  sizes, and SHA-256 values. Private bucket paths have been omitted.
- `backbone-source-manifest.json` lists all files required by the pinned
  SigLIP2 snapshot, with their sizes and SHA-256 values.
- `training_manifest.json` records the source-domain train/validation split and
  the held-out target domains. `evaluation.json` records the measured results.

## Evaluation and limits

Validation macro F1 is 0.924 on 148 source-domain validation images drawn from
741 total source-domain train/validation images. Test
macro F1 is 0.560 over 3,631 held-out images across 19 target domains. The
source-domain validation split is stratified by image and is not grouped by
capture session. Official target-domain test images were not used to train or
select the heads.

This model does not localize defects, and the scores are not calibrated. The
results do not establish performance on VoiceVision customer products or
production lines. Keep it experimental and do not use it as a standalone
accept/reject decision.

## Reproduction

The trainer is `tools/train_robustad_siglip2.py` at the repository root. It
expects the full pinned RobustAD snapshot, a local snapshot of the pinned
SigLIP2 backbone, and a dataset source manifest with this model's dataset
entry. Its training dependencies are in `requirements-robustad.txt`.
