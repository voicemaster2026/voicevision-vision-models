# Depth Anything V2 Rebar Metric Small 0.1.0

Experimental fine-tuning candidate for metric optical-axis depth on synthetic
rebar imagery. This is a research artifact, not a production model.

## Files

- `model.pt`: fine-tuned candidate checkpoint (SHA-256 is in `artifact.json`).
- `pretrained_baseline.pt`: exact pinned pretrained initialization checkpoint.
- `evaluation.json`: pinned dataset integrity report, split, baseline and
  candidate metrics, and limitations.
- `dataset-manifest.json`: per-file hashes for the pinned CC-BY-4.0 dataset.
- `training.log`: captured training output. It is incomplete and does not
  establish the final epoch or convergence history.

Training and evaluation implementations are in `tools/` at the repository
root. Install `requirements-training.txt` for the training environment. The
upstream Depth Anything V2 source revision and pretrained model revision are
pinned in `evaluation.json` and `artifact.json`.

## Evaluation and limitations

Validation used 336 frames from `lattice_zed_504` and `pile_zed_504`, with
camera-ring groups held out. Candidate validation abs_rel was 0.0400, versus
4.4961 for the pretrained baseline. On the 504-frame held-out synthetic
`pier_504` scene, however, candidate abs_rel was 0.9464 and RMSE was 11.855 m;
the pretrained baseline was also poor (abs_rel 0.7818, RMSE 10.043 m). This
shows that the candidate does not generalize to this different synthetic
scene. See `evaluation.json` for full metrics and exact provenance.

Do not use these weights for physical measurements, production inspection, or
accept/reject decisions. The dataset's physical COLMAP sequence lacks aligned
undistorted RGB in the pinned snapshot, so this artifact has no physical
metrology validation. Dataset content is not redistributed here; the manifest
allows retrieval and integrity verification from its source.

## Licensing

The upstream pretrained model is Apache-2.0. The source dataset is CC-BY-4.0;
users must follow its attribution terms. This repository does not relicense the
upstream model or dataset. See the pinned source links in `artifact.json`.
