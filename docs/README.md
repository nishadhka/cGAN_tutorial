# Documentation index

Reference and design docs for the East Africa cGAN rainfall-postprocessing work.

## Concept & rationale

- [faq-cgan-training-gefs.md](faq-cgan-training-gefs.md) — FAQ on the cGAN
  training rationale: why operational forecasts (IFS/GEFS) are trained on
  instead of ERA5, why postprocessing is model/region/lead-time specific, why
  multiple atmospheric predictors are used, and the variable sets of the two
  reference papers plus a recommended GEFS set.
- [east_africa_cgan_variable_selection_rationale.md](east_africa_cgan_variable_selection_rationale.md)
  — why the Kenya EP-cGAN uses a large conditioning domain (−15..25 N,
  20..53 E), the driver → diagnostic field → ECMWF Open Data variable chain
  (Somali/Turkana jets, IOD, ITCZ), and the final channel set constrained to
  fields reachable from the AWS S3 ECMWF Open Data feed.
- [tf_vs_pytorch_cgan_comparison.md](tf_vs_pytorch_cgan_comparison.md) —
  technical comparison of the TensorFlow tutorial cGAN and the PyTorch EP-cGAN.

## Planning

- [east_africa_kenya_training_plan.md](east_africa_kenya_training_plan.md) —
  domain options, dataset strategy, GPU budget/benchmarks, and the
  pilot-then-scale training plan.

## Workflows & guides

- [cGAN_GEFS_Workflow.md](cGAN_GEFS_Workflow.md) — GEFS cGAN forecast workflow.
- [GEFS_INFERENCE_GUIDE.md](GEFS_INFERENCE_GUIDE.md) — running cGAN inference on
  GEFS.
- [tfrecords_creation_workflow.md](tfrecords_creation_workflow.md) — creating
  TFRecords for cGAN training.
