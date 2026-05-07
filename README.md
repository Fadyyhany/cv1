# cv1 - Analog Clock Time Recognition (MATLAB)

This repository contains a fully rewritten, modular MATLAB pipeline for analog clock time recognition with **regression-based hand angle prediction**.

## What this pipeline does

- Predicts **hour/minute** hand angles, and **second** hand angle when second labels exist.
- Uses **transfer learning** with a strong pretrained backbone (default: `resnet50`) and staged fine-tuning:
  - Stage 1: frozen backbone, train regression head.
  - Stage 2: unfreeze backbone, low-LR fine-tuning (domain adaptation to real data).
- Applies heavy **domain-generalization augmentations**:
  - rotation (with angle-label correction), affine/perspective transforms,
  - blur, noise, brightness/contrast/color jitter,
  - grayscale conversion, random occlusion,
  - edge blending for background suppression / hand emphasis.
- Includes train/validation split, early stopping, checkpointing, LR scheduling, and mixed precision (when MATLAB version/toolbox supports it).
- Reports angle and time metrics, training curves, and error distributions.
- Evaluates seen vs unseen sets and exports summary artifacts.

## Directory structure

- `/home/runner/work/cv1/cv1/matlab_pipeline/run_clock_pipeline.m` - main entry point
- `/home/runner/work/cv1/cv1/matlab_pipeline/defaultClockConfig.m` - configurable paths/hyperparameters
- `/home/runner/work/cv1/cv1/matlab_pipeline/loadClockDataset.m` - dataset + label-to-angle conversion
- `/home/runner/work/cv1/cv1/matlab_pipeline/augmentClockImage.m` - heavy augmentation policy
- `/home/runner/work/cv1/cv1/matlab_pipeline/createClockRegressionModel.m` - backbone + multi-output head
- `/home/runner/work/cv1/cv1/matlab_pipeline/trainClockRegressor.m` - staged training loop
- `/home/runner/work/cv1/cv1/matlab_pipeline/computeClockMetrics.m` - MAE/angular/time metrics
- `/home/runner/work/cv1/cv1/matlab_pipeline/evaluateClockModel.m` - seen/unseen evaluation + plots
- `/home/runner/work/cv1/cv1/matlab_pipeline/visualizeTrainingHistory.m` - training/validation/lr plots
- `/home/runner/work/cv1/cv1/matlab_pipeline/predictClockTime.m` - inference utility

## MATLAB version

Target: **MATLAB R2025b+** (or closest compatible version with Deep Learning Toolbox and pretrained model support package).

## Configure datasets

Edit paths in `/home/runner/work/cv1/cv1/matlab_pipeline/defaultClockConfig.m`:

- Synthetic dataset image + CSV label path
- Real dataset image + CSV label path
- Optional seen/unseen evaluation sets (`cfg.evaluation.seenSets`, `cfg.evaluation.unseenSets`)

Expected labels: `hour`, `minute`, and optional `second` columns (or first numeric columns fallback).

## Run training

```matlab
cd('/home/runner/work/cv1/cv1/matlab_pipeline');
artifacts = run_clock_pipeline();
```

Optional custom config override:

```matlab
cfg = struct();
cfg.model = struct('backbone', 'resnet50');
cfg.training = struct('stage1Epochs', 15, 'stage2Epochs', 12);
artifacts = run_clock_pipeline(cfg);
```

## Evaluate seen vs unseen cases

The pipeline automatically evaluates:

- `seen_validation`: validation split from synthetic set
- `unseen_validation`: validation split from real set
- Additional configured seen/unseen case folders (if provided in config)

Saved artifacts are written under:

- `/home/runner/work/cv1/cv1/artifacts/models`
- `/home/runner/work/cv1/cv1/artifacts/figures`
- `/home/runner/work/cv1/cv1/artifacts/checkpoints`
- `/home/runner/work/cv1/cv1/artifacts/evaluation_summary_*.csv`

## Design recommendations

- **Architecture**: `resnet50` is a strong and stable baseline in MATLAB for transfer learning.
- **Loss**: angle regression with circular-error metrics avoids discrete-time class bias.
- **Augmentation**: keep aggressive augmentation enabled to reduce synthetic shape/background overfitting.
- **Evaluation**: prioritize unseen-case time error and failure-case visuals, not only training loss.
- **Generalization-first**: avoid tuning for synthetic-only accuracy; use real-set fine-tuning and unseen-case metrics as primary selection criteria.
