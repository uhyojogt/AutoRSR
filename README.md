# AutoRSR: Segmentation Inference

Segmentation-only inference code for **AutoRSR (automated retinal surgery recommendation)**.
The model is a U-Net-like segmentation network with a ResNet-34 encoder and
domain-specific batch normalization (DSBN).

This repository includes the model definition, checkpoint loading, image
preprocessing and segmentation-mask export. It does **not** contain training
code, anatomical analysis, surgical-recommendation rules, patient data or
pretrained weights.

## Installation

Python 3.10 or 3.11 is recommended. Install a PyTorch build appropriate for
your CPU or CUDA environment, following the [official installation instructions](https://pytorch.org/get-started/locally/), then:

```bash
git clone https://github.com/uhyojogt/AutoRSR.git
cd AutoRSR
python -m pip install -e .
```

Alternatively, run from this directory after installing `requirements.txt`.
No pretrained parameters or images are downloaded automatically.

## Checkpoint

A compatible trained AutoRSR checkpoint is required. **Weights are not bundled
with this release.** Use the exported segmentation state dictionary from the
global EMA teacher, not an optimizer checkpoint or a single domain-specific
teacher.

Supported formats:

- A raw segmentation `state_dict`.
- A state dictionary inside `state_dict`, `model_state_dict` or `model`.
- A full DSBN-boundary checkpoint with `backbone.*` and `boundary_*` parameters.
  Only the segmentation backbone is used at inference.
- DataParallel keys prefixed with `module.`.

Loading is strict: missing, incompatible or unexpected segmentation parameters
raise an error. Plain shared-BatchNorm checkpoints and SegFormer checkpoints
are not supported by this release.

Recent PyTorch versions use weights-only loading by default in this code. On
older versions without that facility, upgrade PyTorch or explicitly pass
`--trust-checkpoint` **only for a checkpoint from a trusted source**. Unrestricted
PyTorch checkpoint loading can execute code.

## Run Inference

Put images in one directory. Supported extensions are JPG, JPEG, PNG, BMP, TIF
and TIFF. Filenames must have unique stems across extensions. Ground-truth
masks are not required.

```bash
python -m autorsr \
  --images-dir data/images \
  --checkpoint checkpoints/autorsr_teacher.pth \
  --domain ningbo \
  --device cuda:0 \
  --output outputs/ningbo \
  --save-color \
  --save-overlay
```

For CPU inference, use `--device cpu`. `autorsr-infer` is an equivalent command
after installation. Output directories must be new or empty to prevent
accidental overwriting.

### Normalization Branch

`--domain` is required and selects a fixed branch in the global teacher:

| Images | Argument | Domain ID |
| --- | --- | --- |
| Ningbo | `--domain ningbo` | 0 |
| Daqing | `--domain daqing` | 1 |
| Japanese | `--domain japan` | 2 |

For a previously unseen center, choose the intended existing branch explicitly.
The external Xi'an and Hunan evaluation used the Ningbo branch. The code does
not infer domains from directory names or create new normalization statistics.
For mixed-center data, run each center separately with its selected branch.

### Preprocessing and Output

The default network input is **800 pixels high by 1024 pixels wide**. Images
are converted from BGR to RGB, resized with nearest-neighbor interpolation,
converted to float32 and divided by 255. No ImageNet mean/standard-deviation
normalization or test-time augmentation is applied. These steps preserve the
original test implementation.

The class argmax is computed at network resolution, then resized back to each
image's native dimensions with nearest-neighbor interpolation.

| Pixel value | Class | Color in optional visualizations |
| --- | --- | --- |
| 0 | Background | Black |
| 1 | Retinal detachment | Blue |
| 2 | Retinal break | Green |
| 3 | Optic disc | Red |

```text
outputs/ningbo/
  masks/          # Native-resolution, single-channel PNG labels: 0, 1, 2, 3
  color_masks/    # Optional colorized masks, not suitable as label inputs
  overlays/       # Optional image-mask overlays
  predictions.csv # Filenames, dimensions, selected domain and class pixel counts
  run.json        # Inference settings and checkpoint SHA-256
```

The original input images are never modified. Keep images, generated outputs
and checkpoint files outside version control. The supplied `.gitignore`
excludes these artifacts.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use synthetic tensors and images only, and do not download weights or
require patient data. They cover preprocessing, domain routing, checkpoint
format handling, native-resolution export and output-overwrite protection.

## Scope and License

This is research software, not a clinically validated standalone diagnostic
or treatment-selection product. This release outputs segmentation masks only.

Original project code is provided under the [MIT License](LICENSE).
The ResNet implementation includes building blocks adapted from TorchVision,
whose BSD 3-Clause notice is retained in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Licenses for datasets and any separately supplied checkpoints are not granted
by this code license.
