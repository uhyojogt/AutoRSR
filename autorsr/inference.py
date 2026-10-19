"""Four-class segmentation with a trained AutoRSR DSBN checkpoint."""
import argparse
from collections.abc import Mapping
import csv
import hashlib
import inspect
import json
from pathlib import Path
import time
import warnings

import cv2
import numpy as np
import torch

from .model import DOMAIN_NAMES, domain_name_to_id, resnet34_dsbn


CLASS_NAMES = ("background", "retinal_detachment", "retinal_break", "optic_disc")
RGB_PALETTE = np.array([[0, 0, 0], [0, 0, 255], [0, 255, 0], [255, 0, 0]], dtype=np.uint8)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def collect_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError("Image directory does not exist: {}".format(directory))
    images = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise ValueError("No supported images found in {}".format(directory))
    stems = [p.stem for p in images]
    if len(stems) != len(set(stems)):
        raise ValueError("Image filenames must have unique stems, including across different extensions.")
    return images


def validate_size(size):
    if len(size) != 2 or any(n < 32 or n % 32 for n in size):
        raise ValueError("Network height and width must be positive multiples of 32.")


def preprocess_image(image_bgr, size=(800, 1024)):
    validate_size(size)
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a nonempty three-channel BGR image.")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)
    # Match the original test pipeline, including its nearest-neighbor image resize.
    array = rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(np.ascontiguousarray(array)).unsqueeze(0)


def segmentation_state_dict(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise ValueError("Expected a tensor state_dict, not a serialized model object.")
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(checkpoint.get(key), Mapping):
            checkpoint = checkpoint[key]
            break
    state = {}
    for key, value in checkpoint.items():
        if not isinstance(key, str) or not torch.is_tensor(value):
            raise ValueError("The selected state_dict must contain only named tensors.")
        normalized = key[len("module."):] if key.startswith("module.") else key
        if normalized in state:
            raise ValueError("Duplicate checkpoint key after prefix removal: " + normalized)
        state[normalized] = value
    if any(key.startswith("backbone.") for key in state):
        auxiliary = [key for key in state if not key.startswith("backbone.")]
        if any(not key.startswith("boundary_") for key in auxiliary):
            raise ValueError("Unexpected non-backbone parameters in the full checkpoint.")
        state = {key[len("backbone."):]: value for key, value in state.items() if key.startswith("backbone.")}
    if not state:
        raise ValueError("Checkpoint contains no segmentation parameters.")
    return state


def load_model(checkpoint, device, trust_checkpoint=False):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise ValueError("Checkpoint does not exist: {}".format(checkpoint))
    options = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        options["weights_only"] = not trust_checkpoint
    elif not trust_checkpoint:
        raise ValueError("This PyTorch version lacks safe weights-only loading. Upgrade PyTorch or use "
                         "--trust-checkpoint only for a checkpoint whose source you trust.")
    if trust_checkpoint:
        warnings.warn("Unrestricted checkpoint loading is enabled. Never use an untrusted checkpoint.", RuntimeWarning)
    state = segmentation_state_dict(torch.load(str(checkpoint), **options))
    model = resnet34_dsbn(4)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def predict_image(model, image_bgr, domain, device, size=(800, 1024)):
    if domain not in DOMAIN_NAMES:
        raise ValueError("Domain must be one of: " + ", ".join(DOMAIN_NAMES))
    tensor = preprocess_image(image_bgr, size).to(device)
    domain_ids = torch.full((1,), domain_name_to_id(domain), dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(tensor, domain_ids=domain_ids)
        if tuple(logits.shape) != (1, 4, size[0], size[1]):
            raise ValueError("Unexpected segmentation output shape: {}".format(tuple(logits.shape)))
        network_mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    height, width = image_bgr.shape[:2]
    return cv2.resize(network_mask, (width, height), interpolation=cv2.INTER_NEAREST)


def write_png(path, array):
    if not cv2.imwrite(str(path), array):
        raise OSError("Could not write image: {}".format(path))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--domain", choices=DOMAIN_NAMES, required=True,
                        help="Explicit normalization branch; it is never inferred from filenames.")
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory.")
    parser.add_argument("--device", default="auto", help="auto, cpu, or a CUDA device such as cuda:0")
    parser.add_argument("--size", nargs=2, type=int, default=(800, 1024), metavar=("HEIGHT", "WIDTH"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--save-color", action="store_true", help="Also save colorized masks.")
    parser.add_argument("--save-overlay", action="store_true", help="Also save image-mask overlays.")
    parser.add_argument("--trust-checkpoint", action="store_true",
                        help="Allow pickle loading only for checkpoints from a trusted source.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    validate_size(args.size)
    if args.threads < 1:
        raise ValueError("--threads must be positive.")
    images = collect_images(args.images_dir)
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        raise ValueError("Refusing to overwrite a nonempty output path: {}".format(args.output))
    torch.set_num_threads(args.threads)
    device = torch.device("cuda:0" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu.")
    model = load_model(args.checkpoint, device, args.trust_checkpoint)
    folders = {"mask": args.output / "masks"}
    if args.save_color:
        folders["color"] = args.output / "color_masks"
    if args.save_overlay:
        folders["overlay"] = args.output / "overlays"
    for directory in folders.values():
        directory.mkdir(parents=True, exist_ok=True)
    rows = []
    start = time.monotonic()
    for index, path in enumerate(images, 1):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image: {}".format(path))
        mask = predict_image(model, image, args.domain, device, tuple(args.size))
        name = path.stem + ".png"
        write_png(folders["mask"] / name, mask)
        if args.save_color or args.save_overlay:
            color = cv2.cvtColor(RGB_PALETTE[mask], cv2.COLOR_RGB2BGR)
            if args.save_color:
                write_png(folders["color"] / name, color)
            if args.save_overlay:
                overlay = image.copy()
                mixed = cv2.addWeighted(image, .65, color, .35, 0)
                overlay[mask > 0] = mixed[mask > 0]
                write_png(folders["overlay"] / name, overlay)
        row = {"image": path.name, "mask": "masks/" + name,
               "height": mask.shape[0], "width": mask.shape[1], "domain": args.domain}
        row.update({name + "_pixels": int(np.count_nonzero(mask == label))
                    for label, name in enumerate(CLASS_NAMES)})
        rows.append(row)
        print("[{}/{}] {}".format(index, len(images), path.name), flush=True)
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = {"classes": dict(enumerate(CLASS_NAMES)), "domain": args.domain,
                "input_size_hw": list(args.size), "images": len(rows),
                "device": str(device), "torch_version": torch.__version__,
                "checkpoint_sha256": digest.hexdigest(), "seconds": time.monotonic() - start,
                "image_interpolation": "nearest", "mask_interpolation": "nearest"}
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Saved {} native-resolution masks to {}".format(len(rows), folders["mask"]))
    return 0
