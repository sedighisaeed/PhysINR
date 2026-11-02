#!/usr/bin/env python
"""Sandbox adapter executed *inside* an isolated environment by the runners.

This is the child-side counterpart of ``hydro_bench/core/base_runner.py``. The
conda and docker runners invoke it as::

    conda run -n <env> python <method>__adapter.py --input-manifest path.json

It reads the JSON manifest written by the harness, runs the method's own
codebase against every listed input, and writes an ``output_manifest.json``
back that the harness validates. Because it runs inside the method's
environment, it may import any dependency that environment provides --
including the method's own (possibly conflicting) PyTorch build.

A copy of this file is expected at
``hydro_bench/methods/_sandbox/<method>__adapter.py``. Methods that need a
custom entry point (e.g. a specific preprocessing function) vendor their own
copy; the harness reads whichever one exists.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HydroBench sandbox adapter")
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the manifest and exit without running")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Optional sample cap (smoke tests only)")
    return parser.parse_args(argv)


def load_manifest(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"input manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"input manifest is invalid JSON: {exc}") from exc
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise SystemExit(f"unexpected manifest schema_version: "
                         f"{payload.get('schema_version')}")
    return payload


# ------------------------------------------------------------------ adapters
def _read_image(path: Path):
    """Read an image with whatever imaging library the env provides.

    PIL is preferred; OpenCV is the fallback. Both are common enough in deep
    learning environments that one is essentially always present.
    """
    try:
        from PIL import Image
        import numpy as np

        return np.asarray(Image.open(path).convert("RGB"))
    except ImportError:
        pass
    try:
        import cv2
        import numpy as np

        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except ImportError as exc:
        raise SystemExit(
            "neither PIL nor OpenCV is available in this environment"
        ) from exc


def _write_image(path: Path, image) -> None:
    try:
        from PIL import Image
        Image.fromarray(image).save(path)
        return
    except ImportError:
        pass
    import cv2

    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def resolve_algorithm(method: str, params: Dict[str, Any]):
    """Import the method's adapter module from inside its own environment.

    The module path is read from ``configs/methods.yaml``, so the code that
    runs in here is exactly the code the harness would have run in-process.
    """
    module_name = _adapter_module(method) or f"hydro_bench.methods.deep.{_snake(method)}"
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"cannot import adapter for method '{method}' ({module_name}): "
            f"{exc}") from exc
    for attr in dir(module):
        obj = getattr(module, attr)
        if (isinstance(obj, type) and obj.__module__ == module_name
                and getattr(obj, "name", None) == method):
            return obj(**params)
    raise SystemExit(f"adapter module {module_name} has no class for '{method}'")


def _adapter_module(method: str) -> Optional[str]:
    """Look up the adapter path registered for ``method`` in methods.yaml."""
    try:
        from hydro_bench.config_defaults import method_config
        return method_config(method).get("adapter")
    except Exception:
        # configs unavailable in this environment -> fall back to naming
        return None


def _snake(name: str) -> str:
    out = []
    for i, ch in enumerate(name):
        if ch.isupper() and i and not name[i - 1] == "-":
            out.append("_")
        if ch in "-+":
            out.append("_")
        else:
            out.append(ch.lower())
    return "".join(out).strip("_")


# --------------------------------------------------------------------- main
def run(payload: Dict[str, Any], dry_run: bool = False,
        max_images: Optional[int] = None) -> Dict[str, Any]:
    method = payload.get("method", "unknown")
    params = payload.get("params", {}) or {}
    samples = payload.get("samples", [])
    if max_images is not None:
        samples = samples[:max_images]

    output_manifest = {
        "schema_version": SCHEMA_VERSION,
        "method": method,
        "params": params,
        "output_dir": payload.get("output_dir"),
        "samples": [],
        "errors": [],
    }

    if dry_run:
        output_manifest["dry_run"] = True
        return output_manifest

    try:
        algorithm = resolve_algorithm(method, params)
    except SystemExit:
        raise
    except Exception:
        output_manifest["errors"].append(
            f"failed to construct {method}: {traceback.format_exc()}")
        return output_manifest

    if hasattr(algorithm, "warmup"):
        algorithm.warmup()

    for sample in samples:
        sample_id = sample["id"]
        in_path, out_path = Path(sample["input"]), Path(sample["output"])
        try:
            image = _read_image(in_path)
            if image is None:
                raise RuntimeError("could not read input image")
            result = algorithm.enhance(image)
            if result.shape != image.shape:
                raise RuntimeError(
                    f"output shape {result.shape} != input {image.shape}")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            _write_image(out_path, result)
            entry = {"id": sample_id, "input": str(in_path),
                     "output": str(out_path), "shape": list(result.shape)}
        except Exception as exc:  # record per-image failures, keep going
            output_manifest["errors"].append(f"{sample_id}: {exc}")
            entry = {"id": sample_id, "input": str(in_path),
                     "output": str(out_path), "error": str(exc)}
        output_manifest["samples"].append(entry)

    if hasattr(algorithm, "cleanup"):
        algorithm.cleanup()
    return output_manifest


def write_output_manifest(payload: Dict[str, Any], manifest: Dict[str, Any]) -> Path:
    in_path = Path(payload.get("_manifest_path", "input_manifest.json"))
    out_path = in_path.parent / "output_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_path


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    payload = load_manifest(args.input_manifest)
    payload["_manifest_path"] = str(args.input_manifest)
    manifest = run(payload, dry_run=args.dry_run, max_images=args.max_images)
    write_output_manifest(payload, manifest)
    n_ok = sum(1 for s in manifest["samples"] if "error" not in s)
    print(f"[hydro-bench sandbox] {manifest['method']}: "
          f"{n_ok}/{len(manifest['samples'])} images restored", file=sys.stderr)
    if manifest["errors"]:
        print(f"[hydro-bench sandbox] {len(manifest['errors'])} errors",
              file=sys.stderr)
    return 1 if manifest["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
