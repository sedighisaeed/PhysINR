"""Default configuration paths and metric groups shared by the CLI and API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_ROOT / "configs"

DATASETS_CONFIG = CONFIG_DIR / "datasets.yaml"
METHODS_CONFIG = CONFIG_DIR / "methods.yaml"
DOWNSTREAM_CONFIG = CONFIG_DIR / "downstream.yaml"

# Metrics that require ground truth, and those that do not. Splitting them
# lets the CLI refuse to schedule FR metrics on an unpaired dataset.
DEFAULT_FR_METRICS: List[str] = ["psnr", "ssim", "ms_ssim",
                                 "lpips_alex", "lpips_vgg"]
DEFAULT_NR_METRICS: List[str] = ["uciqe", "uiqm", "uicm", "uism",
                                 "uiconm", "ccf", "brisque", "musiq"]


def load_configs(config_dir: Path = CONFIG_DIR) -> Dict[str, Any]:
    """Read all three YAML configs into one dict."""
    config_dir = Path(config_dir)
    out: Dict[str, Any] = {}
    for key, path in (("datasets", DATASETS_CONFIG),
                      ("methods", METHODS_CONFIG),
                      ("downstream", DOWNSTREAM_CONFIG)):
        path = config_dir / path.name
        if not path.is_file():
            raise FileNotFoundError(f"config file missing: {path}")
        with open(path, encoding="utf-8") as handle:
            # Each file is keyed by its own section name (``datasets:``,
            # ``methods:``); unwrap that level so ``load_configs()["methods"]``
            # is the methods mapping itself rather than a re-wrapped file.
            parsed = yaml.safe_load(handle) or {}
        out[key] = parsed.get(key, parsed)
    return out


def method_config(method: str, config_dir: Path = CONFIG_DIR) -> Dict[str, Any]:
    """Look up one method's entry from ``configs/methods.yaml``."""
    configs = load_configs(config_dir)
    methods = configs.get("methods", {})
    if method not in methods:
        raise KeyError(f"method '{method}' is not in {METHODS_CONFIG}")
    return methods[method]
