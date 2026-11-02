"""Generate the full-890 UIEB manifest.

The canonical T90 subset used throughout the underwater literature is not
recoverable from the UIEB distribution: the 90 image ids are not shipped with
it. Rather than invent a split and label it T90, the whole 890-image reference
set is evaluated as one split -- well-defined, larger, and identical for every
method.
"""

from __future__ import annotations

import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_DIR = PACKAGE_ROOT / "hydro_bench" / "datasets" / "manifests"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def main() -> int:
    src = MANIFEST_DIR / "uieb.json"
    base = json.loads(src.read_text(encoding="utf-8"))
    root = Path(base["root"])
    if not root.is_absolute():
        root = PACKAGE_ROOT / root

    refs = sorted(
        p.name for p in (root / "reference_imgs").iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES)
    raws = sorted(
        p.name for p in (root / "raw_input_imgs").iterdir()
        if p.suffix.lower() in IMAGE_SUFFIXES)

    # Every reference image must have a raw counterpart, or the paired loader
    # would score against a ground truth the harness could not locate.
    missing = [r for r in refs if r not in set(raws)]
    if missing:
        raise SystemExit(
            f"{len(missing)} reference images have no raw counterpart "
            f"(e.g. {missing[:3]}); the full-set manifest would be unusable")

    # The UIEB manifest already carries a sha256 for every shipped file; reuse
    # those rather than re-hashing 890 images. Verification keys on the
    # reference images this split actually scores.
    base_files = base.get("files", {})
    known = {k: v for k, v in base_files.items()
             if k.startswith("reference_imgs/")}
    unknown = [n for n in refs
               if f"reference_imgs/{n}" not in known]
    if unknown:
        raise SystemExit(
            f"{len(unknown)} reference images have no checksum in the base "
            f"manifest (e.g. {unknown[:3]}); verify the UIEB checkout")

    manifest = {
        "schema_version": base["schema_version"],
        "dataset": "UIEB-890",
        "root": base["root"],
        "created_utc": base["created_utc"],
        # One split over the entire reference set. Named "test" so the existing
        # CLI and loader paths apply unchanged.
        "splits": {"test": refs},
        "files": known,
    }
    out = MANIFEST_DIR / "uieb_full890.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out.relative_to(PACKAGE_ROOT)}: {len(refs)} reference images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
