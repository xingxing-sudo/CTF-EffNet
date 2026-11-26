"""Create the historical ShipsEar directory layout from a release manifest."""

import argparse
import csv
import filecmp
import gzip
from pathlib import Path, PurePosixPath
import shutil


METHODS = ("chirplet", "cqt", "mel", "stft", "wst")
PARTITIONS = ("train", "val", "test", "holdout")
CLASSES = {
    "A_Large_Merchant_Ship", "B_Small_Speedboat", "C_Fishing_Boat",
    "D_Ferry", "E_Ambient_Noise",
}


def prepare_split(manifest, feature_root, split_roots, use_links):
    destinations = {}
    images = set()
    missing = []
    with gzip.open(manifest, "rt", newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            part = row["partition"]
            rel = PurePosixPath(row["relative_image"])
            if (
                part not in PARTITIONS or len(rel.parts) != 2
                or rel.parts[0] not in CLASSES or rel.suffix.lower() != ".png"
                or rel.name in {"", ".", ".."} or "\\" in row["relative_image"]
            ):
                raise ValueError(f"Invalid split row: {row}")
            if rel in images:
                raise ValueError(f"Duplicate split membership: {rel}")
            images.add(rel)
            source = feature_root / rel
            target = split_roots[part] / rel
            if not source.is_file():
                missing.append(source)
            destinations[target] = source

    if missing:
        preview = "\n".join(str(p) for p in missing[:5])
        raise ValueError(f"{len(missing)} feature images are missing, including\n{preview}")

    for root in split_roots.values():
        for entry in root.rglob("*"):
            if entry.is_dir() and entry.is_symlink():
                raise ValueError(f"Unexpected linked output directory: {entry}")
            if entry.is_file() or entry.is_symlink():
                if entry not in destinations or not entry.is_file():
                    raise ValueError(f"Unexpected file in existing split: {entry}")

    for target, source in destinations.items():
        if target.exists() and not filecmp.cmp(source, target, shallow=False):
            raise ValueError(f"Existing file differs from the source: {target}")

    for target, source in destinations.items():
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if use_links:
            target.symlink_to(source)
        else:
            shutil.copyfile(source, target)

    return len(destinations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--link", action="store_true", help="Use symlinks instead of copying images.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = "chirplet_picture_pro" if args.method == "chirplet" else f"{args.method}_picture"
    parent = root / args.method / folder
    feature_root = parent / f"all_{args.method}_picture"
    small = parent / f"{args.method}_shipsear_small_10_train"
    split_roots = {part: small / part for part in PARTITIONS if part != "holdout"}
    split_roots["holdout"] = parent / f"{args.method}_shipsear_small_90_test"
    manifest = root / "splits" / f"{args.method}_small_sample.csv.gz"
    try:
        count = prepare_split(manifest, feature_root, split_roots, args.link)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Verified {count} assigned {args.method} feature images.")


if __name__ == "__main__":
    main()
