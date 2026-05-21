from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model snapshot for local inspection."
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-pattern", action="append", default=None)
    parser.add_argument("--ignore-pattern", action="append", default=None)
    args = parser.parse_args()

    if args.local_dir.exists() and not args.local_dir.is_dir():
        raise NotADirectoryError(f"local-dir is not a directory: {args.local_dir}")
    if args.cache_dir is not None and args.cache_dir.exists() and not args.cache_dir.is_dir():
        raise NotADirectoryError(f"cache-dir is not a directory: {args.cache_dir}")

    args.local_dir.mkdir(parents=True, exist_ok=True)
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=args.local_dir,
        cache_dir=args.cache_dir,
        allow_patterns=args.allow_pattern,
        ignore_patterns=args.ignore_pattern,
        local_dir_use_symlinks=False,
    )

    metadata = {
        "model_id": args.model_id,
        "revision": args.revision,
        "snapshot_path": str(snapshot_path),
        "local_dir": str(args.local_dir.resolve()),
        "cache_dir": None if args.cache_dir is None else str(args.cache_dir.resolve()),
        "allow_patterns": args.allow_pattern,
        "ignore_patterns": args.ignore_pattern,
    }
    metadata_path = args.local_dir / "download_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
