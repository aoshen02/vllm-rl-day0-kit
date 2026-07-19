"""Publish one checkpoint through vLLM's NCCL weight-transfer engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hf_checkpoint_nccl_publisher import (
    CheckpointManifest,
    NcclCheckpointPublisher,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--buffer-size-mb", type=int, default=512)
    parser.add_argument("--update-bucket-size-mb", type=int, default=512)
    parser.add_argument("--packed-num-buffers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--enable-mtp",
        action="store_true",
        help="Send the same canonical checkpoint to the draft model after main.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    buffer_size_bytes = args.buffer_size_mb * 1024**2
    manifest = CheckpointManifest.load(
        model=args.model,
        revision=args.revision,
        checkpoint_path=args.checkpoint_path,
    )
    publisher = NcclCheckpointPublisher(
        base_url=args.base_url,
        manifest=manifest,
        device=args.device,
        buffer_size_bytes=buffer_size_bytes,
        update_bucket_size_bytes=args.update_bucket_size_mb * 1024**2,
        num_buffers=args.packed_num_buffers,
        timeout_seconds=args.timeout,
    )
    try:
        initialization = publisher.initialize()
        updates = [publisher.publish_update(target="main")]
        if args.enable_mtp:
            updates.append(publisher.publish_update(target="draft"))
    finally:
        publisher.shutdown()

    result = {
        "checkpoint": manifest.summary(buffer_size_bytes),
        "initialization": initialization,
        "mtp_enabled": args.enable_mtp,
        "update": updates[0],
        "updates": updates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"send_weights_completed": True}))


if __name__ == "__main__":
    main()
