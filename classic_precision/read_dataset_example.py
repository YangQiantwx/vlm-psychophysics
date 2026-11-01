"""Read one source video by canonical ID, legacy ID, or exact relative path."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def read_sample(root, value=None):
    root = Path(root)
    with (root / "catalog.jsonl").open() as stream:
        rows = [json.loads(line) for line in stream]
    matches = [r for r in rows if value is None or value in
               (r["sample_id"], r["legacy_sample_id"], r["path"], r["legacy_path"],
                r.get("legacy_public_path"), *r.get("alias_paths", []))]
    if value is None:
        matches = [r for r in matches if (root / r["path"]).is_file()][:1]
    if len(matches) != 1:
        raise ValueError("Choose one downloaded sample by ID or exact relative path")
    row = matches[0]
    relative = PurePosixPath(row["path"])
    if relative.is_absolute() or ".." in relative.parts or "\\" in row["path"]:
        raise ValueError("Unsafe media path")
    path = root / row["path"]
    with path.open("rb") as stream:
        actual_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_sha != row["video_sha256"]:
        raise ValueError("Video SHA256 differs")
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        codec = stream.codec_context.name
        pixel_format = "rgb48le" if row["domain"] == "classic" else "rgb24"
        frame_count, first_shape = 0, None
        for frame in container.decode(video=0):
            if first_shape is None:
                first_shape = frame.to_ndarray(format=pixel_format).shape
            frame_count += 1
    if frame_count != row["frames"]:
        raise ValueError("Decoded frame count differs")
    return {"sample_id": row["sample_id"], "legacy_sample_id": row["legacy_sample_id"],
            "path": row["path"], "video_sha256": actual_sha, "frames": frame_count,
            "codec": codec, "first_frame_shape": first_shape}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--id", help="Canonical ID, legacy ID, or exact relative path; defaults to first downloaded video")
    args = parser.parse_args(argv)
    print(json.dumps(read_sample(args.root, args.id), indent=2))


if __name__ == "__main__":
    main()
