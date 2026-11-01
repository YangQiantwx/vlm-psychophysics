"""Download public T1–T3 folders with a pinned Hugging Face snapshot.

This file also runs on its own: python download_dataset.py --help.
Only huggingface_hub is required; it never imports model runtimes.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath

REPOSITORIES = {
    ("classic", None): "DrQY/vlm-psychophysics-classic-t1-t3",
    **{("natural", task): f"DrQY/vlm-psychophysics-natural-{task.lower()}"
       for task in ("T1", "T2", "T3")},
}
SUPPORT_FILES = ("README.md", "catalog.jsonl", "metadata.jsonl", "aliases.csv",
                 "source-assets.json", "metadata-demo.csv", "download_dataset.py",
                 "read_example.py", "requirements-download.txt", "release.json",
                 "LICENSE", "LICENSES/*", "attribution.json", "validation/*")


def relative_folder(value):
    value = value.rstrip("/")
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("Folder must be a repository-relative path")
    if any(c in value for c in "*?[]"):
        raise ValueError("Specify a folder, not a wildcard pattern")
    if any(part.lower() in ("results", "predictions", "t4", "t5") for part in path.parts):
        raise ValueError("Current download scope contains T1–T3 source data only")
    return path.as_posix()


def selections(domain, task=None):
    if domain == "classic":
        return [(REPOSITORIES[(domain, None)], "classic", task)]
    return [(REPOSITORIES[(domain, t)], f"natural/{t}", t)
            for t in ((task,) if task else ("T1", "T2", "T3"))]


def download(domain, task, folder, revision, output, workers=8):
    from fnmatch import fnmatchcase
    from huggingface_hub import HfApi, snapshot_download

    if workers < 1:
        raise ValueError("workers must be positive")
    if folder is not None:
        folder = relative_folder(folder)
        if domain == "natural":
            prefix = folder.partition("/")[0]
            if prefix not in ("T1", "T2", "T3"):
                raise ValueError("Natural folder must start with T1, T2, or T3 to choose its repository")
            task = prefix
    reports = []
    for repo_id, suffix, selected_task in selections(domain, task):
        api = HfApi()
        info = api.repo_info(repo_id, repo_type="dataset", revision=revision)
        commit = info.sha
        files = [item.rfilename for item in info.siblings]
        selected_folder = folder or selected_task
        if selected_folder:
            prefix = selected_folder + "/"
            folder_files = [p for p in files if p.startswith(prefix)]
            if not folder_files:
                raise ValueError(f"No files in {repo_id}@{commit}:{selected_folder}/")
            expected = set(folder_files)
            expected.update(p for p in files if any(fnmatchcase(p, rule) for rule in SUPPORT_FILES))
        else:
            # Current source scope only, even when an older revision has results/.
            expected = {p for p in files if p.startswith(("T1/", "T2/", "T3/"))
                        or any(fnmatchcase(p, rule) for rule in SUPPORT_FILES)}
        destination = Path(output) / suffix
        destination.mkdir(parents=True, exist_ok=True)
        kwargs = dict(repo_id=repo_id, repo_type="dataset", revision=commit,
                      local_dir=str(destination), allow_patterns=sorted(expected),
                      max_workers=workers)
        if "local_dir_use_symlinks" in inspect.signature(snapshot_download).parameters:
            kwargs["local_dir_use_symlinks"] = False
        snapshot_download(**kwargs)
        missing = [p for p in expected if not (destination / p).is_file()]
        if missing:
            raise ValueError(f"Incomplete snapshot: {missing[:3]}")
        verified = 0
        catalog = destination / "catalog.jsonl"
        if catalog.is_file():
            with catalog.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row["path"] not in expected:
                        continue
                    with (destination / row["path"]).open("rb") as media:
                        actual = hashlib.file_digest(media, "sha256").hexdigest()
                    if actual != row["video_sha256"]:
                        raise ValueError(f"Video SHA256 differs: {row['sample_id']}")
                    verified += 1
        receipt = {"repo_id": repo_id, "revision": commit,
                   "folder": selected_folder, "files": len(expected),
                   "videos_sha256_verified": verified, "output": str(destination.resolve())}
        (destination / "download-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        reports.append(receipt)
    return reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=("classic", "natural"), required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--task", choices=("T1", "T2", "T3"))
    group.add_argument("--folder", help="Repository-relative folder, copied in full")
    parser.add_argument("--revision", required=True, help="Commit SHA or ref; resolved and recorded as a commit")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    print(json.dumps(download(args.domain, args.task, args.folder, args.revision,
                              args.output, args.workers), indent=2))


if __name__ == "__main__":
    main()
