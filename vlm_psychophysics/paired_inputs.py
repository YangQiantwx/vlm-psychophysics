"""Bind equal-canvas T5 to the original paired trials and exact reference PNGs."""

from __future__ import annotations

import hashlib
import json

from classic_precision import current_release, jinwei_release


VERSION = "equal-canvas-v1"
MANIFEST = f"t5-{VERSION}/metadata.json"
SCOPES = tuple(f"{scope}/{VERSION}" for scope in current_release.PAIRED_SCOPES)
RELEASE_REVISION = "cdbda2c954c5ac70442e5dbaf8a649a163892d81"
MANIFEST_SHA256 = "635d50bb139e0ae5d05ddf429847bac18a592cc9af5916f33e7d395cf43259d3"
STATIC_PROMPT_SHA256 = (
    "292fe3e35590dc1dc6d90850fbf6dad43df773f68cf03051ed1bf73814e1355b"
)


def source_scope(scope):
    return scope.removesuffix(f"/{VERSION}")


def manifest_record():
    return {
        "path": MANIFEST,
        "sha256": MANIFEST_SHA256,
        "rows": 768,
        "revision": RELEASE_REVISION,
    }


def reference_scope(scope):
    return f"{scope}/references"


def source_binding():
    return {
        "repo_id": current_release.REPO_ID,
        "revision": current_release.REVISION,
        "index_sha256": jinwei_release.INDEX_SHA,
        "manifests": {
            scope: {"video": item[2], "static": item[3]}
            for scope, item in jinwei_release.PAIRED_MANIFESTS.items()
        },
    }


def read_overlay(root, *, offline):
    from . import data

    path = data.fetch_release_file(
        root, MANIFEST, MANIFEST_SHA256, offline=offline, revision=RELEASE_REVISION
    )
    manifest = json.loads(path.read_bytes())
    prompt = manifest["static_prompt"]
    if (
        manifest["schema"] != f"t5-{VERSION}"
        or manifest["source"] != source_binding()
        or set(manifest["scopes"]) != set(SCOPES)
        or hashlib.sha256(prompt.encode("utf-8")).hexdigest() != STATIC_PROMPT_SHA256
    ):
        raise ValueError("Unexpected equal-canvas source or static prompt")
    return manifest


def selection_rows(scope, manifest, parents, inspection):
    from . import data

    entry = manifest["scopes"][scope]
    mapping = entry["trials"]
    references = entry["references"]
    if mapping.keys() != parents.keys() or references.keys() != {
        row["reference_path"] for row in parents.values()
    }:
        raise ValueError(f"Equal-canvas coverage differs for {scope}")
    prompt = manifest["static_prompt"]
    normalized, stills = {}, {} if inspection is not None else None
    for sample, parent in parents.items():
        replacement = references[parent["reference_path"]]
        reference_path = current_release.payload_path(
            reference_scope(scope), replacement["path"]
        )
        video_id, static_id = mapping[sample]
        row = {
            **parent,
            "reference_path": reference_path,
            "reference_sha256": data.require_sha(replacement["sha256"]),
            "reference_pixel_sha256": data.require_sha(replacement["pixel_sha256"]),
            "canonical_video_id": video_id,
        }
        normalized[sample] = row
        if stills is not None:
            old = inspection[sample]
            reference = {"role": "reference", **data.source_reference(scope, row)}
            images = [reference, *old["images"][1:]]
            if (
                old["purpose"] != "stimulus inspection"
                or old["model_evaluated"] is not False
                or any(
                    old["images"][0][key] != value
                    for key, value in data.source_reference(
                        source_scope(scope), parent
                    ).items()
                )
            ):
                raise ValueError(f"Original paired reference differs for {sample}")
            stills[sample] = {
                **old,
                "original_video_ID": sample,
                "static_case_id": static_id,
                "prompt": prompt,
                "choices": data.T5_CHOICES,
                "images": images,
                "model_input_sha256": data.static_identity(
                    scope,
                    "T5",
                    images,
                    old["frame_indices"],
                    prompt,
                    data.T5_CHOICES,
                    sample,
                ),
            }
    return normalized, stills
