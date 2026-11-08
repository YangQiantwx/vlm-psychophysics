"""Export the frozen equal-canvas input binding and its 25 new PNGs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from vlm_psychophysics import data, paired_inputs as equal


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_reference(source, published, descriptor):
    payload = source.read_bytes()
    require(
        hashlib.sha256(payload).hexdigest() == descriptor["sha256"],
        "Reference file differs",
    )
    require(
        data.file_sha(published) == descriptor["original_reference"]["sha256"],
        "Source reference file differs",
    )
    with Image.open(source) as canvas, Image.open(published) as original:
        require(
            canvas.mode == original.mode == "RGB" and not canvas.info,
            "Reference RGB8 encoding differs",
        )
        old, new = np.asarray(original), np.asarray(canvas)
    require(
        old.shape == (160, 160, 3) and new.shape == (512, 768, 3),
        "Reference shape differs",
    )
    expected = np.empty_like(new)
    expected[:] = old[0, 0]
    expected[176:336, 304:464] = old
    require(
        np.array_equal(new, expected)
        and hashlib.sha256(new.tobytes()).hexdigest() == descriptor["pixel_sha256"],
        "Reference centering or pixels differ",
    )
    return payload


def export(full, source, out):
    index = data.fetch_release_file(
        source, "dataset-index.json", data.jinwei_release.INDEX_SHA, offline=True
    )
    index_entries = json.loads(index.read_bytes())["scopes"]
    catalogs = {}
    catalog_shas = {}
    for cohort in ("single", "dual"):
        raw = (full / cohort / "catalog.json").read_bytes()
        catalogs[cohort] = json.loads(raw)
        catalog_shas[cohort] = hashlib.sha256(raw).hexdigest()
    stage = {}
    scopes = {}
    static_prompt = None
    for scope in equal.SCOPES:
        domain = scope.split("/")[0]
        base = equal.source_scope(scope)
        parents, _ = data.read_manifest(
            source, base, index_entries[base], "video", offline=True
        )
        inspection, _ = data.read_manifest(
            source, base, index_entries[base], "static", offline=True
        )
        parents_by_event_path = {
            row["images"][1]["path"]: sample for sample, row in inspection.items()
        }
        trials, references, static_sources = {}, {}, {}
        for cohort, catalog in catalogs.items():
            items = catalog["items"]
            associations = [
                row for row in catalog["associations"] if row["domain"] == domain
            ]
            require(len(associations) == 192, f"Wrong {domain} {cohort} cohort size")
            for alias in associations:
                sample = alias["original_video_id"]
                video_id, static_id = alias["video_id"], alias["static_id"]
                video, static = items[video_id], items[static_id]
                parent, still = parents[sample], inspection[sample]
                require(
                    sample not in trials
                    and video["sample_id"] == video_id
                    and static["sample_id"] == static_id
                    and video["source_video_id"] == sample
                    and parent["condition"] == cohort == alias["cohort"]
                    and parent["truth"]
                    == still["truth"]
                    == alias["truth"]
                    == video["row"]["truth"]
                    == static["row"]["truth"]
                    and video["row"]["choices"]
                    == static["row"]["choices"]
                    == parent["choices"]
                    == data.T5_CHOICES,
                    f"Trial, catalog, or GT differs for {sample}",
                )
                video_ref, video_media, video_text = video["content"]
                static_ref, *static_events, static_text = static["content"]
                prompt = static_text["text"]
                require(
                    video_ref == static_ref
                    and video_ref["type"] == "image"
                    and video_media["type"] == "video"
                    and video_media["sha256"] == parent["video_sha256"]
                    and video_text == {"type": "text", "text": parent["prompt"]}
                    and video["row"]["prompt"] == parent["prompt"]
                    and static_text["type"] == "text"
                    and static["row"]["prompt"] == prompt
                    and hashlib.sha256(prompt.encode()).hexdigest()
                    == equal.STATIC_PROMPT_SHA256,
                    f"Native request bytes differ for {sample}",
                )
                static_prompt = prompt
                event_path = (
                    Path(static_events[0]["path"]).relative_to(source).as_posix()
                )
                canonical = parents_by_event_path[event_path]
                require(
                    len(static_events) == len(still["frame_indices"])
                    and still["frame_indices"] == data.static_frame_indices(parent)
                    and [event["frame_index"] for event in static_events]
                    == inspection[canonical]["frame_indices"],
                    f"Event count or chronology differs for {sample}",
                )
                for event, source_event, canonical_event in zip(
                    static_events,
                    still["images"][1:],
                    inspection[canonical]["images"][1:],
                    strict=True,
                ):
                    require(
                        event["type"] == "image"
                        and event["role"] == "event_midpoint"
                        and Path(event["path"]).relative_to(source).as_posix()
                        == canonical_event["path"]
                        and all(
                            event[key] == source_event[key] == canonical_event[key]
                            for key in ("sha256", "pixel_sha256")
                        ),
                        f"Native event bytes differ for {sample}",
                    )
                require(
                    static_sources.setdefault(static_id, canonical) == canonical,
                    "Static canonical source conflict",
                )
                original_path = parent["reference_path"]
                require(
                    video_ref["original_reference"]["sha256"]
                    == parent["reference_sha256"]
                    and video_ref["offset_xy"] == [304, 176]
                    and video_ref["size_wh"] == [768, 512],
                    f"Original reference identity differs for {sample}",
                )
                name = f"{equal.reference_scope(scope)}/{Path(original_path).name}"
                descriptor = dict(
                    path=name,
                    sha256=video_ref["sha256"],
                    pixel_sha256=video_ref["pixel_sha256"],
                )
                if original_path not in references:
                    stage[name] = check_reference(
                        Path(video_ref["path"]), source / original_path, video_ref
                    )
                require(
                    references.setdefault(original_path, descriptor) == descriptor,
                    "Reference binding conflict",
                )
                trials[sample] = [video_id, static_id]
        require(
            trials.keys() == parents.keys() == inspection.keys(),
            f"Original population differs for {scope}",
        )
        require(
            len({pair[0] for pair in trials.values()}) == len(trials),
            f"Duplicate video UID in {scope}",
        )
        require(
            all(trials[sample][1] == uid for uid, sample in static_sources.items()),
            f"Canonical media is outside its alias group in {scope}",
        )
        scopes[scope] = {
            "trials": trials,
            "references": references,
            "static_sources": static_sources,
        }
    require(len(stage) == 25, "Expected exactly 25 new reference PNGs")
    stage[equal.MANIFEST] = (
        data.json_bytes(
            {
                "schema": f"t5-{equal.VERSION}",
                "source": equal.source_binding(),
                "catalog_sha256": catalog_shas,
                "static_prompt": static_prompt,
                "scopes": scopes,
            }
        )
        + b"\n"
    )
    for name, payload in stage.items():
        path = out / name
        require(
            not path.exists() or path.read_bytes() == payload,
            f"Refusing to replace {path}",
        )
    for name, payload in stage.items():
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return {
        "manifest_sha256": data.file_sha(out / equal.MANIFEST),
        "source_revision": data.REVISION,
        "files": len(stage),
        "rows": sum(len(entry["trials"]) for entry in scopes.values()),
        "static_cases": {
            scope: len(entry["static_sources"]) for scope, entry in scopes.items()
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export(args.full.resolve(), args.source.resolve(), args.out.resolve()),
            sort_keys=True,
        )
    )
