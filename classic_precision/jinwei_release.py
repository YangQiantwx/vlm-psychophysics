"""Read and verify the frozen media/metadata used by the provider evaluation."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.request import urlopen

from .current_release import PAIRED_SCOPES, REPO_ID as REPO, REVISION

# The private supplement and existing return archives retain the original public pin.
ORIGINAL_REVISION = "20baa977b183c8ecf56b5e4f7da569c5a93e9464"
ORIGINAL_INDEX_SHA = "096a675cfce1beeede63fdd3c1fec0f49a3c5deeafe7956fb9de96ee45897731"
# Keep the previous layout pin available for return archive verification.
PREVIOUS_REVISION = "13e9cf67bb0b63be562ed7b7a48668fa7b7bca22"
PREVIOUS_INDEX_SHA = "47611f9b112b406d7b585eade36ab260491fde2dce341ece13e3ad12b893c7f7"
HF_URL = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/"
INDEX_SHA = "1fc0fff6cc0e724ab083876cac1257e3d79bc1727c4bdec23ee26c82acd4e3ee"
PAIRED_RELEASE_REVISION = "343f6272cdcdd9beabe3d666ba829483ec742467"
PAIRED_INDEX_SHA = "2b1a15160f2323d6bd7447e0a8631304291f99651d4070e40d09bf534032ffcd"
LAYOUT_REVISION = "6b7b66385e21d40aac0664628877b40d4d0ec3dd"
LAYOUT_INDEX_SHA = "7752daa71c3e1fa4bd39478d1fc204cf1e88929d574e35fa16aa20a54e728925"
PRE_LAYOUT_REVISION = "3ede52f068164d2db4889d6a84cb153037da1a45"
FULL_INDEX_SHA = "1fa80aa8c7e644566f7931d1940c38da196086e1e2671997a7dd4cfd616656a6"
FULL_MEDIA_REVISION = "87670fa5101bd9fc9fe05edf3d97c936e832bbb6"
T2_960_REVISION = "7b7beae887c5d9f133744b95c85bc53a2b7fb860"
T2_960_INDEX_SHA = "7be5c91fa7b527640b1210a7fa39f65a39272fd70b138b318a8014cfb6e38fbe"
PRIVATE_SHA = "8eb931fa5c8c0103a6c2efbf9b0f27717147a26632f034eaa8e9f06c29c3f964"
PRIVATE_GENERATION = "1793982719470388"
MIME = {".png": "image/png", ".mp4": "video/mp4", ".mkv": "video/x-matroska"}
# Logical video rows, independent static cases, video manifest SHA, static manifest SHA.
SCOPES = {
    "classic/t1": (
        4224,
        4224,
        "e73b1158c0d05d2091ccb34c4d97a1ed314c2c47db8582f07ce2420de735d36d",
        "d4861a54d4a9a621d166f5b922e5dcf56f446cd3910ccb1f1b275078903a0f6f",
    ),
    "classic/t2": (
        3840,
        160,
        "08dd5864c593ebcc7cf187de55525d911cae682beb6b23b10b1283dcece1eae3",
        "0e3bfdfb47952a8d650841575f9d72d4832473ca4dbd042e03eacb19269a7319",
    ),
    "classic/t3": (
        2736,
        114,
        "c6ee0c9d31629cba3840c004ffbb2d13ef571818832521dc1ff6a4e531fd4a96",
        "7d2bebd9082e4c92654b6fcece52ab8f8865e9cda144a902e029a75c0995e2de",
    ),
    "classic/t4": (
        1536,
        1536,
        "3ede68ef45e0e1c97e1509166937da1b9a7d75e52c6ac0452f5d5b7e37cb34d0",
        "ac409fcbc07cb1cc8b31fb029da40266dfc8fda1a19faece456ece02df831ddc",
    ),
    "classic/t5": (
        384,
        44,
        "835df66999e665081859113b6511804086511a42cb7bbd4ca9c0955756fd0f2c",
        "6f0a3ad4c9debc410f88f041d6e37f59a5cf6a066c21f071b148d203ed661b05",
    ),
    "natural/t1": (
        4224,
        4224,
        "4a736ce7b4b6586c9abccb0b2d323ade9bdb51f069132346b51a431be850333b",
        "a61e0bb411a4f4dff666027e889ccd7983a81fc0e5dd16ad5b6b0a2734b2dec2",
    ),
    "natural/t2": (
        3840,
        3840,
        "af2d5854a9872574d511884b1eb2d9f483a28d821a6acafb9ecf1a8140953138",
        "237540dad127b13e73ee7c591df3cf6df192fad6d3cb13ffd90874d16c864fa8",
    ),
    "natural/t3": (
        2736,
        2736,
        "6f994ca8929368ea3a413c3436873da1e95431fc57a7f223f798293522598cef",
        "e77db842d2914b6684bd57e14d487d5c7554c0bf42c58132dcbe171ab12e2632",
    ),
    "natural/t4": (
        1536,
        1536,
        "5f3359250ff3c437696a41e9592d820932192ea943d2657820367a00063443ae",
        "791e1aee2260c08626dcd3cd88aadbdc5ce19e7f16bb922864074d2a8460bd4f",
    ),
    "natural/t5": (
        384,
        381,
        "cc277a90796c763a1e705433b25e60a5eae4e95f0929e322b57d041c1150f35a",
        "a9226cc417f835aa154ac6d073c3a2203e603c0cd4fbc9b7ede973fd6f67862b",
    ),
}
PREVIOUS_MANIFESTS = {
    "classic/t4": (
        768,
        768,
        "78041f9ba97b9c603c1fc50c77cfdd3834dcb8ceaf0ef6a21704ad698e10c82a",
        "02356da18bf29ba361a2e5f5000a776e38d852ea30415b63356911a4aa3a1638",
    ),
    "classic/t5": (
        96,
        0,
        "f9e9e7ef363d4befeca738165b390a62c9cead779fabb49326538d38cb4a4912",
        "6ffb19e4882e4c20420e1b71517680467f67b140bf23ee4a11ed42bf2c95d5bd",
    ),
    "natural/t5": (
        96,
        0,
        "32c17a6f607189d18ed0d48e77d9cf33b778bd81f235d87a5c1dee60dc652dac",
        "5dbd6d27b6dd0b6a1ed456979bcd5a1cac443561cfc25ac0ecf222f8729e45cb",
    ),
}
PAIRED_MANIFESTS = {
    "classic/t5/paired-v1": (
        384,
        0,
        "e725303bb8e9a558b0eff992333cecb7d6e26976d00aaee85d1fab53d4084811",
        "76dd75322f7165d010b880cb87f2a0759e611eac61f18f752754660c9234d74d",
    ),
    "natural/t5/paired-v1": (
        384,
        0,
        "0046739d27f802fdb7135755c0228b546d126d81ab99dba28eabb3adba3684a3",
        "f78d514e203e0e8db50b520be8d43402ecb7f7df2a43750fa240fca104adab87",
    ),
}


def manifest_scope(scope):
    return PAIRED_MANIFESTS[scope] if scope in PAIRED_SCOPES else SCOPES[scope]


def known_public_release(revision, index_sha):
    return (revision, index_sha) in (
        (REVISION, INDEX_SHA),
        (PAIRED_RELEASE_REVISION, PAIRED_INDEX_SHA),
        (LAYOUT_REVISION, LAYOUT_INDEX_SHA),
        (FULL_MEDIA_REVISION, FULL_INDEX_SHA),
        (PRE_LAYOUT_REVISION, FULL_INDEX_SHA),
        (T2_960_REVISION, T2_960_INDEX_SHA),
        (PREVIOUS_REVISION, PREVIOUS_INDEX_SHA),
        (ORIGINAL_REVISION, ORIGINAL_INDEX_SHA),
    )


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def checked(data, expected, name):
    require(sha(data) == expected, f"SHA-256 mismatch for {name}")
    return data


def relative(path, scope):
    value = PurePosixPath(path)
    require(
        not value.is_absolute()
        and ".." not in value.parts
        and value.is_relative_to(scope),
        f"Unexpected relative path {value}",
    )
    return value.as_posix()


def media(path, digest, scope, **fields):
    path = relative(path, scope)
    require(re.fullmatch(r"[0-9a-f]{64}", digest), f"Missing SHA-256 for {path}")
    return dict(
        path=path, sha256=digest, mime=MIME[PurePosixPath(path).suffix], **fields
    )


def trial(scope, video, protocol, manifest_sha, static=None, private=False):
    domain, name = scope.split("/")
    task = name.upper()
    sid = video["sample_id"]
    require(
        video["domain"] == domain and video["task"] == task,
        f"Wrong domain or task for {sid}",
    )
    choices = ["YES", "NO"] if task == "T5" else ["LEFT", "RIGHT"]
    truth = video["truth"]
    require(
        (truth is None or truth in choices)
        and video.get("choices", choices) == choices
        and video.get("accuracy_eligible", truth is not None) == (truth is not None),
        f"Unexpected ground truth or choices for {sid}",
    )
    prompt = video["prompt"]
    require(
        isinstance(prompt, str) and bool(prompt.strip()), f"Missing prompt for {sid}"
    )
    path = video.get("video_path") or video["path"]
    parameters = (
        video.get("parameters")
        or video.get("source_parameters")
        or video.get("frozen_row", {}).get("source_parameters")
    )
    require(isinstance(parameters, dict), f"Missing parameters for {sid}")
    mask = video.get("mask", parameters.get("mask")) if task == "T5" else None
    original = media(path, video["video_sha256"], scope, role="video")
    source = "private" if private else "public"
    mode = "static" if static is not None else "video"
    inspection = (
        mode == "static"
        and task == "T5"
        and static.get("purpose") == "stimulus inspection"
    )
    case = sid
    if static is not None:
        require(
            (
                static["sample_id"],
                static["truth"],
                static["video_sha256"],
                static["video_path"],
            )
            == (sid, truth, original["sha256"], original["path"]),
            f"Wrong static parent for {sid}",
        )
        if inspection:
            require(
                static["purpose"] == "stimulus inspection"
                and not static["model_evaluated"],
                f"Unexpected T5 static purpose for {sid}",
            )
        else:
            require(
                static["choices"] == choices
                and isinstance(static["prompt"], str)
                and bool(static["prompt"].strip()),
                f"Invalid static prompt or choices for {sid}",
            )
            prompt = static["prompt"]
            case = static["static_case_id"]
        ordered = [
            media(
                item["path"],
                item["sha256"],
                scope,
                **{k: v for k, v in item.items() if k not in ("path", "sha256")},
            )
            for item in static["images"]
        ]
        if task == "T5":
            expected = 3 if inspection else len(static["frame_indices"]) + 1
        else:
            expected = {"T1": 4, "T2": 1, "T3": 1, "T4": 2}[task]
        require(
            len(ordered) == expected and all(x["mime"] == "image/png" for x in ordered),
            f"Unexpected static inputs for {sid}",
        )
    elif task == "T5":
        ordered = [
            media(
                video["reference_path"],
                video["reference_sha256"],
                scope,
                role="reference",
            ),
            original,
        ]
    else:
        ordered = [original]
    request_id = None if inspection else sha(encoded([scope, mode, case]))
    profile = "/".join(
        [scope, mode] + (["masked" if private else "unmasked"] if task == "T5" else [])
    )
    return dict(
        scope=scope,
        domain=domain,
        task=task,
        input=mode,
        sample_id=sid,
        source=source,
        source_manifest_sha256=manifest_sha,
        protocol=protocol,
        source_version=video.get("dataset_revision")
        or video.get("source_version")
        or video.get("protocol_id"),
        source_video=original,
        original_video_ID=sid,
        legacy_sample_id=video.get("legacy_sample_id"),
        source_record_sha256=video.get("source_record_sha256")
        or video.get("source_manifest_sha256"),
        condition_key=video.get("condition_key"),
        parameters=parameters,
        mask=mask,
        background_id=video.get(
            "background_id", video.get("frozen_row", {}).get("background_id")
        ),
        object_id=video.get("object_id", video.get("frozen_row", {}).get("object_id")),
        object_ids_in_point_order=video.get("object_ids_in_point_order"),
        reference=next(
            (item for item in ordered if item.get("role") == "reference"), None
        ),
        media=ordered,
        static_case_id=case if static is not None and not inspection else None,
        model_input_sha256=(static or {}).get("model_input_sha256"),
        static_alias_ids=(static or {}).get("static_alias_ids"),
        frame_indices=(static or {}).get("frame_indices"),
        source_video_format={k: video[k] for k in ("frames", "fps", "width", "height")},
        request_id=request_id,
        profile=profile,
        prompt=prompt,
        prompt_sha256=sha(prompt.encode()),
        choices=choices,
        truth=truth,
        accuracy_eligible=not inspection and truth is not None,
        planned_status="INSPECTION_ONLY" if inspection else "CALLABLE",
    )


def row_map(raw, count):
    rows = [json.loads(line) for line in raw.splitlines()]
    by_id = {r["sample_id"]: r for r in rows}
    require(len(by_id) == len(rows) == count, f"Expected {count} unique manifest IDs")
    return by_id


def public_plan(root=None):
    def get(path, digest):
        if root is not None:
            data = (root / path).read_bytes()
        else:
            with urlopen(HF_URL + path, timeout=120) as response:
                data = response.read()
        return checked(data, digest, path)

    index = json.loads(get("dataset-index.json", INDEX_SHA))
    require(
        set(SCOPES).issubset(index["scopes"])
        and set(index["scopes"]) - set(SCOPES) <= set(PAIRED_SCOPES),
        "Unexpected public scope set",
    )

    def scope_plan(entry):
        name, info = entry
        scope = name
        count, unique_static, video_sha, static_sha = SCOPES[scope]
        require(
            info["media_count"] == info["static_trial_count"] == count,
            f"Scope count differs for {scope}",
        )
        video = row_map(get(relative(info["metadata"], scope), video_sha), count)
        static = row_map(
            get(relative(info["static_metadata"], scope), static_sha), count
        )
        require(
            video.keys() == static.keys(),
            f"Public video and static IDs differ for {scope}",
        )
        protocol = (
            info.get("protocol")
            or {
                "classic/t1": "classic-rgb16-v0.4-plus-m05",
                "classic/t2": "classic-rgb16-v0.4",
                "classic/t3": "classic-rgb16-v0.4",
                "classic/t4": "classic-t4-strict-v1",
                "natural/t1": "natural-t1-rgb-20261030-plus-m05",
                "natural/t3": "natural-t3-lr-paired-20261102",
            }[scope]
        )
        items = []
        for sid in sorted(video):
            items.append(trial(scope, video[sid], protocol, video_sha))
            items.append(
                trial(scope, video[sid], protocol + "/static", static_sha, static[sid])
            )
        require(
            len(
                {
                    r["request_id"]
                    for r in items
                    if r["input"] == "static" and r["request_id"]
                }
            )
            == unique_static,
            f"Independent static count differs for {scope}",
        )
        return items

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = [
            row
            for items in pool.map(
                scope_plan,
                ((scope, index["scopes"][scope]) for scope in sorted(SCOPES)),
            )
            for row in items
        ]
    return rows


class PrivateDelta:
    """Generation-pinned, explicitly selected GCS supplement. No bucket is built in."""

    def __init__(self, uri, metadata_root=None):
        self.root = metadata_root
        self.bucket = None
        self.locations = {}
        if uri:
            from google.cloud import storage

            require(
                uri.startswith("gs://") and uri.endswith("#" + PRIVATE_GENERATION),
                "Use the authorized private index URL with its exact generation suffix",
            )
            bucket, name = uri[5:].rsplit("#", 1)[0].split("/", 1)
            self.bucket = storage.Client().bucket(bucket)
        else:
            require(
                metadata_root is not None,
                "An authorized private index or local metadata root is required",
            )
        if metadata_root is None:
            blob = self.bucket.blob(name, generation=int(PRIVATE_GENERATION))
            raw = blob.download_as_bytes(
                if_generation_match=int(PRIVATE_GENERATION),
                raw_download=True,
                timeout=60,
            )
        else:
            raw = (metadata_root / "index.json").read_bytes()
        self.index = json.loads(checked(raw, PRIVATE_SHA, "private index"))
        require(
            self.index["hf_release_revision"] == ORIGINAL_REVISION
            and self.index["t5_private_masked_trials"] == 192,
            "Unexpected private release",
        )
        if self.bucket:
            require(
                self.index["bucket"] == self.bucket.name,
                "Private bucket differs from pinned index",
            )

    def object_bytes(self, entry):
        generation = int(entry["generation"])
        blob = self.bucket.blob(entry["object"], generation=generation)
        raw = blob.download_as_bytes(
            if_generation_match=generation, raw_download=True, timeout=60
        )
        require(len(raw) == entry["bytes"], "Private object length differs")
        return checked(raw, entry["sha256"], "private object")

    def plan(self, public):
        rows = []
        for scope in ("classic/t5", "natural/t5"):
            index = self.index
            path = index["scopes"][scope]["metadata"]
            entry = index["files"][path]
            raw = (
                (self.root / path).read_bytes()
                if self.root
                else self.object_bytes(entry)
            )
            records = row_map(
                gzip.decompress(checked(raw, entry["sha256"], scope)), 192
            )
            current = {
                r["sample_id"]: r
                for r in public
                if r["scope"] == scope and r["input"] == "video"
            }
            unmasked = {
                sid
                for sid, record in records.items()
                if record["source"]["mask"] == "none"
            }
            require(
                unmasked == current.keys(),
                f"Private/public unmasked IDs differ for {scope}",
            )
            for sid, record in sorted(records.items()):
                source = record["source"]
                require(
                    source["sample_id"] == sid
                    and record["truth"] == source["truth"]
                    and record["video"]["sha256"] == source["video_sha256"]
                    and record["reference"]["sha256"] == source["reference_sha256"],
                    f"Private source mismatch for {sid}",
                )
                if sid in unmasked:
                    expected = current[sid]
                    require(
                        (
                            expected["source_video"]["sha256"],
                            expected["reference"]["sha256"],
                            expected["truth"],
                            expected["prompt"],
                        )
                        == (
                            source["video_sha256"],
                            source["reference_sha256"],
                            source["truth"],
                            source["prompt"],
                        ),
                        f"Private/public unmasked identity differs for {sid}",
                    )
                    continue
                require(
                    record["static"] is None and not source["publicly_redistributable"],
                    f"Unexpected private static or redistribution status for {sid}",
                )
                for key in ("video", "reference"):
                    item = record[key]
                    require(
                        item["path"] == source[key + "_path"],
                        f"Private path differs for {sid}",
                    )
                    self.locations[item["path"]] = item
                rows.append(
                    trial(
                        scope, source, record["protocol"], entry["sha256"], private=True
                    )
                )
        require(len(rows) == 192, "Expected 192 masked T5 trials")
        return rows

    def media_bytes(self, item):
        require(
            self.bucket is not None,
            "Use an authorized --private-index or provide a verified local media root",
        )
        location = self.locations[item["path"]]
        require("direct" in location, "Expected a direct object for T5 media")
        raw = self.object_bytes(location["direct"])
        require(len(raw) == location["bytes"], "Private T5 media length differs")
        return raw


def public_media(item):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=REPO,
        repo_type="dataset",
        revision=REVISION,
        filename=item["path"],
        token=False,
    )
    return Path(path).read_bytes()


def unique_requests(rows):
    requests = {}
    signatures = {}
    identities = set()
    for row in rows:
        identity = row["scope"], row["input"], row["sample_id"]
        require(identity not in identities, f"Duplicate release trial {identity}")
        identities.add(identity)
        key = row["request_id"]
        if key is None:
            continue
        signature = (
            row["profile"],
            row["prompt"],
            tuple(row["choices"]),
            row["truth"],
            tuple((item["mime"], item["sha256"]) for item in row["media"]),
        )
        if key in signatures:
            require(signatures[key] == signature, f"Inconsistent static alias {key}")
        else:
            requests[key] = row
            signatures[key] = signature
    return requests


def counts(rows):
    grouped = {}
    for row in rows:
        group = grouped.setdefault(row["profile"], Counter())
        group["logical_rows"] += 1
        group["planned_scorable_rows"] += row["accuracy_eligible"]
        group["inspection_rows"] += row["request_id"] is None
    for row in unique_requests(rows).values():
        grouped[row["profile"]]["unique_requests"] += 1
    return {name: dict(value) for name, value in sorted(grouped.items())}
