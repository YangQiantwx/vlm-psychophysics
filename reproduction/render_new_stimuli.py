"""Render the frozen Natural T4 and unmasked T5 stimuli from a Hub snapshot."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys

BUNDLE = Path(__file__).resolve().parent
os.environ["OPENBLAS_NUM_THREADS"] = "1"

def load(path):
    return json.loads(path.read_text())

def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]

def t4_job(job):
    import media
    payload, output, expected = job
    media.DATA = output
    sid = media.encode_case(payload)
    record = media.verify_case(payload)
    assert record["video_sha256"] == expected["video_sha256"], sid
    assert record["stream_sha256"] == expected["RGB_stream_sha256"], sid
    return sid

def t5_job(job):
    import prepare_t45_final as renderer
    output, row, expected = job
    record = renderer.render_one((output,row))
    assert record["video_sha256"] == expected["video_sha256"], row["sample_id"]
    assert record["stream_sha256"] == expected["RGB_stream_sha256"], row["sample_id"]
    return row["sample_id"]

def prepare_t4(output, wanted):
    source = BUNDLE/"natural-t4"
    metadata = BUNDLE.parent/"natural/t4/metadata.jsonl"
    if not source.is_dir():
        source = BUNDLE/"natural-t4-strict-v1"
        metadata = BUNDLE.parent/"natural/t4/strict-v1/metadata.jsonl"
    sys.path.insert(0,str(source))
    from design import cases, sha
    path = source/"combinations.json"
    plan = load(path)
    digest = sha(path)
    for asset in plan["assets"].values():
        asset["path"] = str(source/"prepared"/Path(asset["path"]).name)
    published = {r["sample_id"]:r for r in rows(metadata)}
    assert len(published)==1536
    return t4_job, [((*case,plan["assets"],digest),output,published[case[0]])
                    for case in cases(plan) if not wanted or case[0] in wanted]

def prepare_t5(output, wanted):
    source = BUNDLE/"t5"
    if not source.is_dir():
        source = BUNDLE/"t5-unmasked"
    sys.path.insert(0,str(source))
    from prepare_t45_final import paths
    published = {}
    for domain in ("classic","natural"):
        published.update({r["sample_id"]:r for r in rows(BUNDLE.parent/f"{domain}/t5/metadata.jsonl")})
    assert len(published)==192
    _, data, _ = paths(output)
    for folder in (data/"videos",data/"qa"):
        folder.mkdir(parents=True,exist_ok=True)
    for name in ("prepared","references"):
        dest = data/name
        if not dest.exists():
            dest.symlink_to(source/name,target_is_directory=True)
        assert dest.resolve() == (source/name).resolve()
    return t5_job, [(output,r,published[r["sample_id"]]) for r in load(source/"rows.json")
                    if not wanted or r["sample_id"] in wanted]

def main():
    global BUNDLE
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("task",choices=("natural-t4","t5-unmasked"))
    p.add_argument("--snapshot", required=True, type=Path, help="Root of the pinned Hub snapshot")
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--sample-id",action="append",default=[])
    p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    BUNDLE = a.snapshot.resolve() / "reproduction"
    out=a.output.resolve()
    if out.is_relative_to(BUNDLE.parent):
        p.error("Use an output directory outside the downloaded snapshot.")
    prepare = prepare_t4 if a.task=="natural-t4" else prepare_t5
    job,inputs = prepare(out,set(a.sample_id))
    if not inputs or (a.sample_id and len(inputs)!=len(set(a.sample_id))):
        p.error("One or more requested sample IDs were not found.")
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for sid in pool.map(job,inputs):
            print(sid,"SHA and decoded RGB match",flush=True)

if __name__=="__main__":
    main()
