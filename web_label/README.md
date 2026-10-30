# Video label API and public website

`frontend/` is the editable, unbundled HTML/CSS/JS for the five-task viewer and label page. `backend/label/` contains the Flask API, Docker build, private-inventory sync and synthetic tests. `build.py` recreates the public Pages stage using SHA-pinned public inputs in `public-inputs.json`; media and scientific inputs are external dependencies, not committed videos. `workspace_publish.py`, `workspace_transfer.py` and `workspace_paths.py` preserve the full workspace export/R2 publication implementation.

Use the existing `qy` Python. The commands below assume `conda activate qy`; on the current Mac its interpreter is `/Users/qianyang/miniconda3/envs/qy/bin/python`. No separate Python environment is needed.

```sh
python -m pip install -r web_label/requirements.txt
python -B -m unittest discover -s web_label/backend/label -p 'test_*.py'
python -B -m unittest discover -s web_label -p 'test_build.py'
python -B web_label/build.py --output /tmp/vision-tasks-build
python -m http.server 18795 --directory /tmp/vision-tasks-build
```

Open `/label/` or `/experiments/?task=T4`. `--label-only` builds the four-file label client. `--api URL` changes its generated API configuration. The complete public build fetches about 50 MB of pinned metadata/review assets; the 21,072 lossless label videos stay in private GCS. The existing R2 `MEDIA` binding supplies the explicit legacy aliases in `asset-routes.json`.

Deploy with the operator's existing gcloud and Wrangler logins. `init` provisions only the named private bucket, dedicated runtime service account and Docker repository in an existing project; it does not change billing. Existing installations can skip it.

```sh
python -B web_label/deploy.py init --project PROJECT --bucket ISOLATED_BUCKET
python -B web_label/deploy.py backend --project PROJECT --bucket ISOLATED_BUCKET --origins https://FRONTEND.pages.dev
npm --prefix web_label ci
python -B web_label/deploy.py frontend --stage /tmp/vision-tasks-build --pages-project PAGES_PROJECT --media-bucket MEDIA_BUCKET
```

The API defaults to preview/test mode (`HUMAN_ENABLED=0`), with min 0/max 2 instances, one CPU and 512 MiB. Email is self-reported identity; no email is sent or verified. Records and sessions remain private GCS objects. Do not commit those objects, local sync outputs, credentials, or real participant data. Deployment IDs and public endpoints are recorded separately in the benchmark workspace's `plans/video-library-20261030/label-deployment.json`.

Initialize a private test inventory only from a completed `validated-browser-library-v1` index. The sync verifies exact counts, all-frame source/browser identity, remote object SHA/size and 160×160 T5 references; browser trial responses expose only the media/display whitelist. Zero-signal controls retain `accuracy_eligible=false`.

```sh
python -B web_label/backend/label/sync_inventory.py --project PROJECT --bucket ISOLATED_BUCKET --index gs://ISOLATED_BUCKET/manifests/browser-current-20261030.json --expected-counts '{"T1":7680,"T2":6720,"T3":5472,"T4":816,"T5":384}' --output /tmp/private/label-manifest-test.json --receipt /tmp/private/label-sync-receipt.json --activate
```

Allocation shuffles within manifest strata and visits the remaining strata in shuffled rounds. Users choose the task; each batch is the next 20 items of a frozen per-person order. There are no exact per-batch domain/condition/answer quotas or same-pair spacing rule. The release, seed, algorithm, batch and position reconstruct the sequence. This test implementation is not a finalized human study protocol.

Play becomes available only after a complete download, SHA256 verification and browser decode readiness. The client retains a three-video prefetch window, releases past blobs, fixes rate 1 and records playback time, callbacks/dropped frames, visibility, CSS dimensions and DPR. Downloading a large T1 video can delay initial readiness. One-play enforcement governs this UI/API, rather than acting as DRM.

For final T4/T5 curves, `build.py --analysis-root BENCHMARK_WORKSPACE` accepts only coordinator output with `analysis/summary.json` status exactly `COMPLETE`, plus `cells.csv` and both final PNGs. It never reads trial output. The workspace export equivalent is `CVPR_WORKSPACE=BENCHMARK_WORKSPACE python -B web_label/workspace_publish.py --root BENCHMARK_WORKSPACE --site-root BENCHMARK_WORKSPACE/site --output PUBLIC_OUTPUT --refresh-t45`. Full scientific re-export uses the same entry without `--refresh-t45` and requires the retained workspace inputs and existing upload receipt. Set `CLOUDFLARE_ACCOUNT_ID` and `MEDIA_BUCKET` for `workspace_transfer.py`; authentication stays in the existing Wrangler profile.

After a new immutable Pages release, update its public-input references with `python -B web_label/build.py --capture-stage STAGE --base-url https://DEPLOYMENT.vision-tasks.pages.dev`. Editable code remains under `frontend/`; the capture refuses unversioned code or private record directories.
