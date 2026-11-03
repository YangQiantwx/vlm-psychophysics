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

Open `/label/` or `/experiments/?task=T4`. `--label-only` builds the six-file label client, including the generated API configuration. `--api URL` changes its generated API configuration. The complete public build fetches about 50 MB of pinned metadata/review assets; the 21,072 lossless label videos stay in private GCS. The existing R2 `MEDIA` binding supplies the explicit legacy aliases in `asset-routes.json`.

Deploy with the operator's existing gcloud and Wrangler logins. `init` provisions only the named private bucket, dedicated runtime service account and Docker repository in an existing project; it does not change billing. Existing installations can skip it.

```sh
python -B web_label/deploy.py init --project PROJECT --bucket ISOLATED_BUCKET
python -B web_label/deploy.py backend --project PROJECT --bucket ISOLATED_BUCKET --origins https://FRONTEND.pages.dev
npm --prefix web_label ci
python -B web_label/deploy.py frontend --stage /tmp/vision-tasks-build --pages-project PAGES_PROJECT --media-bucket MEDIA_BUCKET
```

The API defaults to preview/test mode (`HUMAN_ENABLED=0`), with min 0/max 2 instances, concurrency 8 matching eight Gunicorn threads, one CPU and 512 MiB. Email is self-reported identity; no email is sent or verified. Records and sessions remain private GCS objects. Do not commit those objects, local sync outputs, credentials, or real participant data. Deployment IDs and public endpoints are recorded separately in the benchmark workspace's `plans/video-library-20261030/label-deployment.json`.

Initialize a private test inventory only from a completed `validated-browser-library-v1` index. The sync verifies exact counts, all-frame source/browser identity, remote object SHA/size and 160×160 T5 references; browser trial responses expose only the media/display whitelist. Zero-signal controls retain `accuracy_eligible=false`.

```sh
python -B web_label/backend/label/sync_inventory.py --project PROJECT --bucket ISOLATED_BUCKET --index gs://ISOLATED_BUCKET/manifests/browser-current-20261030.json --expected-counts '{"T1":7680,"T2":6720,"T3":5472,"T4":816,"T5":384}' --output /tmp/private/label-manifest-test.json --receipt /tmp/private/label-sync-receipt.json --activate
```

Allocation shuffles within manifest strata and visits the remaining strata in shuffled rounds. Users choose the task; each batch is the next 20 items of a frozen per-person order. There are no exact per-batch domain/condition/answer quotas or same-pair spacing rule. The release, seed, algorithm, batch and position reconstruct the sequence. This test implementation is not a finalized human study protocol.

Play becomes available after complete downloads, SHA256 verification and browser decode readiness. The current clip is downloaded first and enables Play as soon as its own SHA and decode checks pass. Future clips download in the background and never gate the current Play button. A real byte-count progress bar distinguishes download, integrity check and decoding; connecting/session restoration have an indeterminate indicator. Three transfers run concurrently, with up to eight current/future clips and a 384 MiB compressed-media budget. The paused decoded video element moves into the stage on Next; no automatic playback occurs. Past clips and object URLs are released. Initial preparation may take tens of seconds for large T1 clips or a slow connection. One-play enforcement still uses the server start gate, fixes rate 1, and records frame callbacks/drops, visibility, CSS dimensions and DPR. Delivery timing is included in playback evidence.

Playback and answer events are synchronously journaled as separate localStorage entries before showing choices or advancing. Uploads run in the background with bounded retry backoff; only an explicit server acknowledgement removes its own record. Refresh restores pending records, upload order and session state. Storage failure prevents advance, while the UI distinguishes local pending records from server-confirmed saves. The next Play and batch completion still require acknowledgement of preceding records. `node --test web_label/test_outbox.mjs` checks offline/reload, late acknowledgements, identity isolation, conflicting answers and storage failure.

The server transfers WebM through a bounded GCS reader (4 MiB buffer, 1 MiB response chunks) without `Content-Length`. Media requests have a five-minute deadline; ordinary API calls retain a 30-second client deadline. This supports files beyond Cloud Run's [32 MiB limit for non-streamed HTTP/1 responses](https://docs.cloud.google.com/run/quotas#cloud_run_limits), while the browser still waits for and verifies the complete file before enabling Play.

`backend/label/probe_large.py` provides an authenticated cloud transfer check for a qualified item larger than 32 MiB. It creates an isolated synthetic test identity, selects a deterministic seed before its first claim, preserves the initial state and all production ordering rules, and verifies the complete assigned response SHA without writing the video locally. Its private receipt discloses the QA seed selection; no global manifest pointer is changed.

For final T4/T5 curves, `build.py --analysis-root BENCHMARK_WORKSPACE` accepts only coordinator output with `analysis/summary.json` sealed status `COMPLETE` or `COMPLETE_COSMOS_X_T45_1200`, plus `cells.csv` and both final PNGs. It never reads trial output. The workspace export equivalent is `CVPR_WORKSPACE=BENCHMARK_WORKSPACE python -B web_label/workspace_publish.py --root BENCHMARK_WORKSPACE --site-root BENCHMARK_WORKSPACE/site --output PUBLIC_OUTPUT --refresh-t45`. Full scientific re-export uses the same entry without `--refresh-t45` and requires the retained workspace inputs and existing upload receipt. Set `CLOUDFLARE_ACCOUNT_ID` and `MEDIA_BUCKET` for `workspace_transfer.py`; authentication stays in the existing Wrangler profile.

After a new immutable Pages release, update its public-input references with `python -B web_label/build.py --capture-stage STAGE --base-url https://DEPLOYMENT.vision-tasks.pages.dev`. Editable code remains under `frontend/`; the capture refuses unversioned code or private record directories.

Private analysis export uses `backend/label/export_records.py`. `bind --manifest PRIVATE_MANIFEST --manifest-object GCS_MANIFEST_OBJECT --catalog SOURCE_CATALOG [--catalog ...] --output PRIVATE_BINDING [--upload]` checks every trial against the exact video SHA and retains filenames, parameters and source/model IDs. Optional upload creates an immutable private GCS binding without changing the active inventory. `export --binding PRIVATE_BINDING --email EMAIL --mode test --output PRIVATE_CSV` reads only that participant, joins each answer to its pinned release, and rejects missing versions or mismatched identities. Output includes email (self-reported), ID, original filename, angle/complete parameters, original response, playback flag, timing and source hashes. Supply multiple bindings for an identity spanning releases. Never put binding/CSV outputs in the public website or Git repository.
