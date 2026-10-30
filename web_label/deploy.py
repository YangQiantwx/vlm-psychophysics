"""Deploy the isolated label API or a complete Cloudflare Pages build.

Uses the operator's existing gcloud/Wrangler authentication; no keys are read.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('init', 'backend', 'frontend'))
    p.add_argument('--project', default=os.environ.get('GOOGLE_CLOUD_PROJECT'))
    p.add_argument('--bucket', default=os.environ.get('LABEL_BUCKET'))
    p.add_argument('--region', default='us-west1')
    p.add_argument('--service', default='cvpr-label-api')
    p.add_argument('--repository', default='cvpr-label-build')
    p.add_argument('--origins', default=os.environ.get('LABEL_ORIGINS'))
    p.add_argument('--pages-project', default='vision-tasks')
    p.add_argument('--media-bucket', default='vision-tasks-media')
    p.add_argument('--stage', type=Path, default=HERE / 'dist')
    p.add_argument('--wrangler-cli', type=Path, default=HERE / 'node_modules/wrangler/wrangler-dist/cli.js')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    env = dict(os.environ, CLOUDSDK_PYTHON=sys.executable)

    def run(command, check=True, cwd=HERE):
        print(shlex.join(map(str, command)), flush=True)
        if args.dry_run:
            return 0
        return subprocess.run(list(map(str, command)), env=env, check=check, cwd=cwd).returncode

    if args.action == 'frontend':
        if not (args.stage / 'label/index.html').is_file():
            p.error('Build a complete public stage first')
        if not args.wrangler_cli.is_file():
            p.error('Install the pinned existing package-lock with npm ci, or provide --wrangler-cli')
        config = json.loads((HERE / 'wrangler.jsonc').read_text())
        config['name'] = args.pages_project
        config['pages_build_output_dir'] = str(args.stage.resolve())
        config['r2_buckets'][0]['bucket_name'] = args.media_bucket
        # Pages rejects --config paths. Use an isolated cwd with the standard
        # filename so the checked-in configuration never needs to be rewritten.
        with tempfile.TemporaryDirectory(prefix='cvpr-pages-') as folder:
            (Path(folder) / 'wrangler.json').write_text(json.dumps(config, indent=2) + '\n')
            run(['node', args.wrangler_cli.resolve(), 'pages', 'deploy', args.stage.resolve(), '--project-name', args.pages_project, '--branch', 'main'], cwd=folder)
        return
    if not args.project or not args.bucket:
        p.error('--project and --bucket are required for cloud resources')
    project = ['--project', args.project]
    sa = args.service + '@' + args.project + '.iam.gserviceaccount.com'
    if args.action == 'init':
        run(['gcloud', 'services', 'enable', 'run.googleapis.com', 'cloudbuild.googleapis.com', 'artifactregistry.googleapis.com', 'storage.googleapis.com', *project])
        if run(['gcloud', 'storage', 'buckets', 'describe', 'gs://' + args.bucket, *project, '--format=value(name)'], check=False) != 0:
            run(['gcloud', 'storage', 'buckets', 'create', 'gs://' + args.bucket, '--location', args.region, '--uniform-bucket-level-access', '--public-access-prevention', *project])
        if run(['gcloud', 'iam', 'service-accounts', 'describe', sa, *project, '--format=value(email)'], check=False) != 0:
            run(['gcloud', 'iam', 'service-accounts', 'create', args.service, *project])
        run(['gcloud', 'storage', 'buckets', 'add-iam-policy-binding', 'gs://' + args.bucket, '--member', 'serviceAccount:' + sa, '--role', 'roles/storage.objectAdmin', *project])
        if run(['gcloud', 'artifacts', 'repositories', 'describe', args.repository, '--location', args.region, *project, '--format=value(name)'], check=False) != 0:
            run(['gcloud', 'artifacts', 'repositories', 'create', args.repository, '--repository-format', 'docker', '--location', args.region, *project])
        return
    if not args.origins:
        p.error('--origins must list the permitted frontend origins')
    tag = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d-%H%M%S')
    image = f'{args.region}-docker.pkg.dev/{args.project}/{args.repository}/api:{tag}'
    run(['gcloud', 'builds', 'submit', HERE / 'backend/label', '--tag', image, *project])
    variables = '^|^LABEL_BUCKET=' + args.bucket + '|LABEL_ORIGINS=' + args.origins + '|HUMAN_ENABLED=0'
    run(['gcloud', 'run', 'deploy', args.service, '--image', image, '--region', args.region, '--service-account', sa,
         '--allow-unauthenticated', '--min-instances=0', '--max-instances=2', '--cpu=1', '--memory=512Mi',
         '--concurrency=8', '--timeout=300', '--no-cpu-boost', '--set-env-vars', variables, *project])


if __name__ == '__main__':
    main()
