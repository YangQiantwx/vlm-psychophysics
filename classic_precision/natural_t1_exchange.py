"""One-shot, peer-restricted Tailscale transfer of this dataset's exact files.

No account credentials are forwarded. Receivers validate every file before rename.
Servers expose only a sealed manifest, never a directory browser or arbitrary paths.
"""
from __future__ import annotations
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import os
from pathlib import Path
import secrets
import tarfile
import threading
import urllib.request

from . import natural_t1_rgb as t
from . import natural_t13_stimuli as s

HOST_IP={'fourier':'100.65.75.63','radon':'100.106.240.56'}


def entries(root,kind,host):
    p,d,r=t.paths(root)
    out=[]
    if kind=='backgrounds':
        for f in s.read(p/'background-features.json').values():
            path=Path(root)/f['prepared_relative']
            assert path.is_file() and not path.is_symlink()
            out.append({'path':f['prepared_relative'],'sha256':f['prepared_sha256'],'bytes':path.stat().st_size})
    else:
        wait=s.read(r/f'{host}-wait.json');result=s.read(r/f'{host}-result.json')
        assert wait['actual_child_wait']==0 and wait['original_lock_free'] and result['status']=='COMPLETE_CPU_SHARD'
        plan=s.read(p/'distributed.json');ids=set(plan['assignments'][host])
        for b in s.read(p/'allocation.json')['blocks']:
            if b['block_id'] not in ids:continue
            q=d/'qa'/(b['block_id']+'.json');v=s.read(q)
            assert v['status']=='PASS' and v['criteria_sha256']==plan['criteria_sha256']
            out.append({'path':str(q.relative_to(root)),'sha256':s.sha(q),'bytes':q.stat().st_size})
            for c in v['cases']:
                path=d/'videos/T1'/(c['sample_id']+'.mp4')
                out.append({'path':str(path.relative_to(root)),'sha256':c['video_sha256'],'bytes':c['bytes']})
        for name in ['launch','pilot-parity','worker-start','progress','result','wait']:
            path=r/f'{host}-{name}.json'
            out.append({'path':str(path.relative_to(root)),'sha256':s.sha(path),'bytes':path.stat().st_size})
    assert len(out)==len({x['path'] for x in out})
    return {'kind':kind,'host':host,'allocation_sha256':s.sha(p/'allocation.json'),'files':out}


def serve(root,kind,host,port):
    root=Path(root);p,d,r=t.paths(root);manifest=entries(root,kind,host)
    peer=HOST_IP['radon' if host=='fourier' else 'fourier'];token=secrets.token_urlsafe(32)
    payload=json.dumps(manifest).encode();state={'received':False,'completed':False}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            if self.client_address[0]!=peer or self.path!='/'+token or state['received']:
                self.send_error(403);return
            state['received']=True
            try:
                self.send_response(200);self.send_header('Content-Type','application/x-tar');self.end_headers()
                with tarfile.open(mode='w|',fileobj=self.wfile) as tar:
                    info=tarfile.TarInfo('manifest.json');info.size=len(payload);tar.addfile(info,io.BytesIO(payload))
                    for item in manifest['files']:
                        path=root/item['path'];assert not path.is_symlink() and path.stat().st_size==item['bytes']
                        info=tarfile.TarInfo(item['path']);info.size=item['bytes'];info.mode=0o644
                        with path.open('rb') as f:tar.addfile(info,f)
                state['completed']=True
            finally:
                threading.Thread(target=self.server.shutdown,daemon=True).start()
    server=HTTPServer((HOST_IP[host],port),Handler)
    ready=r/f'exchange-{kind}-{host}-ready.json'
    ready.write_text(json.dumps({'url':f'http://{HOST_IP[host]}:{server.server_port}/{token}','manifest_sha256':hashlib.sha256(payload).hexdigest(),'files':len(manifest['files']),'bytes':sum(x['bytes'] for x in manifest['files'])}))
    ready.chmod(0o600)
    timer=threading.Timer(1800,server.shutdown);timer.daemon=True;timer.start()
    try:server.serve_forever(poll_interval=1)
    finally:server.server_close();timer.cancel();ready.unlink(missing_ok=True)
    s.write(r/f'exchange-{kind}-{host}-sent.json',state)
    assert state['completed'],state


def receive(root,kind,host,url,manifest_sha):
    root=Path(root);p,d,r=t.paths(root)
    assert url.startswith('http://'+HOST_IP[host]+':')
    closed=[]
    with urllib.request.urlopen(url,timeout=120) as response,tarfile.open(mode='r|',fileobj=response) as tar:
        member=next(iter(tar));assert member.name=='manifest.json' and member.size<10_000_000
        payload=tar.extractfile(member).read();assert hashlib.sha256(payload).hexdigest()==manifest_sha
        manifest=json.loads(payload);assert manifest['kind']==kind and manifest['host']==host
        assert manifest['allocation_sha256']==s.sha(p/'allocation.json')
        expected={x['path']:x for x in manifest['files']};seen=set()
        for member in tar:
            if member.name=='manifest.json':continue
            assert member.isfile() and member.name in expected and member.name not in seen
            rel=Path(member.name);assert not rel.is_absolute() and '..' not in rel.parts
            assert str(rel).startswith(('datasets/natural-t13-20261029/backgrounds/',f'datasets/{t.NAME}/',f'runs/stimulus-preparation/{t.NAME}/'))
            item=expected[member.name];assert member.size==item['bytes'];seen.add(member.name)
            dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            existing=dest.exists()
            if existing:assert not dest.is_symlink() and s.sha(dest)==item['sha256'],str(dest)
            temp=dest.with_name(dest.name+'.exchange-partial');assert not temp.exists()
            digest=hashlib.sha256();stream=tar.extractfile(member)
            output=None if existing else temp.open('xb')
            try:
                while True:
                    data=stream.read(4<<20)
                    if not data:break
                    digest.update(data)
                    if output:output.write(data)
            finally:
                if output:output.close()
            assert digest.hexdigest()==item['sha256'],str(dest)
            if not existing:temp.replace(dest)
            closed.append({'path':member.name,'sha256':item['sha256'],'bytes':item['bytes'],'existing':existing})
            if len(closed)%100==0:s.write(r/f'exchange-{kind}-{host}-progress.json',{'verified_files':len(closed),'total':len(expected)})
        assert seen==set(expected)
    receipt={'status':'PASS_TRANSFER_BYTE_IDENTITY','kind':kind,'from_host':host,'files':len(closed),'bytes':sum(x['bytes'] for x in closed),'manifest_sha256':manifest_sha,'allocation_sha256':manifest['allocation_sha256']}
    s.write(r/f'exchange-{kind}-{host}-receipt.json',receipt)
    return receipt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['serve','receive']);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--kind',choices=['backgrounds','shard'],required=True);ap.add_argument('--host',choices=HOST_IP,required=True);ap.add_argument('--port',type=int,default=0)
    ap.add_argument('--url');ap.add_argument('--manifest-sha');a=ap.parse_args()
    if a.command=='serve':serve(a.root,a.kind,a.host,a.port)
    else:print(json.dumps(receive(a.root,a.kind,a.host,a.url,a.manifest_sha)))


if __name__=='__main__':main()
