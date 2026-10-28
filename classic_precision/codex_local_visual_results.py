"""Audit closed local-model CLI sessions without decoding or running inference."""
from __future__ import annotations
import argparse
import base64
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shlex
import struct

from .astra_video_results import approved_shell


def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def lines(p):
    with Path(p).open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def archived_sha(source):
    # The execution worker serialized JSON and omitted encrypted reasoning blobs.
    # Reproduce that exact archival representation, not the raw-file formatting.
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if k!='encrypted_content'}
        if isinstance(v,list):return [clean(x) for x in v]
        return v
    h=hashlib.sha256()
    for row in lines(source):h.update((json.dumps(clean(row),ensure_ascii=False)+'\n').encode())
    return h.hexdigest()


def review_shell(command, sample):
    """Whitelist the observed decoder, technical probe, and directory bookkeeping."""
    safe = command.replace('2>&1','')
    # FFprobe technical stream fields, never tags or content-derived statistics.
    probe = re.search(r'ffprobe\b[^;&|\n]*', safe)
    if probe:
        args=shlex.split(probe[0]);i=1;inputs=[]
        permitted={'-v':{'error'},'-select_streams':{'v:0'},
                   '-of':{'default=noprint_wrappers=1'}}
        while i<len(args):
            flag=args[i]
            if flag in permitted:
                if i+1>=len(args) or args[i+1] not in permitted[flag]: return False
                i+=2
            elif flag=='-show_entries':
                if i+1>=len(args) or not args[i+1].startswith('stream='):return False
                if not set(args[i+1][7:].split(',')) <= {'width','height','nb_frames','r_frame_rate','duration','codec_name'}:return False
                i+=2
            elif flag.startswith('-'):return False
            else: inputs.append(flag);i+=1
        if len(inputs)!=1 or Path(inputs[0]).name!=Path(sample['neutral_video']).name:return False
        safe=safe[:probe.start()]+'pwd'+safe[probe.end():]
    safe=re.sub(r'\b(head|tail)\s+-\d+\b','wc -l',safe)
    safe=re.sub(r'\bls\s+-la\b','ls -1',safe)
    return approved_shell(safe,{'public_directory':sample['public_directory'],
                                'video_path':sample['neutral_video']})[0]


def image_refs(value):
    if isinstance(value,dict):
        if 'evidence_blob_sha256' in value: yield value['evidence_blob_sha256']
        else:
            for v in value.values():yield from image_refs(v)
    elif isinstance(value,list):
        for v in value:yield from image_refs(v)


def audit_case(out, folder, sample, prompt):
    result=read(folder/'result.json');launch=read(folder/'launch.json')
    identity=read(folder/'session-identity.json');rollout=folder/'rollout.jsonl'
    assert result['real_wait'] and result['owned_cli_exited'] and result['returncode']==0
    assert launch['initial_images']==0
    assert prompt.strip() in launch['prompt'] and sample['neutral_video'] in launch['prompt']
    assert sha(sample['neutral_video'])==sample['source_sha256']
    assert archived_sha(identity['rollout_path'])==sha(rollout)
    pending={};frame_indices=[];frame_paths=[];image_shas=[];commands=[];unknown=[];sizes=Counter();bits=Counter();messages=[]
    returned_calls=[]
    for row in lines(rollout):
        if row.get('type')!='response_item':continue
        p=row['payload'];typ=p.get('type')
        if typ=='function_call':
            args=json.loads(p.get('arguments','{}'));pending[p['call_id']]={'name':p['name'],'args':args}
            if p['name']=='exec_command':
                cmd=args.get('cmd','');commands.append({'cmd':cmd,'reviewed':review_shell(cmd,sample)})
            elif p['name']!='view_image':unknown.append(p['name'])
        elif typ=='function_call_output':
            call=pending.pop(p['call_id']);returned_calls.append(call['name'])
            if call['name']=='view_image':
                path=Path(call['args']['path']);path.resolve().relative_to(Path(sample['public_directory']).resolve())
                imgs=[x for x in p.get('output',[]) if x.get('type')=='input_image']
                assert len(imgs)==1
                data=base64.b64decode(imgs[0]['image_url'].split(',',1)[1])
                h=hashlib.sha256(data).hexdigest();assert h==sha(path)
                assert data[:8]==b'\x89PNG\r\n\x1a\n'
                size=struct.unpack('>II',data[16:24]);assert size==(768,512)
                sizes[str(size)]+=1;bits[str(data[24])]+=1
                frame_paths.append(str(path));frame_indices.append(int(re.search(r'(\d+)\.png$',path.name)[1]));image_shas.append(h)
        elif typ=='message' and p.get('role')=='assistant' and p.get('channel')!='analysis':
            messages.extend(x['text'] for x in p.get('content',[]) if x.get('type')=='output_text')
    assert not pending and not unknown and all(c['reviewed'] for c in commands)
    decodes=[c for c in commands if re.search(r'\bffmpeg\b',c['cmd'])]
    assert len(decodes)==1 and len(list(Path(frame_paths[0]).parent.glob('*.png')))==180
    request_refs=set();http=list(lines(folder/'request-results.jsonl'));controls=list(lines(folder/'requests.jsonl'))
    assert len(http)==len(controls)==result['requests'] and all(x['http_status']==200 for x in http)
    expected={'temperature':0,'top_p':1,'seed':0,'mm_processor_kwargs':{'do_resize':False,'do_sample_frames':False},'chat_template_kwargs':{'enable_thinking':False}}
    for c in controls:
        assert all(c['controls'].get(k)==v for k,v in expected.items())
    for p in sorted(folder.glob('request-*.json')):
        request_refs.update(image_refs(read(p)))
    assert request_refs==set(image_shas)
    for h in request_refs:assert sha(out/'payload-images'/h)==h
    assert controls[0]['images_in_context']==0
    strict=result['answer'] in ('LEFT','RIGHT')
    last=result['answer'].strip().splitlines()[-1]
    semantic=last if last in ('LEFT','RIGHT') else None
    ordered=frame_indices==list(range(1,181));complete=set(frame_indices)==set(range(1,181))
    violations=[]
    if not ordered:violations.append('REVISITED_OR_OUT_OF_ORDER_FRAMES')
    if not strict:violations.append('EXTRA_TEXT_IN_FINAL_ANSWER')
    record={'case':result['case'],'sample_id':sample['sample_id'],'sf':sample['sf'],'truth':sample['truth'],
      'answer_raw':result['answer'],'semantic_final_label':semantic,'semantic_correct':semantic==sample['truth'],
      'strict_format':strict,'returned_images':len(frame_indices),'unique_frames':len(set(frame_indices)),
      'repeated_frame_views':len(frame_indices)-len(set(frame_indices)),'chronological_no_revisit':ordered,
      'all_180_distinct_frames_returned':complete,'all_returned_pngs_match_decode':True,'shapes':dict(sizes),'png_bit_depth':dict(bits),
      'actual_requests':result['requests'],'maximum_images_in_one_request':result['maximum_images_in_context'],
      'all_payload_blobs_match_returned_frames':True,'generation_controls_verified':expected,
      'technical_metadata_probe':any('ffprobe' in c['cmd'] for c in commands),
      'technical_metadata_scope':'dimensions/frame count/fps/duration/codec; separate from visual judgments, not a zero-metadata claim',
      'pixel_processing_commands_observed':False,'reviewed_commands':commands,'frame_indices_in_order':frame_indices,
      'raw_visible_assistant_messages':messages,'http_engine_seconds_sum':sum(x['engine_http_seconds'] for x in http),
      'protocol_violations':violations,'visual_trace_status':'PASS_FULL180_IN_ORDER' if ordered else 'VIOLATION_REVISIT',
      'strict_qualified':ordered and strict and complete,'session_id':identity['session_id'],'resume_command':identity['resume_command'],
      'source_sha256':sample['source_sha256'],'rollout_sha256':sha(rollout),
      'original_rollout_sha256':sha(identity['rollout_path']),'archived_representation_verified':True,
      'result_sha256':sha(folder/'result.json'),
      'real_wait_owned_exit':True,'auditor_sha256':sha(__file__)}
    (folder/'visual-trace-audit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    return record


def run(root, attempt):
    root=Path(root);plan=root/'plans/codex-local-visual-20261027'/attempt;out=root/'runs/reproducibility/codex-local-visual-20261027'/attempt
    samples=read(plan/'input-manifest.json')['records'];prompt=(plan/'prompt.txt').read_text()
    records=[audit_case(out,d,samples[int(d.name.split('-')[1])-1],prompt) for d in sorted(out.glob('case-*')) if (d/'result.json').exists()]
    engine=read(out/'engine-receipt.json');assert engine['qsa']['state']=='applied'
    hooks=list(lines(out/'actual-vision-inputs.jsonl'))
    assert hooks and all(h['device'].startswith('cuda') and h['dtype']=='torch.bfloat16' and all(g==[1,32,48] for g in h['grid_thw']) for h in hooks)
    server=read(out/'server.wait.json');parent=read(out/'supervisor.wait.json');lease=read(out/'lease-closure.json')
    assert server['real_wait'] and server['returncode']==0 and not server['owned_survivors']
    assert parent['real_wait'] and parent['returncode']==0 and lease['original_same_inode_free'] and not lease['owned_survivors']
    summary={'status':'COMPLETE_WITH_PROTOCOL_VIOLATIONS','cases':len(records),'semantic_correct':sum(r['semantic_correct'] for r in records),
      'strict_format_cases':sum(r['strict_format'] for r in records),'strict_qualified_cases':sum(r['strict_qualified'] for r in records),
      'actual_gpu_vision_hook_records':len(hooks),'cache_policy':'prefix and multimedia cache enabled; repeated identical periodic-frame content may reuse cached features',
      'every_case_observed_180_unique_frames':all(r['unique_frames']==180 for r in records),'case_summaries':[{k:r[k] for k in ('case','sample_id','sf','truth','semantic_final_label','semantic_correct','strict_format','returned_images','unique_frames','repeated_frame_views','actual_requests','maximum_images_in_one_request','strict_qualified','protocol_violations','http_engine_seconds_sum')} for r in records],
      'terminal_wait_owned_exit_original_lock_verified':True,'scientific_calls_added':0,'auditor_sha256':sha(__file__),
      'limitations':['Sequential image-tool route, not one simultaneous 180-frame video tensor.','Input cache and context retention differ from the formal video protocol.','Only one case satisfies the frame-order/no-revisit and final-format checks; do not present 1/1 as stable model accuracy.','All cases used a technical FFprobe; not a literally metadata-free session.']}
    (out/'visual-trace-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--attempt',default='qwen-followup1');a=p.parse_args()
    print(json.dumps(run(a.root,a.attempt),ensure_ascii=False))
