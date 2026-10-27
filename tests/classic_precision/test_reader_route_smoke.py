from types import SimpleNamespace

import pytest

from classic_precision import reader_route_smoke as s
from classic_precision import reader_route_repair as r


class Engine:
    def __init__(self): self.calls=[]
    def chat(self,request,**kwargs):
        self.calls.append(('chat',request,kwargs));return ['answer']
    def generate(self,request,sampling,**kwargs):
        self.calls.append(('generate',request,{'sampling':sampling,**kwargs}));return ['answer']


def test_native_request_overrides_resampling_metadata_without_changing_other_routes():
    # This is the actual dispatch seam: engine defaults alone did not prevent
    # metadata=True from sampling 180 frames down to six.
    for model in ('qwen','cosmos'):
        e=Engine();s.submit(e,model,'C','file', 'sampling')
        assert e.calls==[('chat','file',{'sampling_params':'sampling','use_tqdm':False,
            'chat_template_kwargs':{'enable_thinking':False},
            'mm_processor_kwargs':{'do_resize':False,'do_sample_frames':False}})]
    e=Engine();s.submit(e,'nemo','C','file','sampling')
    assert 'mm_processor_kwargs' not in e.calls[0][2]  # unsupported do_resize never reaches Nemo
    for model in ('qwen','cosmos','nemo','gemma'):
        a=Engine();b=Engine()
        s.submit(a,model,'A','same','sampling');s.submit(b,model,'B','same','sampling')
        assert a.calls==b.calls==[('generate','same',{'sampling':'sampling','use_tqdm':False})]


def test_greedy_and_native_eos_drift_is_rejected():
    p=SimpleNamespace(n=1,temperature=0.,top_p=1.,seed=0,repetition_penalty=1.,
        max_tokens=16384,ignore_eos=False,stop=[],stop_token_ids=[],
        structured_outputs=SimpleNamespace(choice=['LEFT','RIGHT']))
    assert s.generation_receipt(p)['thinking'] is False
    for key,value in [('temperature',.1),('seed',1),('ignore_eos',True),('stop',['LEFT'])]:
        old=getattr(p,key);setattr(p,key,value)
        with pytest.raises(ValueError,match='greedy/native-EOS'):s.generation_receipt(p)
        setattr(p,key,old)


def test_repair_refuses_existing_answer_and_unapproved_claim(tmp_path):
    out=tmp_path/'old';claims=tmp_path/'claims';claims.mkdir()
    parent={'output':str(out),'claim_root':str(claims),'assigned_ids':['a','b']}
    r.check_parent(parent,{})
    (claims/'a.json').write_text('{}')
    with pytest.raises(ValueError,match='claim set changed'):r.check_parent(parent,{})
    (claims/'a.json').unlink();out.mkdir();(out/'raw.jsonl').write_text('{"sample_id":"a"}\n')
    with pytest.raises(ValueError,match='Original raw exists'):r.check_parent(parent,{})
