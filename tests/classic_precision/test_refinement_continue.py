import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from classic_precision import refinement_continue as c


def test_partition_keeps_only_proven_unattempted_and_rejects_new_claims():
    profile={'assigned_ids':['a','b','c']}
    audit={'completed_ids':['a'],'unattempted_ids':['b','c'],'claimed_ids':['a'],'UNCERTAIN_ids':[],
        'real_wait':{'real_wait_observed':True,'returncode':0,'owned_engines_exited':True,
            'original_physical_lease_released':True,'terminal_status':'PAUSED','completed_ids':['a']}}
    assert c.remaining_ids(profile,audit,{'a'})==['b','c']
    with pytest.raises(ValueError,match='Claims changed'):
        c.remaining_ids(profile,audit,{'a','b'})
    audit['UNCERTAIN_ids']=['b']
    with pytest.raises(ValueError,match='UNCERTAIN'):
        c.remaining_ids(profile,audit,{'a','b'})


def handoff():
    closed={'returncode':0,'real_wait_observed':True,'owned_engines_exited':True,
        'original_physical_lease_released':True,'terminal_status':'COMPLETE'}
    summary={'status':'COMPLETE_PASS','expected':1728,'technical_pass':1728,'closures':[closed]*15}
    decision={'schema':'reader-route-decision-v1','status':'REPORTED','summary_sha256':c.digest(summary),
        'formal_action':'CONTINUE_EXISTING_CONTROLLED_PROTOCOL','authorization_message':c.AUTHORIZATION,
        'judgment':'Retain controlled protocol; native sensitivity separately reported.','reported_HST':'test'}
    return summary,decision


def test_incomplete_smoke_or_missing_reported_judgment_blocks_continuation():
    s,d=handoff();c.validate_handoff(s,d,c.digest(s))
    s['technical_pass']=1727
    with pytest.raises(ValueError,match='incomplete'):
        c.validate_handoff(s,d,c.digest(s))
    s,d=handoff();d['status']='PREPARED'
    with pytest.raises(ValueError,match='report/judgment'):
        c.validate_handoff(s,d,c.digest(s))


def test_paused_continuation_holds_successor_without_retry(tmp_path,monkeypatch):
    s,d=handoff();report=tmp_path/'report.json';c.write_new(report,s)
    d['summary_sha256']=c.file_sha256(report);decision=tmp_path/'decision.json';c.write_new(decision,d)
    output=tmp_path/'first-output';profile=tmp_path/'first-profile.json'
    c.write_new(profile,{'assigned_ids':['b'],'pause_file':str(tmp_path/'PAUSE'),
        'claim_root':str(tmp_path/'claims'),'output':str(output),'environment':{},'model':'nemo'})
    q=tmp_path/'queue.json';c.write_new(q,{'status':'SEALED','stages':[str(profile),str(tmp_path/'must-not-launch')],
        'reported_decision':c.descriptor(decision),'smoke_report':c.descriptor(report),'automatic_retry':False})
    calls=[]
    def child(argv):
        calls.append(argv)
        def wait():
            c.write_new(output.with_suffix('.wait.json'),{'returncode':0,'real_wait_observed':True,
                'owned_engines_exited':True,'original_physical_lease_released':True,
                'terminal_status':'PAUSED','completed_ids':[]})
            return 0
        return SimpleNamespace(wait=wait)
    monkeypatch.setattr(c.subprocess,'Popen',child)
    assert c.queue(q)=={'status':'PAUSED'}
    assert len(calls)==1 and c.read(q)['status']=='PAUSED'
    with pytest.raises(ValueError,match='cannot be replayed'):
        c.queue(q)


def test_nemo_continuation_refuses_unaccepted_runtime_restoration(tmp_path,monkeypatch):
    from classic_precision.nemo_kernel_replay import CONTRACT
    p=tmp_path/'profile.json'
    c.write_new(p,{'model':'nemo','continuation':{'nemo_legacy_runtime':{
        'kernel_config':CONTRACT,'canonical_audit':{'path':'unused','sha256':'unused'}}}})
    monkeypatch.setattr(c,'verified',lambda ref:{'status':'PARTIAL','technical_pass':1})
    with pytest.raises(ValueError,match='not accepted'):c.run_nemo(p)


def test_nemo_supervision_keeps_real_wait_but_selects_gated_entry(monkeypatch):
    from classic_precision import refinement_worker as worker
    observed=[]
    def popen(argv,**kwargs):observed.append(argv);return 'child'
    monkeypatch.setattr(worker.subprocess,'Popen',popen)
    def supervise(path):
        return worker.subprocess.Popen(['python','-B','-m','classic_precision.refinement_worker','run','--profile',path])
    monkeypatch.setattr(worker,'supervise',supervise)
    assert c.supervise_nemo('fixed.json')=='child'
    assert observed==[['python','-B','-m','classic_precision.refinement_continue','run-nemo','--profile','fixed.json']]
    assert worker.subprocess.Popen is popen
