"""Inspect/export the pinned reproduction configuration without loading a model."""
from __future__ import annotations
import argparse
import copy
import json
from importlib.resources import files
from pathlib import Path


def configuration(model, protocol, model_dir=None, tokenizer_dir=None, data_root=None):
    manifest=json.loads(files('classic_precision').joinpath('reproduction.json').read_text())
    p=manifest['protocols'][protocol]
    if model not in p['supported_models']:
        raise ValueError(p.get('unsupported',{}).get(model,'Unsupported model/protocol'))
    if p.get('config_export_supported') is False:
        raise ValueError('Controlled RGB16 requires the frozen per-model profile; see EXPERIMENT_PLAN.md')
    spec=copy.deepcopy(manifest['models'][model]);engine=spec.pop('engine_kwargs')
    bindings={'${MODEL_DIR}':model_dir,'${TOKENIZER_DIR}':tokenizer_dir or model_dir,'${DATA_ROOT}':data_root}
    for k,v in list(engine.items()):
        if isinstance(v,str) and v in bindings and bindings[v] is not None:engine[k]=str(Path(bindings[v]).resolve())
    adapter=None
    if protocol=='native-original':
        if model in ('qwen','cosmos'):engine['mm_processor_kwargs']['do_resize']=False
        elif model=='nemo':adapter={'callable':'classic_precision.native_path_smoke:nemo_original_geometry','only_config_fields':{'video_target_num_patches':1536,'video_maintain_aspect_ratio':True},'pixel_transform':False}
    return {'schema':manifest['schema'],'status':'CONFIG_ONLY_NOT_A_RUN','model':model,'protocol':protocol,'versions':spec,'packages':manifest['packages'],'engine_kwargs':engine,'native_config_adapter':adapter,'generation':manifest['generation'],'input_contract':p,'QSA_runtime_control':'runtime_compat/classic_precision_qsa_determinism.py;install in model worker and verify state=applied' if model=='qwen' else None,'release_gaps':manifest['release_gaps']}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('list',help='List protocols, model support, and code groups')
    s=sub.add_parser('show',help='Export exact native engine settings; no inference')
    s.add_argument('--model',choices=['qwen','nemo','cosmos','gemma'],required=True)
    s.add_argument('--protocol',choices=['native-default','native-original','controlled-rgb16'],required=True)
    for name in ['model-dir','tokenizer-dir','data-root','output']:s.add_argument('--'+name,type=Path)
    a=p.parse_args(argv)
    if a.command=='list':
        m=json.loads(files('classic_precision').joinpath('reproduction.json').read_text());v={k:m[k] for k in ['scope','protocols','code_groups','release_gaps']}
    else:
        try:v=configuration(a.model,a.protocol,a.model_dir,a.tokenizer_dir,a.data_root)
        except ValueError as e:p.exit(2,str(e)+'\n')
    text=json.dumps(v,ensure_ascii=False,indent=2)+'\n'
    if getattr(a,'output',None):
        with a.output.open('x') as f:f.write(text)
    else:print(text,end='')

if __name__=='__main__':main()
