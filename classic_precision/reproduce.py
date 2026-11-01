"""Inspect/export the pinned reproduction configuration without loading a model."""
from __future__ import annotations
import argparse
import copy
import json
from importlib.resources import files
from pathlib import Path


def current_configuration(model, task, model_dir=None, tokenizer_dir=None, data_root=None):
    """Export the accepted X recipe without importing a GPU backend."""
    manifest = json.loads(files('classic_precision').joinpath('current_x_protocols.json').read_text())
    try:
        variant = manifest['default_variant_by_model_task'][model][task]
    except KeyError as error:
        raise ValueError('Current X requires a supported model and T1, T2 or T3') from error
    recipe = copy.deepcopy(manifest['variants'][variant])
    bindings = {'${MODEL_DIR}': model_dir, '${TOKENIZER_DIR}': tokenizer_dir or model_dir,
                '${DATA_ROOT}': data_root}
    for settings in ('engine_kwargs', 'effective_LLM_kwargs'):
        for key, value in list(recipe[settings].items()):
            if isinstance(value, str) and value in bindings and bindings[value] is not None:
                recipe[settings][key] = str(Path(bindings[value]).resolve())
    return {'schema': manifest['schema'], 'status': 'CONFIG_ONLY_NOT_A_RUN',
            'protocol': 'current-x', 'variant': variant, 'task': task,
            'recipe': recipe, 'data_root': str(Path(data_root).resolve()) if data_root else None,
            'source_identity_contract': manifest['source_identity_contract']}


def configuration(model, protocol, model_dir=None, tokenizer_dir=None, data_root=None, task='T1'):
    if protocol == 'current-x':
        return current_configuration(model, task, model_dir, tokenizer_dir, data_root)
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
    s.add_argument('--protocol',choices=['current-x','native-default','native-original','controlled-rgb16'],required=True)
    s.add_argument('--task',choices=['T1','T2','T3'],default='T1')
    for name in ['model-dir','tokenizer-dir','data-root','output']:s.add_argument('--'+name,type=Path)
    a=p.parse_args(argv)
    if a.command=='list':
        m=json.loads(files('classic_precision').joinpath('reproduction.json').read_text());v={k:m[k] for k in ['scope','protocols','code_groups','release_gaps']}
        current=json.loads(files('classic_precision').joinpath('current_x_protocols.json').read_text())
        v['current_x']={'manifest':'current_x_protocols.json','scope':current['scope'],
                        'variants':list(current['variants']),
                        'default_variant_by_model_task':current['default_variant_by_model_task']}
        v['historical_configuration_note']='native-default/native-original are preserved historical exports; use current-x for the current official X protocol.'
    else:
        try:v=configuration(a.model,a.protocol,a.model_dir,a.tokenizer_dir,a.data_root,a.task)
        except ValueError as e:p.exit(2,str(e)+'\n')
    text=json.dumps(v,ensure_ascii=False,indent=2)+'\n'
    if getattr(a,'output',None):
        with a.output.open('x') as f:f.write(text)
    else:print(text,end='')

if __name__=='__main__':main()
