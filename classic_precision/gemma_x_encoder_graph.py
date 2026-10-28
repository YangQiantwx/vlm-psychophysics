"""Observe actual official encoder CUDA graph replay, without changing math."""
from __future__ import annotations

from . import official_x_runner as base
from .official_x_gemma_adapter import validate_vision as validate_base

_install = base.install_probe
_read = base.read_probe


def manager(worker):
    return worker.model_runner.model_state.encoder_runner.cudagraph_manager


class Restore:
    def __init__(self, obj, value):
        self.obj, self.value = obj, value

    def remove(self):
        self.obj._run_budget_graph = self.value


def install_probe(worker, spec, model):
    result = _install(worker, spec, model)
    mgr = manager(worker)
    base.require(mgr is not None and mgr.is_captured(), 'Official encoder graphs not captured')
    state = worker._official_x_probe
    state['encoder_graph_before'] = mgr.get_cumulative_stats()
    original = mgr._run_budget_graph

    def observed(mm_kwargs, token_budget, path='default', axis_keys=()):
        before = mgr.graph_hits
        output = original(mm_kwargs, token_budget, path, axis_keys)
        if mgr.graph_hits > before:
            key = token_budget if not axis_keys else (token_budget, axis_keys)
            graph = mgr._get_graph_set(path)[key]
            modality = mgr.model.get_input_modality(mm_kwargs)
            source_key = 'pixel_values_videos' if modality == 'video' else 'pixel_values'
            count = int(mm_kwargs[source_key].shape[0])
            actual = graph.input_buffers['pixel_values'][:count]
            index = len(state['outer'])
            row = {'tensor': base.tensor_description(actual, True), 'grid_thw': None,
                   'num_frames_argument': count if modality == 'video' else None,
                   'source_modality': modality, 'actual_encoder_graph_replay': True,
                   'token_budget': token_budget, 'axis_keys': list(axis_keys)}
            state['outer'].append(row)
            state['normalized'].append({**row, 'outer_index': index})
        return output

    mgr._run_budget_graph = observed
    state['handles'].append(Restore(mgr, original))
    return {**result, 'encoder_graph_captured': True,
            'encoder_graph_stats': state['encoder_graph_before']}


def read_probe(worker, model):
    worker._official_x_probe['encoder_graph_after'] = manager(worker).get_cumulative_stats()
    return _read(worker, model)


def bootstrap(profile):
    # Only the audit callbacks sent by RPC are replaced; no model function is.
    base.install_probe = install_probe
    base.read_probe = read_probe
    return {'audit_only': True, 'model_math_changed': False}


def validate_vision(profile, item, deep):
    result = validate_base(profile, item, deep)
    video_frames = sum(row['tensor']['shape'][0] for w in deep for row in w['normalized']
                       if row.get('actual_encoder_graph_replay') and row.get('source_modality') == 'video')
    base.require(video_frames == 180, 'All 180 video frames must actually use encoder graph replay')
    return {**result, 'actual_encoder_graph_video_frames': video_frames}
