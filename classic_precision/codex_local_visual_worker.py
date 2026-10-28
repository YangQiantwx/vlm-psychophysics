"""Audit-only vLLM worker for the separate Codex CLI visual experiment."""
import os
from pathlib import Path
from vllm.v1.worker.gpu_worker import GPUWorker
from .embodied_admission import append, new
from .nvfp4_cosmos_official import _configure_torch, _engine_receipt, validate_engine_receipt


class LocalVisualWorker(GPUWorker):
    def load_model(self, *, load_dummy_weights=False):
        import torch
        _configure_torch(torch)
        return super().load_model(load_dummy_weights=load_dummy_weights)

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        receipt = _engine_receipt(self)
        validate_engine_receipt([receipt])
        out = Path(os.environ['CVPR_CLI_AUDIT_ROOT'])
        new(out/'engine-receipt.json', receipt)
        def observe(module, args, kwargs):
            x = args[0]
            grid = kwargs.get('grid_thw')
            append(out/'actual-vision-inputs.jsonl', {
                'shape': list(x.shape), 'dtype': str(x.dtype), 'device': str(x.device),
                'grid_thw': grid.tolist() if grid is not None else None})
        self._cli_visual_observation = self.get_model().visual.register_forward_pre_hook(
            observe, with_kwargs=True)
        return result
