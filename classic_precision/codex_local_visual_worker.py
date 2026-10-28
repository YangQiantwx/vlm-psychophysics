"""Audit-only vLLM worker for the separate Codex CLI visual experiment."""
import os
from pathlib import Path
from vllm.v1.worker.gpu_worker import Worker
from .embodied_admission import append, new
from .nvfp4_cosmos_official import _configure_torch, _engine_receipt, validate_engine_receipt


def prepare_qsa():
    """Import the reviewed lazy target before requiring an applied selector."""
    import importlib
    from .native_path_smoke import qsa_control
    from .embodied_admission import require
    from runtime_compat import classic_precision_qsa_determinism as qsa
    before = qsa_control(None)
    require(before and before['state'] in ('pending_target_import', 'applied'),
            'Reviewed QSA installation missing')
    target = importlib.import_module(qsa.TARGET_MODULE)
    after = qsa_control(None)
    require(after['state'] == 'applied'
            and getattr(target._topk, '_classic_precision_qsa_determinism', None) == after,
            'Actual QSA target does not carry the reviewed selector')
    return {'before': before, 'after': after}


class LocalVisualWorker(Worker):
    def load_model(self, *, load_dummy_weights=False):
        import torch
        if os.environ.get('CVPR_CLI_MODEL','cosmos')=='cosmos':_configure_torch(torch)
        else:
            # API startup can leave the native QSA module lazily unimported.
            # The original helper validates its source and installs the exact
            # same selector; no processor/model mathematics is changed here.
            new(Path(os.environ['CVPR_CLI_AUDIT_ROOT'])/'qsa-preload.json', prepare_qsa())
        return super().load_model(load_dummy_weights=load_dummy_weights)

    def compile_or_warm_up_model(self):
        result = super().compile_or_warm_up_model()
        out = Path(os.environ['CVPR_CLI_AUDIT_ROOT'])
        if os.environ.get('CVPR_CLI_MODEL','cosmos')=='cosmos':
            receipt = _engine_receipt(self)
            validate_engine_receipt([receipt])
        else:
            from collections import Counter
            from .native_path_smoke import qsa_control
            from .inference import _kv_capacity, _weight_receipt
            from .embodied_admission import require
            qsa=qsa_control(self)
            require(qsa['state']=='applied','QSA selector not actually applied')
            model=self.get_model();kernels=Counter()
            for name,module in model.named_modules():
                kernel=getattr(getattr(module,'quant_method',None),'kernel',None)
                if kernel is not None:kernels[type(kernel).__name__]+=1
            receipt={'model_class':type(model).__name__,'kernels':dict(kernels),
                     'qsa':qsa,'capacity':_kv_capacity(self),'weights':_weight_receipt(self)}
            require((receipt['capacity']['capacity_tokens'] or 0)>=98304,'Insufficient Qwen context capacity')
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
