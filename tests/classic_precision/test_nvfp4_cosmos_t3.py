from copy import deepcopy
import unittest

from classic_precision import nvfp4_cosmos_t3 as t


class ContractTests(unittest.TestCase):
    def setUp(self):
        ids = [f"t3_{i}" for i in range(1776)]
        self.catalog = {"items": {sid: {"sample_id": sid, "task": "T3", "row": {
            "source_parameters": {"angle_deg": .5, "tilt_sign": 1}, "prompt": "fixed"}}
            for sid in ids}}
        self.a = {"schema": t.SCHEMA + "/allocation", "reused_ids": ids[:144],
                  "new_assignments": dict(zip(("fourier0", "fourier1", "radon0"),
                                              (ids[144:688], ids[688:1232], ids[1232:]))),
                  "new_bf16_calls": 0, "automatic_retry": False}
        runtime = {"checkpoint": "/new", "original_checkpoint": "/old", "resources": {
            "min_available_ram_gib": 20, "min_free_gpu_gib": 80}, "physical_locks": {"0": {}, "1": {}}}
        self.base = {"runtime": runtime, "engine_settings": t.o.engine_settings(runtime),
                     "generation": t.o.inference.GENERATION}
        groups = [self.a["new_assignments"][k] for k in ("fourier0", "fourier1")]
        self.p = {"schema": t.SCHEMA, "host": "fourier", "shard_names": ["fourier0", "fourier1"],
                  "assignments": groups, "runtime": runtime, "generation": deepcopy(self.base["generation"]),
                  "engine_settings": deepcopy(self.base["engine_settings"]), "panel": {"items": [
                      {**deepcopy(self.catalog["items"][sid]), "selected_cell_id": "T3/angle=0.5/sign=1"}
                      for sid in sum(groups, [])]}, "max_new_calls": 1088, "max_new_calls_per_id": 1,
                  "max_total_attempts_per_id": 1, "automatic_retry": False, "new_bf16_calls": 0,
                  "deep_calls_per_fresh_worker": 2, "prior_failed_nvfp4_attempts": {}}

    def check(self):
        t.validate_contract(self.p, self.base, self.a, self.catalog)

    def test_full_coverage_and_exact_pilot_controls(self): self.check()
    def test_reused_id_cannot_be_launched(self):
        self.a["new_assignments"]["radon0"][0] = self.a["reused_ids"][0]
        with self.assertRaises(ValueError): self.check()
    def test_cross_host_overlap_rejected(self):
        self.a["new_assignments"]["radon0"][0] = self.a["new_assignments"]["fourier0"][0]
        with self.assertRaises(ValueError): self.check()
    def test_missing_id_rejected(self):
        self.a["new_assignments"]["radon0"].pop()
        with self.assertRaises(ValueError): self.check()
    def test_temperature_rejected(self):
        self.p["generation"]["temperature"] = .1
        with self.assertRaises(ValueError): self.check()
    def test_frame_sampling_rejected(self):
        self.p["engine_settings"]["mm_processor_kwargs"]["do_sample_frames"] = True
        with self.assertRaises(ValueError): self.check()
    def test_kv_precision_rejected(self):
        self.p["engine_settings"]["kv_cache_dtype"] = "fp8"
        with self.assertRaises(ValueError): self.check()
    def test_prompt_change_rejected(self):
        self.p["panel"]["items"][0]["row"]["prompt"] = "new"
        with self.assertRaises(ValueError): self.check()
    def test_order_change_rejected(self):
        self.p["panel"]["items"].reverse()
        with self.assertRaises(ValueError): self.check()
    def test_retry_rejected(self):
        self.p["automatic_retry"] = True
        with self.assertRaises(ValueError): self.check()
    def test_model_path_can_only_follow_runtime(self):
        self.p["runtime"] = deepcopy(self.p["runtime"])
        self.p["runtime"]["checkpoint"] = "/host2/same-asset"
        with self.assertRaises(ValueError): self.check()
        self.p["engine_settings"]["model"] = "/host2/same-asset"
        self.check()
    def test_radon_local_worker0_same_scientific_controls(self):
        self.p.update(host="radon", shard_names=["radon0"], assignments=[self.a["new_assignments"]["radon0"]], max_new_calls=544)
        self.p["runtime"] = deepcopy(self.p["runtime"]); self.p["runtime"]["physical_locks"] = {"0": {}}
        self.p["panel"]["items"] = [{**deepcopy(self.catalog["items"][sid]), "selected_cell_id": "T3/angle=0.5/sign=1"}
                                      for sid in self.p["assignments"][0]]
        self.check()


if __name__ == "__main__": unittest.main()
