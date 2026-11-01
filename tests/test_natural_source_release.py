"""Consumer identity projection, with optional accepted-source integration.

Integration uses CVPR_NATURAL_BASELINE, CVPR_NATURAL_REVISION and
CVPR_NATURAL_COLOR_ASSETS. No video generation, decoding or upload is performed.
"""
from copy import deepcopy
import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest

from classic_precision.dataset_identity import IdentityRegistry, identity_digest
from classic_precision import natural_source_release as release


def fixture():
    payload = {'parent_sample_id':'old','reason':'source correction'}
    record = dict(sample_id='child',parent_sample_id='old',video_sha256='a'*64,identity_payload=payload)
    row = dict(sample_id='child',parent_sample_id='old',parent_is_alias=False,
               domain='natural',task='T2',dataset_revision=release.NAME,video_sha256='a'*64,
               path='T2/child_rev12345678.mp4',truth='LEFT',parameters={'delta_Y':.1},
               full_identity_sha256=release.digest(payload))
    return row,record


class ProjectionTests(unittest.TestCase):
    def test_schema_digest_fix_retains_recipe_digest_and_all_science(self):
        row,record = fixture(); before = deepcopy(row)
        with self.assertRaisesRegex(ValueError,'Full source identity digest differs'):
            IdentityRegistry.from_rows([row])
        projected = release.project_row(row,record)
        self.assertEqual(row,before)
        self.assertEqual(projected['revision_identity_payload_sha256'],before['full_identity_sha256'])
        self.assertEqual(projected['full_identity_sha256'],identity_digest('natural','T2','a'*64))
        changed = {'full_identity_sha256','revision_identity_payload_sha256','legacy_sample_id','legacy_path'}
        self.assertEqual({k:v for k,v in projected.items() if k not in changed},
                         {k:v for k,v in before.items() if k not in changed})

    def test_child_self_alias_resolves_and_parent_never_resolves(self):
        row,record = fixture(); projected = release.project_row(row,record)
        registry = IdentityRegistry.from_rows([projected])
        for key in ['child',row['path']]:self.assertEqual(registry.resolve(key,domain='natural').canonical_id,'child')
        self.assertEqual(projected['legacy_sample_id'],'child')
        self.assertEqual(projected['legacy_path'],row['path'])
        with self.assertRaises(KeyError):registry.resolve('old',domain='natural')

    def test_recipe_or_encoded_identity_mismatch_is_rejected(self):
        row,record = fixture(); record['identity_payload']['reason']='tampered'
        with self.assertRaises(AssertionError):release.project_row(row,record)
        row,record = fixture(); record['video_sha256']='b'*64
        with self.assertRaises(AssertionError):release.project_row(row,record)


ENV = all(os.environ.get(k) for k in ['CVPR_NATURAL_BASELINE','CVPR_NATURAL_REVISION','CVPR_NATURAL_COLOR_ASSETS'])


@unittest.skipUnless(ENV,'Set accepted-source integration locations')
class AcceptedSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(); cls.output = Path(cls.temp.name)
        cls.baseline = Path(os.environ['CVPR_NATURAL_BASELINE'])
        cls.revision = Path(os.environ['CVPR_NATURAL_REVISION'])
        cls.manifest = release.stage(cls.baseline,cls.revision,cls.output,os.environ['CVPR_NATURAL_COLOR_ASSETS'])
        cls.records = release.read(cls.revision/'source-records.json')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_all_235_returned_files_and_10416_identities_validate(self):
        m = self.manifest
        self.assertEqual(m['returned_metadata_files_SHA_verified'],235)
        self.assertEqual(m['total_natural_videos'],10416)
        self.assertEqual(m['new_source_videos'],200)
        self.assertEqual(m['unchanged_catalog_lines_byte_equal'],10216)
        for d in m['datasets']:
            dest = Path(d['staging'])
            self.assertEqual(len(IdentityRegistry.from_jsonl(dest/'catalog.jsonl').bindings),d['source_videos'])
            self.assertEqual(len(IdentityRegistry.from_csv(dest/'aliases.csv').bindings),d['source_videos'])

    def test_original_render_catalogs_remain_byte_exact_and_only_consumer_fields_change(self):
        audit = release.read(self.revision/'source-audit.json')
        allowed = {'full_identity_sha256','revision_identity_payload_sha256','legacy_sample_id','legacy_path'}
        ids = {r['sample_id'] for r in self.records}
        for task in ['T2','T3']:
            source = self.revision/'catalogs'/(task+'.jsonl')
            self.assertEqual(release.sha(source),audit['final_catalog_sha256'][task])
            view = self.output/release.REPOSITORIES[task]/'catalog.jsonl'
            for before,after in zip(release.lines(source),release.lines(view)):
                a,b = json.loads(before),json.loads(after)
                if a['sample_id'] not in ids:self.assertEqual(before,after)
                else:
                    self.assertEqual({k:v for k,v in a.items() if k not in allowed},
                                     {k:v for k,v in b.items() if k not in allowed})
                    self.assertEqual(b['revision_identity_payload_sha256'],a['full_identity_sha256'])
                    self.assertEqual(b['full_identity_sha256'],identity_digest('natural',task,b['video_sha256']))

    def test_original_full_frame_qa_is_retained_exactly_for_each_new_source(self):
        for r in self.records:
            original = self.revision/'qa'/(r['sample_id']+'.json')
            copied = self.output/release.REPOSITORIES[r['task']]/'validation/source-revision-20261101/qa'/(r['sample_id']+'.json')
            self.assertEqual(original.read_bytes(),copied.read_bytes())
        self.assertEqual(sum(len(d['videos']) for d in self.manifest['datasets']),200)

    def test_assets_only_change_for_the_three_profile_corrections(self):
        changes = set()
        for task in ['T2','T3']:
            old = release.read(self.baseline/release.REPOSITORIES[task]/'source-assets.json')
            new = release.read(self.output/release.REPOSITORIES[task]/'source-assets.json')
            for a,b in zip(old['assets'],new['assets']):
                if a==b:continue
                changes.add(a['source_id'])
                self.assertEqual(a['sha256'],b['parent_source_sha256'])
                self.assertEqual(a['source_url'],b['source_url'])
                self.assertEqual(a['raw_sha256'],b['raw_sha256'])
                self.assertEqual(release.sha(self.output/release.REPOSITORIES[task]/b['path']),b['sha256'])
        self.assertEqual(changes,{'photo_obj_025','photo_obj_033','photo_obj_048'})

    def test_existing_standalone_reader_selects_representative_new_id_and_path(self):
        # Media remains on the workstation. Reaching its precise missing path
        # confirms reader selection/schema compatibility, not media decoding.
        for task in ['T2','T3']:
            spec = importlib.util.spec_from_file_location('reader_'+task,
                self.baseline/release.REPOSITORIES[task]/'read_example.py')
            reader = importlib.util.module_from_spec(spec); spec.loader.exec_module(reader)
            dest = self.output/release.REPOSITORIES[task]
            row = next(r for r in self.records if r['task']==task)
            for value in [row['sample_id'],row['proposed_path']]:
                with self.assertRaises(FileNotFoundError) as raised: reader.read_sample(dest,value)
                self.assertEqual(raised.exception.filename,str(dest/row['proposed_path']))


if __name__=='__main__':unittest.main()
