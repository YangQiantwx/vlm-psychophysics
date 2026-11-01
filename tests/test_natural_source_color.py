"""CPU checks for source-profile conversion and finite revision acceptance.

Set CVPR_BENCHMARK_ROOT and CVPR_COLOR_REVISION to additionally inspect the
prepared real 300-asset/200-video candidate. These tests never encode video.
"""
from copy import deepcopy
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageCms

from classic_precision import natural_source_color as color
from classic_precision import natural_source_revision as revision


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        self.array = np.array([[[120,80,21,0],[204,137,45,1]],
                               [[43,80,170,127],[255,254,1,255]]], dtype=np.uint8)

    def test_alpha_and_invisible_rgb_remain_exact(self):
        original = self.array.copy()
        image, profile = color.convert_rgba(Image.fromarray(original),self.profile)
        result = np.asarray(image)
        np.testing.assert_array_equal(result[...,3],original[...,3])
        np.testing.assert_array_equal(result[original[...,3]==0],original[original[...,3]==0])
        np.testing.assert_array_equal(self.array,original)
        self.assertEqual(color.profile_info(profile)['profile_status'],'valid')

    def test_uses_explicit_official_default_transform(self):
        image = Image.fromarray(self.array)
        result,_ = color.convert_rgba(image,self.profile)
        direct = ImageCms.profileToProfile(image.convert('RGB'),ImageCms.ImageCmsProfile(BytesIO(self.profile)),
                  ImageCms.createProfile('sRGB'),renderingIntent=0,outputMode='RGB',flags=0)
        visible = self.array[...,3]>0
        np.testing.assert_array_equal(np.asarray(result)[...,:3][visible],np.asarray(direct)[visible])

    def test_output_profile_and_pixels_are_deterministic(self):
        one,p1 = color.convert_rgba(Image.fromarray(self.array),self.profile)
        two,p2 = color.convert_rgba(Image.fromarray(self.array),self.profile)
        self.assertEqual(p1,p2)
        self.assertEqual(one.tobytes(),two.tobytes())
        self.assertEqual(p1[84:100],bytes(16))

    def test_missing_or_invalid_profile_cannot_be_silently_converted(self):
        for blob in [None,b'bad ICC']:
            with self.assertRaises(ValueError):color.convert_rgba(Image.fromarray(self.array),blob)
        self.assertEqual(color.profile_info(None)['profile_status'],'missing')
        self.assertEqual(color.profile_info(b'bad ICC')['profile_status'],'invalid')

    def test_preserves_untagged_assets_without_guessing_profile(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); path = root/'assets/library/a.png'; path.parent.mkdir(parents=True)
            Image.fromarray(self.array).save(path); before = path.read_bytes()
            row = color.inspect_asset((root,dict(id='gpt_obj_001',path='a.png',sha256=color.sha(path))))
            self.assertEqual(row['action'],'retain_original')
            self.assertEqual(row['profile_status'],'missing')
            self.assertEqual(path.read_bytes(),before)

    def test_macos_cannot_start_batch_video_generation_or_decode(self):
        with patch.object(revision.sys,'platform','darwin'):
            with self.assertRaisesRegex(AssertionError,'execution workstation'):
                revision.execute('/missing','/missing',4,render=True)


@unittest.skipUnless(os.environ.get('CVPR_BENCHMARK_ROOT') and os.environ.get('CVPR_COLOR_REVISION'),
                     'Set source and candidate locations for integration')
class RealCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ['CVPR_BENCHMARK_ROOT'])
        cls.out = Path(os.environ['CVPR_COLOR_REVISION'])
        cls.plan = revision.check_plan(cls.root,cls.out)
        cls.color = color.read(cls.out/cls.plan['color_dir']/'color-repair.json')

    def test_300_assets_audited_only_three_converted(self):
        self.assertEqual(self.color['asset_count'],300)
        self.assertEqual(self.color['unchanged_assets'],297)
        self.assertEqual({a['asset_id'] for a in self.color['converted_assets']},color.ASSETS)
        for asset in self.color['converted_assets']:
            self.assertTrue(asset['alpha_byte_equal'])
            self.assertTrue(asset['foreground_original_raw_RGB_byte_equal'])
            self.assertTrue(asset['T2_preparation']['alpha_exactly_unchanged'])

    def test_exact_current_184_and_minimum_16_form_one_200_revision(self):
        changes = self.plan['changes']
        self.assertEqual(len(changes),200)
        self.assertEqual(sum(r['reason']=='ICC' for r in changes),184)
        self.assertEqual(sum(r['reason']=='local-balance' for r in changes),16)
        self.assertEqual(self.plan['total_natural_videos'],10416)
        self.assertTrue(all(r['task']!='T1' for r in changes))
        for r in changes:
            self.assertNotEqual(r['sample_id'],r['parent_sample_id'])
            self.assertFalse(r['parent_is_alias'])
            self.assertNotEqual(r['proposed_path'],r['parent']['path'])
        self.assertEqual(len({r['sample_id'] for r in changes}),200)
        self.assertTrue({r['sample_id'] for r in changes}.isdisjoint(r['parent_sample_id'] for r in changes))

    def test_final_feature_and_all_patch_geometry_gates_pass(self):
        gates = self.plan['final_feature_gates']
        self.assertTrue(gates['T2']['PASS'] and gates['T3']['PASS'])
        for row in self.plan['T3_geometry']:
            self.assertTrue(row['alpha_byte_equal'])
            self.assertLessEqual(abs(row['axis_error_deg']),.01)

    def test_representative_each_asset_and_both_tasks_pixel_banks(self):
        selected = {}
        for c in self.plan['changes']:
            selected.setdefault((c['task'],c['object_id'],c['reason']),c)
        self.assertEqual(len(selected),8)  # three ICC T2, one ICC T3, four reassigned T2 objects
        for c in selected.values():
            banks,indices,metric = revision.banks(self.root,self.out,self.plan,c)
            self.assertEqual(len(indices),180)
            self.assertEqual(sum(indices),30)
            self.assertEqual(banks[0].shape,(512,768,3))
            self.assertEqual(banks[0].dtype,np.uint8)
            self.assertFalse(np.array_equal(banks[0],banks[1]))
            if c['task']=='T2':
                self.assertLess(abs(metric['decoded_delta_Y']-c['parent']['parameters']['delta_Y']),
                                min(.002,.02*c['parent']['parameters']['delta_Y']))
            else:self.assertLessEqual(abs(metric['actual_angle_deg']-c['parent']['parameters']['angle']),.02)

    def test_tampered_plan_or_prepared_array_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'plan.json'; plan = deepcopy(self.plan); plan['new_videos']=201
            path.write_text(json.dumps(plan))
            with self.assertRaises(AssertionError): revision.check_plan(self.root,td)
            # A self-consistent plan still cannot admit altered prepared bytes.
            plan = deepcopy(self.plan); plan['prepared_file_sha256'] = {'array.npz':'0'*64}
            (Path(td)/'array.npz').write_bytes(b'changed prepared bytes')
            plan.pop('plan_sha256'); plan['plan_sha256'] = color.digest(plan)
            path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(AssertionError,'Prepared file changed'):
                revision.check_plan(self.root,td)


if __name__ == '__main__':
    unittest.main()
