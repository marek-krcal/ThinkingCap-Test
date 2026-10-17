import datetime
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from cloud.dispatch_offline import build_bundle, check_capacity, resume_parent, verify_resume_cache
from cloud.dispatch_offline_grade import build_launcher
from cloud.refresh_offline import refresh
from utils.offline import digest


class DispatchTests(unittest.TestCase):
    def test_minimal_resume_preserves_parent_and_frozen_recipe(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'minimal'
            build_bundle(folder, 'train-004', 4, 'a'*40, 'b'*64, 'owner/parent',
                         2700, 3600, minimal_resume_inputs=True)
            metadata = json.loads((folder/'kernel-metadata.json').read_text())
            self.assertEqual(metadata['kernel_sources'], ['owner/parent'])
            source = (folder/'run.py').read_text()
            self.assertIn('2700', source)
            self.assertIn('b'*64, source)
            with self.assertRaisesRegex(ValueError, 'audited parent'):
                build_bundle(Path(tmp)/'invalid', 'train-004', 4, 'a'*40, 'b'*64,
                             None, 2700, 3600, minimal_resume_inputs=True)

    def test_minimal_resume_checks_all_metadata_pages_without_downloading(self):
        api = MagicMock()
        call = api.build_kaggle_client.return_value.__enter__.return_value.kernels.kernels_api_client.list_kernel_session_output
        call.side_effect = [
            SimpleNamespace(files=[SimpleNamespace(file_name='eval-cache/questions/manifest.json')], next_page_token='page2'),
            SimpleNamespace(files=[SimpleNamespace(file_name='collection/collection-index.json')], next_page_token=None)]
        verify_resume_cache(api, 'owner/parent')
        self.assertEqual(call.call_count, 2)
        api.kernels_output.assert_not_called()

    def test_minimal_resume_rejects_missing_cache(self):
        api = MagicMock()
        call = api.build_kaggle_client.return_value.__enter__.return_value.kernels.kernels_api_client.list_kernel_session_output
        call.return_value = SimpleNamespace(files=[], next_page_token=None)
        with self.assertRaisesRegex(RuntimeError, 'cache manifest'):
            verify_resume_cache(api, 'owner/parent')

    def test_cpu_launcher_logs_before_bounded_initial_fetch(self):
        source = build_launcher('a'*40)
        compile(source, '<cpu-launcher>', 'exec')
        self.assertLess(source.index('CPU grading launcher starting'), source.index('subprocess.run'))
        self.assertIn('timeout=90', source)
        self.assertIn('a'*40, source)
        self.assertNotIn('import torch', source)
        self.assertNotIn('import vllm', source)

    def test_cpu_launcher_rejects_unpinned_ref(self):
        with self.assertRaises(ValueError):
            build_launcher('main')

    def test_launcher_pins_private_input_full_commit_and_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'submission'
            kernel = build_bundle(folder, 'train-000', 1, 'a'*40, 'b'*64, None, 2700, 3600)
            meta = json.loads((folder/'kernel-metadata.json').read_text())
            self.assertEqual(kernel, 'krcalkrcal/thinkingcap-offline-train-000-a1')
            self.assertEqual(meta['id'], kernel)
            self.assertEqual(meta['is_private'], 'true')
            source = (folder/'run.py').read_text()
            compile(source, str(folder/'run.py'), 'exec')
            self.assertIn('a'*40, source)
            self.assertIn('b'*64, source)
            self.assertIn('2700', source)
            self.assertIn('Collection launcher starting',source)
            self.assertIn('timeout=90',source)

    def test_selected_launcher_has_distinct_namespace_and_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/'selected'
            kernel=build_bundle(folder,'train-000',2,'a'*40,'b'*64,'old',2700,3600,
                                'krcalkrcal/thinkingcap-offline-subset-input','ThinkingCap offline subset')
            self.assertEqual(kernel,'krcalkrcal/thinkingcap-offline-subset-train-000-a2')
            metadata=json.loads((folder/'kernel-metadata.json').read_text())
            self.assertEqual(metadata['dataset_sources'],['krcalkrcal/thinkingcap-offline-subset-input'])
            self.assertIn('old',metadata['kernel_sources'])

    def api(self, active=(), used=1000, reserved=0):
        api = MagicMock()
        gpu = SimpleNamespace(total_time_allowed=datetime.timedelta(seconds=21600),
            time_used=datetime.timedelta(seconds=used), time_reserved=datetime.timedelta(seconds=reserved))
        client = api.build_kaggle_client.return_value.__enter__.return_value
        client.kernels.kernels_api_client.get_accelerator_quota_statistics.return_value = SimpleNamespace(
            gpu_quota=gpu, quota_refresh_time='next Saturday')
        api.kernels_list.return_value = [SimpleNamespace(ref=x) for x in active]
        api.kernels_status.return_value = SimpleNamespace(status=SimpleNamespace(name='RUNNING'))
        return api

    def test_two_sessions_or_insufficient_quota_prevent_submission(self):
        with self.assertRaisesRegex(RuntimeError, 'Two account sessions'):
            check_capacity(self.api(('one','two')), {'attempts':[]}, 3600)
        with self.assertRaisesRegex(RuntimeError, 'Wait for quota reset'):
            check_capacity(self.api(used=20000), {'attempts':[]}, 3600)

    def test_local_outstanding_budget_not_double_counted_with_server_reservation(self):
        ledger = {'attempts':[dict(kernel='one',timeout=3600)]}
        result = check_capacity(self.api(('one',),used=15000,reserved=3600), ledger, 2000)
        self.assertEqual(result['reserved'],3600)

    def test_omitted_gpu_metadata_and_not_yet_listed_push_still_occupy_slots(self):
        api = self.api(('one','two'))
        api.kernels_list.return_value = [SimpleNamespace(ref=x,enable_gpu=False) for x in ('one','two')]
        with self.assertRaisesRegex(RuntimeError, 'Two account sessions'):
            check_capacity(api, {'attempts':[]}, 3600)
        ledger = {'attempts':[dict(kernel=x,state='submitted',timeout=3600) for x in ('one','two')]}
        with self.assertRaisesRegex(RuntimeError, 'Two account sessions'):
            check_capacity(self.api(), ledger, 3600)

    def test_pending_resume_keeps_previous_saved_and_graded_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data=root/'data'; data.mkdir(); state=root/'state'; state.mkdir()
            (data/'collection-plan.json').write_text(json.dumps(dict(expected_answers=8,jobs=[dict(id='train-000')])))
            artifacts=root/'old-artifacts'; (artifacts/'collection').mkdir(parents=True)
            labels=[dict(id='x',sample_index=i,correct=True,truncated=False,output_tokens=2) for i in range(3)]
            (artifacts/'collection/labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in labels))
            ledger=dict(plan_sha256=digest(data/'collection-plan.json'),attempts=[
                dict(job_id='train-000',kernel='old',state='partial',completed_answers=3,new_answers=3,artifacts=str(artifacts)),
                dict(job_id='train-000',kernel='new',state='submitted',completed_answers=3,new_answers=0)])
            (state/'ledger.json').write_text(json.dumps(ledger))
            api=MagicMock(); api.kernels_status.return_value=SimpleNamespace(status=SimpleNamespace(name='RUNNING'))
            result=refresh(api,data,state)
            self.assertEqual(result['progress']['saved_answers'],3)
            self.assertEqual(result['progress']['graded_answers'],3)

    def test_draft_404_requires_fresh_explicit_inactive_acknowledgement(self):
        import requests
        api=self.api()
        api.kernels_list.return_value=[SimpleNamespace(ref='draft',current_version_number=None)]
        response=requests.Response(); response.status_code=404
        api.kernels_status.side_effect=requests.HTTPError(response=response)
        with self.assertRaises(requests.HTTPError):
            check_capacity(api, {'attempts':[]}, 3600)
        result=check_capacity(api, {'attempts':[]}, 3600, ['draft'])
        self.assertEqual(result['active'],[])
        self.assertEqual(result['verified_inactive_drafts'],['draft'])
        api.kernels_list.return_value=[SimpleNamespace(ref='draft',current_version_number=0)]
        self.assertEqual(check_capacity(api, {'attempts':[]}, 3600, ['draft'])['active'],[])
        api.kernels_list.return_value=[SimpleNamespace(ref='draft',current_version_number=1)]
        with self.assertRaises(requests.HTTPError):
            check_capacity(api, {'attempts':[]}, 3600, ['draft'])

    def test_startup_retry_requires_stopped_zero_output_and_audited_parent(self):
        parent=dict(state='partial',kernel='bootstrap',completed_answers=8)
        stopped=dict(state='cancelled_startup',kernel='stalled',new_answers=0,
            completed_answers=8,parent_kernel='bootstrap',remote_status='CANCEL_ACKNOWLEDGED',verified_output_bytes=0)
        self.assertEqual(resume_parent([parent,stopped]),parent)
        for field,value in (('remote_status','RUNNING'),('new_answers',1),('verified_output_bytes',5),('parent_kernel','other')):
            with self.subTest(field=field),self.assertRaises(ValueError):
                resume_parent([parent,dict(stopped,**{field:value})])
