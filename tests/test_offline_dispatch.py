import datetime
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from cloud.dispatch_offline import build_bundle, check_capacity
from cloud.refresh_offline import refresh
from utils.offline import digest


class DispatchTests(unittest.TestCase):
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
