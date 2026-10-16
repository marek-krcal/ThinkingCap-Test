import datetime
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from cloud.dispatch_offline import build_bundle, check_capacity


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
