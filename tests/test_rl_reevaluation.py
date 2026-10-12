import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cloud.reevaluate_rl import find_adapter


class SavedAdapterVerification(unittest.TestCase):
    def test_only_exact_saved_trained_adapter_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root/'source/runs/binary-rl-overfit/simpo/training/adapter'
            adapter.mkdir(parents=True)
            (adapter/'adapter_model.safetensors').write_bytes(b'test weights')
            summary = adapter.parent/'training-summary.json'
            summary.write_text(json.dumps({'steps': 100, 'skipped_nonfinite_updates': 0}))
            digest = hashlib.sha256(b'test weights').hexdigest()
            with patch.dict('cloud.reevaluate_rl.EXPECTED', {'simpo': digest}):
                self.assertEqual(find_adapter(root, 'simpo')[0], adapter)
                summary.write_text(json.dumps({'steps': 0, 'skipped_nonfinite_updates': 0}))
                with self.assertRaises(RuntimeError):
                    find_adapter(root, 'simpo')
            with self.assertRaises(RuntimeError):
                find_adapter(root, 'simpo')
