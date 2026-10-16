import json
from pathlib import Path
import tempfile
import unittest
from cloud.report_offline_scope import report


class ScopeTests(unittest.TestCase):
    def test_legacy_out_of_scope_training_not_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for split in ('test','validation','train'):
                data=root/'data'/('offline-subset' if split=='train' else 'offline-input');data.mkdir(parents=True,exist_ok=True)
                (data/(split+'.jsonl')).write_text(json.dumps(dict(id=split,seed_index=1))+'\n')
            for name,split in (('offline-campaign','test'),('offline-subset-campaign','train')):
                state=root/'runs'/name;folder=state/'a'/'collection';folder.mkdir(parents=True)
                (folder/'collection-index.json').write_text(json.dumps(dict(completed_keys=[[split,0]])))
                label=dict(id=split,sample_index=0,seed=10001,correct=True,truncated=False,output_tokens=2)
                (folder/'labels.jsonl').write_text(json.dumps(label)+'\n')
                attempts=[dict(job_id=split+'-000',kernel='kernel/'+split,state='partial',artifacts=str(folder.parent.relative_to(root)))]
                if name=='offline-campaign':
                    attempts.append(dict(job_id='train-000',kernel='old',state='partial',artifacts='does-not-exist'))
                (state/'ledger.json').write_text(json.dumps(dict(attempts=attempts)))
            result=report(root)
            self.assertEqual(result['target_answers'],24)
            self.assertEqual(result['saved_answers'],2)
            self.assertEqual(result['graded_answers'],2)
            self.assertFalse(result['full_coverage_verified'])
