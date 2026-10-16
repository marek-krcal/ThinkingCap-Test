import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from utils.offline import digest, read_generation


class ReconcileTests(unittest.TestCase):
    def fixture(self,root):
        data=root/'input'; data.mkdir(); source=root/'old'/'raw';source.mkdir(parents=True)
        q=dict(id='gsm8k/1',dataset='gsm8k',seed_index=123,messages=[dict(role='user',content='2+2?')],
               answer='4',task='math',split='train',dataset_revision='frozen')
        (data/'train.jsonl').write_text(json.dumps(q)+'\n')
        plan=dict(collection_splits=['train'],model='qwen',revision='pinned',sampling={'temperature':.6},
                  jobs=[dict(id='train-000',start=0,stop=1,expected_answers=8)],
                  data=dict(files={'train.jsonl':dict(sha256=digest(data/'train.jsonl'))}))
        (data/'collection-plan.json').write_text(json.dumps(plan))
        record=dict(id=q['id'],dataset='gsm8k',sample_index=2,seed=9001+123000+2,
                    completion_tokens=[1],output_tokens=1,finish_reason='stop',truncated=False,text='Final answer: 4')
        (source/'prompts.jsonl').write_text(json.dumps(dict(q,prompt_tokens=[5]))+'\n')
        (source/'records.jsonl').write_text(json.dumps(record)+'\n')
        manifest=dict(kind='train',status='partial',arguments=dict(samples=8,seed=9001,model='qwen',revision='pinned',dtype='half'),
                      sampling=plan['sampling'],caps={'gsm8k':4096},expected_answers=8,completed_answers=1,
                      prompts_sha256=digest(source/'prompts.jsonl'),records_sha256=digest(source/'records.jsonl'))
        (source/'generation-manifest.json').write_text(json.dumps(manifest))
        label={k:record.get(k) for k in ('id','sample_index','seed','dataset','output_tokens','truncated','boundary_error','finish_reason','stop_reason')}
        label['correct']=True
        (source.parent/'labels.jsonl').write_text(json.dumps(label)+'\n')
        return data,source

    def run_reconcile(self,data,source,out,extra=()):
        script=Path(__file__).resolve().parents[1]/'cloud/reconcile_offline_subset.py'
        return subprocess.run([sys.executable,'-I',str(script),'--data',str(data),'--source',str(source),
                               '--out',str(out),*extra],capture_output=True,text=True)

    def test_cpu_import_keeps_exact_tokens_seed_and_skip_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data,source=self.fixture(root);before=digest(source/'records.jsonl')
            result=self.run_reconcile(data,source,root/'out')
            self.assertEqual(result.returncode,0,result.stderr)
            imported=root/'out/bootstrap/train-000'
            index=json.loads((imported/'collection-index.json').read_text())
            self.assertEqual(index['completed_keys'],[['gsm8k/1',2]])
            self.assertEqual(index['expected_answers'],8)
            self.assertEqual(read_generation(imported/'raw')[2][0]['seed'],132003)
            self.assertEqual(before,digest(source/'records.jsonl'))
            self.assertEqual(json.loads((root/'out/reuse-provenance.json').read_text())['regenerated_answers'],0)

    def test_changed_prompt_or_duplicate_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data,source=self.fixture(root)
            result=self.run_reconcile(data,source,root/'duplicate',['--source',str(source)])
            self.assertNotEqual(result.returncode,0)
            q=json.loads((data/'train.jsonl').read_text());q['messages'][0]['content']='different'
            (data/'train.jsonl').write_text(json.dumps(q)+'\n')
            plan=json.loads((data/'collection-plan.json').read_text())
            plan['data']['files']['train.jsonl']['sha256']=digest(data/'train.jsonl')
            (data/'collection-plan.json').write_text(json.dumps(plan))
            result=self.run_reconcile(data,source,root/'changed')
            self.assertNotEqual(result.returncode,0)
            self.assertIn('identity changed',result.stderr)
