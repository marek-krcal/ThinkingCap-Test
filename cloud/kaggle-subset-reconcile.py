"""Disposable stdlib-only CPU reconciliation launcher; no model or GPU."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ref=os.environ['THINKINGCAP_REF']
if not re.fullmatch('[0-9a-f]{40}',ref): raise ValueError('Pin a full commit')
root=Path('/kaggle/temp/subset-reconcile-repo')
def run(command,cwd=None):
    subprocess.run(command,cwd=cwd,check=True,timeout=300)
run(['git','clone','--depth','1','--no-checkout','https://github.com/marek-krcal/ThinkingCap-Test.git',str(root)])
run(['git','fetch','--depth','1','origin',ref],root)
run(['git','checkout','--detach','FETCH_HEAD'],root)
os.environ['CUDA_VISIBLE_DEVICES']=''
dataset=os.environ['THINKINGCAP_OFFLINE_DATASET'].split('/')[-1]
plans=[p for p in Path('/kaggle/input').rglob('collection-plan.json') if dataset in p.parts]
if len(plans)!=1: raise ValueError('Attach exactly the selected private input')
import hashlib
if hashlib.sha256(plans[0].read_bytes()).hexdigest()!=os.environ['THINKINGCAP_PLAN_SHA']:
    raise ValueError('Selected plan changed')
command=[sys.executable,'-I',str(root/'cloud/reconcile_offline_subset.py'),'--data',str(plans[0].parent)]
for kernel in json.loads(os.environ['THINKINGCAP_REUSE_SOURCES']):
    slug=kernel.split('/')[-1]
    found=[p for p in Path('/kaggle/input').rglob('generation-manifest.json') if slug in p.parts]
    if len(found)!=1: raise ValueError('Missing/ambiguous saved source: '+kernel)
    command+=['--source',str(found[0].parent)]
run(command,root)
