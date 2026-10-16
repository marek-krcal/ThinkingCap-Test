"""Small disposable CPU-only grading environment; all outputs remain on Kaggle."""
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

scratch = Path('/kaggle/temp/thinkingcap-cpu')
scratch.mkdir(parents=True, exist_ok=True)
root = scratch/'repo'
started = time.monotonic()
deadline = started+int(os.environ.get('THINKINGCAP_TIMEOUT_SECONDS','3600'))
timings = {}


def run(label, command, cwd=None):
    before = time.monotonic()
    process = subprocess.Popen(command,cwd=cwd,start_new_session=True)
    try:
        code = process.wait(timeout=max(1,deadline-before))
        if code: raise subprocess.CalledProcessError(code,command)
    except BaseException:
        try: os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError: pass
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL); process.wait()
        raise
    timings[label] = time.monotonic()-before
    Path('/kaggle/working/setup-timing.json').write_text(json.dumps(timings, indent=2)+'\n')


ref = os.environ['THINKINGCAP_REF']
os.environ['UV_CACHE_DIR'] = str(scratch/'uv-cache')
os.environ['NLTK_DATA'] = str(scratch/'nltk_data')
run('clone',['git','clone','--depth','1','--no-checkout','https://github.com/marek-krcal/ThinkingCap-Test.git',str(root)])
run('fetch',['git','fetch','--depth','1','origin',ref],root)
run('checkout',['git','checkout','--detach','FETCH_HEAD'],root)
run('installer',[sys.executable,'-m','pip','install','-q','uv'])
python = str(scratch/'env/bin/python')
run('environment',['uv','venv','--python','3.12',str(scratch/'env')])
run('dependencies',['uv','pip','install','--python',python,'-r','requirements-grade.txt'],root)
cache = next(Path('/kaggle/input').rglob('eval-cache/questions/manifest.json'))
shutil.copytree(cache.parent,root/'questions')
old_nltk = cache.parent.parent/'nltk_data'
if old_nltk.exists(): shutil.copytree(old_nltk,Path(os.environ['NLTK_DATA']))
run('grade_saved',[python,'cloud/grade_collection.py'],root)
print('CPU-only saved-answer grading completed',flush=True)
