"""Verify reusable generation followed by grading from immutable saved answers."""
import os
import urllib.request

COMMIT = '00248bdabe6c6caab5c272b5d6d250f848d26364'
os.environ.update(THINKINGCAP_REF=COMMIT, THINKINGCAP_MODE='smoke',
    THINKINGCAP_OFFLINE_VERIFY='1', THINKINGCAP_TIMEOUT_SECONDS='1800')
url = ('https://raw.githubusercontent.com/marek-krcal/ThinkingCap-Test/'
       + COMMIT + '/cloud/kaggle-vllm/run.py')
with urllib.request.urlopen(url, timeout=60) as response:
    source = response.read().decode('utf-8')
exec(compile(source, url, 'exec'), {'__name__': '__main__', '__file__': url})
