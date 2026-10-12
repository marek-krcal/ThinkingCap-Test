"""Repair-only 400x1 evaluation of two saved adapters; no retraining."""
import os
import urllib.request

COMMIT = '4fa4a4604152c01de3e97567cba06ec788227fd1'
os.environ.update(THINKINGCAP_REF=COMMIT, THINKINGCAP_MODE='smoke',
                  THINKINGCAP_REEVAL_BINARY_RL='1', THINKINGCAP_BINARY_RL='0',
                  THINKINGCAP_TIMEOUT_SECONDS='2700')
url = ('https://raw.githubusercontent.com/marek-krcal/ThinkingCap-Test/'
       + COMMIT + '/cloud/kaggle-vllm/run.py')
with urllib.request.urlopen(url, timeout=60) as response:
    source = response.read().decode('utf-8')
exec(compile(source, url, 'exec'), {'__name__': '__main__', '__file__': url})
