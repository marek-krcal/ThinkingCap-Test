"""Pinned <=20-minute batched RLOO repeat followed by 400x1 evaluation."""
import os
import urllib.request

COMMIT = '26d1c2f1bf700eeda2d2454a420052089c846728'
os.environ.update(THINKINGCAP_REF=COMMIT, THINKINGCAP_MODE='smoke',
                  THINKINGCAP_BATCHED_RL='1', THINKINGCAP_BINARY_RL='0',
                  THINKINGCAP_REEVAL_BINARY_RL='0', THINKINGCAP_TIMEOUT_SECONDS='3600')
url = ('https://raw.githubusercontent.com/marek-krcal/ThinkingCap-Test/'
       + COMMIT + '/cloud/kaggle-vllm/run.py')
with urllib.request.urlopen(url, timeout=60) as response:
    source = response.read().decode('utf-8')
exec(compile(source, url, 'exec'), {'__name__': '__main__', '__file__': url})
