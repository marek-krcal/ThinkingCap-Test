"""Bounded SimPO/RLOO test-overfit experiment, not a submission result."""
import os
import urllib.request

COMMIT = '2a905b1d8d047fbd9d815b817c7aa80aa6880166'
os.environ.update(THINKINGCAP_REF=COMMIT, THINKINGCAP_MODE='smoke',
                  THINKINGCAP_BINARY_RL='1', THINKINGCAP_TIMEOUT_SECONDS='5400')
url = ('https://raw.githubusercontent.com/marek-krcal/ThinkingCap-Test/'
       + COMMIT + '/cloud/kaggle-vllm/run.py')
with urllib.request.urlopen(url, timeout=60) as response:
    source = response.read().decode('utf-8')
exec(compile(source, url, 'exec'), {'__name__': '__main__', '__file__': url})
