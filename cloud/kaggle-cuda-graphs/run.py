"""One controlled 400-question decode-only CUDA graph trial."""
import os
import urllib.request

COMMIT = '6e726eaf9e0f5105355d06e07d2e41d066eceba4'
os.environ.update(THINKINGCAP_REF=COMMIT, THINKINGCAP_MODE='smoke',
                  THINKINGCAP_LIMIT_PER_BENCHMARK='100', THINKINGCAP_SAMPLES='1',
                  THINKINGCAP_TIMEOUT_SECONDS='3600')
url = ('https://raw.githubusercontent.com/marek-krcal/ThinkingCap-Test/'
       + COMMIT + '/cloud/kaggle-vllm/run.py')
with urllib.request.urlopen(url, timeout=60) as response:
    source = response.read().decode('utf-8')
exec(compile(source, url, 'exec'), {'__name__': '__main__', '__file__': url})
