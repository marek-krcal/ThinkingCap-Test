"""Pinned bootstrap for the bounded evaluation throughput campaign."""
import os
import urllib.request

revision = '0361dc6c9683a53e018b403692abb45ceb8d6c6c'
os.environ.update(THINKINGCAP_OPTIMIZE='1', THINKINGCAP_REF=revision,
                  THINKINGCAP_TIMEOUT_SECONDS='7200',
                  THINKINGCAP_TENSOR_PARALLEL_SIZE='2')
url = f'https://raw.githubusercontent.com/marek-krcal/ThinkingCap-Test/{revision}/cloud/kaggle-vllm/run.py'
with urllib.request.urlopen(url, timeout=60) as response:
    source = response.read()
exec(compile(source, url, 'exec'), globals())
