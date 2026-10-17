"""CPU-only startup/input diagnostic. No model imports, generation, or grading."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request


def main():
    print('Offline startup diagnostic entered Python (CPU only)', flush=True)
    started = time.monotonic()
    root = Path('/kaggle/input')
    parent = root/'notebooks/krcalkrcal/thinkingcap-offline-subset-train-004-a2'
    data = root/'datasets/krcalkrcal/thinkingcap-offline-subset-input'
    report = {'cpu_only': True, 'generated_answers': 0, 'graded_answers': 0,
              'trained': False, 'checks': {}}
    for label, path in (
        ('selected_plan', data/'collection-plan.json'),
        ('selected_train', data/'train.jsonl'),
        ('parent_index', parent/'collection/collection-index.json'),
        ('parent_cache_manifest', parent/'eval-cache/questions/manifest.json')):
        print('Checking mounted input:', label, flush=True)
        content = path.read_bytes()
        report['checks'][label] = {'bytes': len(content),
            'sha256': hashlib.sha256(content).hexdigest()}
    assert report['checks']['selected_train']['sha256'] == 'e8d86ce8fb6c753864d88b752ac8fbe0f0f54f0b19cb417e06a69c81ed031543'
    assert report['checks']['selected_plan']['sha256'] == '860f6c66a5b75d1dc702f47720075830984520994ad88b1e72ffa7ebe385e8e2'
    index = json.loads((parent/'collection/collection-index.json').read_text())
    assert index['cumulative_answers'] == 1241
    url = 'https://raw.githubusercontent.com/marek-krcal/ThinkingCap-Test/2c806092bac7d48c08a103f78dd674e591958ae5/cloud/kaggle-vllm/run.py'
    print('Checking pinned initial fetch without executing fetched code', flush=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        source = response.read(1024*1024)
    compile(source, url, 'exec')
    report['checks']['pinned_fetch'] = {'bytes': len(source),
        'sha256': hashlib.sha256(source).hexdigest(), 'executed': False}
    report['elapsed_seconds'] = time.monotonic()-started
    report['status'] = 'passed'
    Path('/kaggle/working/startup-diagnostic.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
