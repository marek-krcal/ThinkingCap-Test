"""Bounded subprocess lifecycle for a dedicated rollout GPU."""
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading

from utils.train.rollout_worker import PREFIX


class RolloutClient:
    def __init__(self, gpu, deadline, log_path):
        self.process = None
        self.responses = queue.Queue()
        self.deadline = deadline
        self.log_path = Path(log_path)
        self.process = subprocess.Popen([sys.executable, '-u', '-m', 'utils.train.rollout_worker'],
              env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), VLLM_WORKER_MULTIPROC_METHOD='spawn'),
              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
              text=True, bufsize=1, start_new_session=True)
        self.reader = threading.Thread(target=self.read, daemon=True)
        self.reader.start()

    def read(self):
        with self.log_path.open('w') as log:
            for line in self.process.stdout:
                if line.startswith(PREFIX):
                    self.responses.put(json.loads(line[len(PREFIX):]))
                else:
                    log.write(line)
                    log.flush()
                    print('[rollout] '+line.rstrip(), flush=True)
        self.responses.put({'error': 'Rollout worker exited'})

    def receive(self, reserve=30):
        import time
        wait = self.deadline-time.monotonic()-reserve
        if wait <= 0:
            raise TimeoutError('No training time remains')
        try:
            response = self.responses.get(timeout=wait)
        except queue.Empty:
            raise TimeoutError('Rollout exceeded remaining training budget')
        if response.get('error'):
            raise RuntimeError(response['error'])
        return response

    def generate(self, request):
        self.process.stdin.write(json.dumps(request)+'\n')
        self.process.stdin.flush()
        return self.receive()

    def close(self):
        if self.process is None:
            return
        try:
            if self.process.poll() is None:
                os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=5)
        finally:
            # Parent exit can leave engine subprocesses in the process group.
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.reader.join(timeout=2)
