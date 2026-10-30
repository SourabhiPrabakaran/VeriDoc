"""Run unchanged phase scripts in a disposable backend, never real case storage."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8000, help='Loopback port for disposable backend')
    args = parser.parse_args()
    logs = args.log_dir.resolve(); logs.mkdir(parents=True, exist_ok=True)
    backend = Path(__file__).resolve().parents[1]
    # Refuse to use an existing server. Only temporary copies have their base URL
    # redirected when a nondefault port is requested; assertions stay unchanged.
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', args.port))
    env = dict(os.environ, PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    results = []
    with tempfile.TemporaryDirectory(prefix='veridoc-legacy-') as folder:
        isolated = Path(folder)
        env.update(BACKEND_DIR=str(isolated), DEEPFAKE_MODEL_PATH='', DEEPFAKE_MODEL_VERSION='NOT_CONFIGURED')
        shutil.copytree(backend / 'app', isolated / 'app', ignore=shutil.ignore_patterns('__pycache__'))
        scripts = sorted(backend.glob('test_phase*.py'))
        for script in scripts:
            shutil.copy2(script, isolated / script.name)
            if args.port != 8000:
                copy = isolated / script.name
                copy.write_text(copy.read_text(encoding='utf-8').replace(
                    'http://127.0.0.1:8000/api', f'http://127.0.0.1:{args.port}/api'), encoding='utf-8')
        with (logs / 'server.log').open('w', encoding='utf-8') as server_log:
            server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', str(args.port)],
                                      cwd=isolated, env=env, stdout=server_log, stderr=subprocess.STDOUT)
            try:
                ready = False
                for _ in range(60):
                    if server.poll() is not None:
                        raise RuntimeError('Temporary backend failed; inspect server.log')
                    try:
                        urllib.request.urlopen(f'http://127.0.0.1:{args.port}/api/health', timeout=1).close()
                        ready = True; break
                    except OSError:
                        time.sleep(.5)
                if not ready:
                    raise RuntimeError('Temporary backend did not become ready')
                for script in scripts:
                    completed = subprocess.run([sys.executable, script.name], cwd=isolated, env=env,
                                               capture_output=True, text=True, timeout=120)
                    (logs / (script.name + '.log')).write_text(completed.stdout + completed.stderr, encoding='utf-8')
                    row = {'script': script.name, 'exit_code': completed.returncode}
                    results.append(row); print(json.dumps(row), flush=True)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill(); server.wait()
    (logs / 'summary.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    return int(any(item['exit_code'] for item in results))


if __name__ == '__main__':
    sys.exit(main())
