#!/usr/bin/env python3
"""Restricted, signed-release deployment endpoint. Install root-owned outside Git.

The SSH key is forced to this program with one root-owned host configuration.
It cannot execute arbitrary SSH commands. Input is a bounded signed manifest,
not a shell script. No Hermes service operation is implemented here.
"""
from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen

PUBLIC_KEY = "2pgW2LL2C+2DdM0L/fwgmjYsEiiFG2f6eIL2bbKLEvI="
REPOSITORY = "https://github.com/Wouter-Bekker-AI/CtrlSpeak.git"


def verify_request(request, public_key=PUBLIC_KEY):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    raw = base64.b64decode(request['manifest'], validate=True)
    sig = base64.b64decode(request['signature'], validate=True)
    Ed25519PublicKey.from_public_bytes(base64.b64decode(public_key)).verify(sig, raw)
    manifest = json.loads(raw)
    assert manifest['product'] == 'ctrlspeak' and manifest['channel'] == 'stable'
    version = manifest['version']
    assert re.fullmatch(r'\d+\.\d+\.\d+', version) and manifest['tag'] == 'v' + version
    deploy = manifest['deployment']
    assert deploy['schema'] == 1 and deploy['transport_protocol'] == 1
    assert re.fullmatch('[0-9a-f]{40}', deploy['revision'])
    assert deploy['roles'] == {role: version for role in ('worker', 'gateway', 'helper')}
    assert request['action'] in {'stage', 'activate', 'verify', 'rollback', 'desktop', 'canary'}
    return manifest


def run(command, timeout=30, *, cwd=None, env=None):
    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        if process.returncode:
            # Subprocess output may contain protected configuration. Do not echo it.
            raise RuntimeError(f'{Path(command[0]).name} exited {process.returncode}')
        return stdout.strip()
    finally:
        # Reap the entire process group, including orphaned grandchildren.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)


def atomic(path, content, mode=0o600):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as target:
            target.write(content)
            target.flush()
            os.fsync(target.fileno())
        os.chmod(name, mode)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


class Host:
    def __init__(self, config, manifest):
        self.config, self.manifest = config, manifest
        self.role = config['role']
        self.version = manifest['version']
        self.revision = manifest['deployment']['revision']
        self.checkout = Path(config['releases']) / self.revision
        self.source = self.checkout / 'server/whisper_transcription'
        self.state = Path(config['state'])
        self.transaction = self.state / self.revision
        self.receipt = self.transaction / 'receipt.json'
        self.manager = ['systemctl'] + ([] if self.role == 'gateway' else ['--user'])

    def record(self, status):
        atomic(self.receipt, json.dumps({'version': self.version, 'revision': self.revision,
            'role': self.role, 'status': status, 'protocol': 1, 'time': time.time()}).encode())

    def health(self):
        service = self.config['service']
        pid = run(self.manager + ['show', service, '-p', 'MainPID', '--value'])
        env = dict(item.split(b'=', 1) for item in Path('/proc/' + pid + '/environ').read_bytes().split(b'\0') if b'=' in item)
        token = env.get(b'CTRLSPEAK_WORKER_TOKEN' if self.role == 'worker' else b'WHISPER_BEARER_TOKEN', b'').decode()
        if not token:
            clients = json.loads(env.get(b'CTRLSPEAK_CLIENTS_JSON', b'{}'))
            client = next(iter(clients.values()))
            token = client if isinstance(client, str) else client['token']
        request = Request(self.config['health_url'], headers={'Authorization': 'Bearer ' + token})
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    def stage(self):
        import shutil
        assert shutil.disk_usage(self.config['releases']).free > 1024 ** 3, 'insufficient staging disk'
        self.transaction.mkdir(parents=True, exist_ok=True)
        if not self.checkout.exists():
            run(['git', 'clone', '--filter=blob:none', '--no-checkout', '--depth', '1',
                 '--single-branch', '--branch', self.manifest['tag'], REPOSITORY, str(self.checkout)], 180)
            run(['git', '-C', str(self.checkout), 'sparse-checkout', 'set',
                 'server/whisper_transcription', 'scripts', 'tests/fixtures'], 60)
            run(['git', '-C', str(self.checkout), 'checkout', '--detach', self.revision], 90)
        assert run(['git', '-C', str(self.checkout), 'rev-parse', 'HEAD']) == self.revision
        assert not run(['git', '-C', str(self.checkout), 'status', '--porcelain', '--untracked-files=no'])
        if self.role == 'gateway':
            # Source is public; the unprivileged ctrlspeak service must traverse
            # the root-owned checkout despite the deployer's private umask.
            for directory, children, files in os.walk(self.checkout):
                children[:] = [name for name in children if name != '.git']
                Path(directory).chmod(0o755)
                for name in files:
                    path = Path(directory) / name
                    if not path.is_symlink():
                        path.chmod(0o755 if path.stat().st_mode & 0o111 else 0o644)
        # Existing running source must be clean too; never discard later local fixes.
        if self.role != 'helper':
            current = run(self.manager + ['show', self.config['service'], '-p', 'WorkingDirectory', '--value'])
            if (Path(current).parents[1] / '.git').exists():
                assert not run(['git', '-c', 'safe.directory=*', '-C', current, 'status', '--porcelain', '--untracked-files=no'])
        if self.role == 'helper':
            venv = self.source / '.helper-venv'
            if not (venv / 'bin/python').exists():
                run(['python3', '-m', 'venv', str(venv)], 60)
            run([str(venv / 'bin/python'), '-m', 'pip', 'install', '--disable-pip-version-check',
                 '-r', str(self.source / 'requirements-helper.txt')], 180)
            python = str(venv / 'bin/python')
        else:
            python = self.config['python']
        result = json.loads(run([python, '-c', 'import json; from app.audio_transport import self_test; print(json.dumps(self_test()))'], 30, cwd=self.source))
        assert result['bit_perfect'] and result['version'] == self.version
        for name in ('run-service', 'runtime-config', 'hermes-stt-api'):
            (self.source / 'scripts' / name).chmod(0o755)
        self.record('staged') if not self.receipt.exists() else None
        return {'role': self.role, 'staged': self.revision, 'codec': result}

    def activate(self):
        assert self.receipt.exists(), 'stage before activation'
        prior = json.loads(self.receipt.read_text())
        if prior['status'] in ('active', 'verified'):
            return self.verify()
        target = Path(self.config['selector'])
        backup = self.transaction / 'selector-before.json'
        if not backup.exists():
            atomic(backup, json.dumps({'exists': target.exists(), 'content': base64.b64encode(target.read_bytes()).decode() if target.exists() else None}).encode())
        if self.role == 'gateway':
            snapshot = self.transaction / 'corrections.sqlite3'
            if not snapshot.exists():
                with sqlite3.connect('file:' + self.config['database'] + '?mode=ro', uri=True) as original:
                    with sqlite3.connect(snapshot) as destination:
                        original.backup(destination)
                snapshot.chmod(0o600)
        # Write intent before the switch, so interrupted activation is rollback-able.
        self.record('activating')
        if self.role == 'helper':
            # New invocations resolve an immutable source; existing Python/bash
            # processes keep their old runtime. Neither Hermes nor its env is changed.
            launcher = '#!/bin/sh\nset -eu\nexec ' + str(self.source / 'scripts/hermes-stt-api') + ' "$@"\n'
            atomic(target, launcher.encode(), 0o755)
        else:
            config = ('[Service]\nWorkingDirectory=' + str(self.source) + '\nExecStart=\nExecStart=' +
                str(self.source / 'scripts/run-service') + '\nEnvironment=WHISPER_VENV_DIR=' +
                str(Path(self.config['python']).parents[1]) + '\nEnvironment=CTRLSPEAK_RELEASE_REVISION=' +
                self.revision + '\nTimeoutStopSec=330\n')
            atomic(target, config.encode(), 0o644)
            # Reload first, allowing current requests to drain under the new stop
            # budget. Only CtrlSpeak's inference/API unit is restarted.
            run(self.manager + ['daemon-reload'])
            run(self.manager + ['restart', self.config['service']], 360)
        self.record('active')
        return self.verify()

    def verify(self):
        if self.role == 'helper':
            observed = run([self.config['selector'], '--version'])
            assert observed == self.version
            result = json.loads(run([self.config['selector'], '--self-test'], 30))
            assert result['bit_perfect']
        else:
            deadline = time.monotonic() + 90
            while True:
                try:
                    result = self.health()
                    assert result['version'] == self.version and result['revision'] == self.revision
                    assert result['audio_transport']['protocol'] == 1 and result['status'] == 'ready'
                    break
                except Exception:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(2)
        self.record('verified')
        return {'role': self.role, 'version': self.version, 'revision': self.revision, 'verified': True}

    def rollback(self):
        backup = self.transaction / 'selector-before.json'
        if not backup.exists():
            return {'role': self.role, 'rollback': 'no activation recorded'}
        previous = json.loads(backup.read_text())
        target = Path(self.config['selector'])
        if previous['exists']:
            atomic(target, base64.b64decode(previous['content']), 0o755 if self.role == 'helper' else 0o644)
        else:
            target.unlink(missing_ok=True)
        if self.role != 'helper':
            run(self.manager + ['daemon-reload'])
            run(self.manager + ['restart', self.config['service']], 360)
            run(self.manager + ['is-active', '--quiet', self.config['service']])
        # Never restore the old corrections DB: that would lose new user writes.
        self.record('rolled_back')
        return {'role': self.role, 'rollback': 'restored previous selector'}

    def desktop(self):
        assert self.role == 'worker'
        asset = next(a for a in self.manifest['assets'] if a['platform'] == 'linux')
        expected = 'https://github.com/Wouter-Bekker-AI/CtrlSpeak/releases/download/' + self.manifest['tag'] + '/CtrlSpeak-linux-x86_64'
        assert asset['url'] == expected
        target = Path(self.config['desktop'])
        candidate = target.with_name('CtrlSpeak.candidate-' + self.version)
        digest = hashlib.sha256()
        with urlopen(expected, timeout=30) as response, candidate.open('wb') as out:
            deadline, size = time.monotonic() + 600, 0
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                assert size <= asset['size'] and time.monotonic() < deadline
                digest.update(chunk); out.write(chunk)
        assert size == asset['size'] and digest.hexdigest() == asset['sha256']
        candidate.chmod(0o755)
        with tempfile.TemporaryDirectory(prefix='ctrlspeak-package-') as temp:
            metadata = Path(temp) / 'health.json'
            run([str(candidate), '--health-check-file', str(metadata)], 90)
            result = json.loads(metadata.read_text())
            assert result['version'] == self.version and result['audio_transport']['bit_perfect']
        backup = target.with_name('CtrlSpeak.pre-' + self.version)
        if target.exists() and not backup.exists():
            import shutil
            shutil.copy2(target, backup)
        os.replace(candidate, target)
        return {'role': 'ubuntu-desktop', 'version': self.version, 'verified': True}

    def canary(self):
        assert self.role == 'helper'
        # The fixed profile source is read by bash; no env values are printed.
        command = ('set -a; source /home/ubuntu/.hermes/profiles/telegrampersonal/.env; '
                   'set +a; export PYTHONPATH=' + str(self.source) + '; exec ' +
                   str(self.source / '.helper-venv/bin/python') + ' ' +
                   str(self.checkout / 'scripts/stt_canary.py'))
        result = json.loads(run(['/bin/bash', '-c', command], 660))
        return {'role': 'helper', 'canaries': result}


def main():
    import fcntl
    def interrupted(_signum, _frame):
        raise InterruptedError('deployment interrupted; reconcile receipt')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    os.umask(0o077)
    config = json.loads(Path(sys.argv[1]).read_text())
    state = Path(config['state']); state.mkdir(parents=True, exist_ok=True)
    # One transaction per target. A second CI job fails fast rather than waiting.
    with (state / 'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = sys.stdin.buffer.read(32769)
        assert len(raw) <= 32768
        request = json.loads(raw)
        manifest = verify_request(request)
        host = Host(config, manifest)
        result = getattr(host, request['action'])()
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps({'error': type(exc).__name__, 'detail': str(exc)[:160] if isinstance(exc, AssertionError) else 'deployment operation failed; protected host state retained'}), flush=True)
        raise SystemExit(1)
