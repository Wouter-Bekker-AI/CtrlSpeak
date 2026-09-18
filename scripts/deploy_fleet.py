#!/usr/bin/env python3
"""GitHub-hosted coordinator. Stage all roles, activate/verify, or roll back."""
import argparse
import base64
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROLES = ('worker', 'gateway', 'helper')


def transaction(call):
    # No activations until every mandatory host can stage the signed release.
    for role in ROLES:
        call(role, 'stage')
    attempted = []
    try:
        for role in ROLES:
            attempted.append(role)  # Include ambiguous/partially completed calls.
            call(role, 'activate')
        for role in ROLES:
            call(role, 'verify')
        call('helper', 'canary')
        return {'status': 'verified', 'roles': list(ROLES)}
    except BaseException:
        failed = []
        for role in reversed(attempted):
            try:
                call(role, 'rollback')
            except BaseException:
                failed.append(role)
        if failed:
            print('ROLLBACK REQUIRES ATTENTION: ' + ', '.join(failed), file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', default='release')
    parser.add_argument('--desktop-only', action='store_true')
    args = parser.parse_args()
    directory = Path(args.directory)
    common = {'manifest': base64.b64encode((directory / 'update-manifest.json').read_bytes()).decode(),
              'signature': (directory / 'update-manifest.sig').read_text().strip()}
    # Hostnames/usernames are operational environment variables, never guessed.
    hosts = json.loads(os.environ['CTRLSPEAK_DEPLOY_HOSTS'])
    receipts = []

    def call(role, action):
        print(role + ': ' + action, flush=True)
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                   '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=10',
                   '-o', 'ServerAliveCountMax=3', '-i', os.environ['CTRLSPEAK_DEPLOY_KEY'],
                   hosts[role], 'deploy']
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, text=True, start_new_session=True)
        try:
            stdout, _ = process.communicate(json.dumps(dict(common, action=action)), timeout=720)
            if process.returncode:
                raise RuntimeError(role + ' ' + action + ' failed: ' + stdout[-300:])
            result = json.loads(stdout)
            receipts.append(result)
            return result
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)

    try:
        result = call('worker', 'desktop') if args.desktop_only else transaction(call)
        print(json.dumps(result), flush=True)
    finally:
        # Sanitized evidence remains available even on a failed/rolled-back run.
        (directory / 'deployment-receipt.json').write_text(json.dumps(receipts, indent=2))


if __name__ == '__main__':
    main()
