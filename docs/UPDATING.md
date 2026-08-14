# CtrlSpeak v0.5.1 update and release guide

## End-user update flow

The installed standard application uses one permanent filename:

- Windows: `CtrlSpeak.exe`
- Linux: `CtrlSpeak`

The running version is visible in the tray menu and at the top of the control
center. Select **Check for updates** from either location. A manual check always
reports a result: up to date, a verified newer release, a newer-than-stable
development build, manual installation required, or an actionable failure.

When an update is available:

1. Review the current/target versions, artifact, download size, and restart
   notice.
2. Choose **Download and install**. Cancel is the default confirmation choice.
3. Keep using CtrlSpeak while the background download and verification finish.
4. Choose **Restart and install** when the card reports that the candidate is
   verified. Wait for recording/transcription to finish first.
5. CtrlSpeak closes, the helper replaces the executable, and the new version
   reopens. A one-time What's New summary may appear.

If the new version cannot launch or produce the correct health receipt, the
helper restores and relaunches the previous executable. Settings, credentials,
models, CUDA assets, logs, and corrections are stored outside the executable
and remain intact.

v0.4 and earlier cannot update themselves. Install v0.5.0 manually one final
time under the stable filename. Later signed versions can use the GUI.

v0.5.1 is the first normal in-application patch update. It adds the selectable
output-language allowlist and the version-matched maintained Ubuntu Whisper API.
An existing v0.5.0 Desktop installation should discover v0.5.1 through **Check
for updates**; it must not be replaced manually when validating that flow.

## Trust and safety model

The standard client hard-codes:

- GitHub owner/repository: `Wouter-Bekker-AI/CtrlSpeak`
- product: `ctrlspeak`
- channel: `stable`
- variant: `standard`
- supported targets: Windows x86-64 and Linux x86-64
- the Ed25519 release-verification public key

An installable release has exactly these five public assets:

- `CtrlSpeak-windows-x86_64.exe`
- `CtrlSpeak-linux-x86_64`
- `update-manifest.json`
- `update-manifest.sig`
- `SHA256SUMS`

The client refuses tags without a complete signed Release, drafts,
prereleases, malformed versions, wrong product/channel/variant, ambiguous
targets, non-canonical manifests, invalid signatures, non-HTTPS/untrusted URLs,
excessive files, and size or SHA-256 mismatches. A source checkout can discover
the latest release but cannot install it.

Update state lives below `%APPDATA%\CtrlSpeak\updates` on Windows or the normal
XDG CtrlSpeak directory on Linux. Partial downloads are bound to one signed
manifest/asset. Correct HTTP Range responses may resume them; stale or tainted
partials cannot become executable candidates.

## Maintainer release workflow

`APP_VERSION` in `utils/version.py` is the release version source. Windows file
metadata in `packaging/windows_version_info.txt`, the AppStream release, Git tag,
manifest version, and release title must agree.

For each release:

1. Update the version and user-facing documentation/release notes.
2. Run:

   ```text
   python -m pytest -m core_headless
   python -m compileall main.py utils scripts tests
   git diff --check
   ```

3. Perform the physical Windows/Ubuntu checks in `docs/TESTING.md` appropriate
   to the change.
4. Commit and push the reviewed `v0.5` branch.
5. Create an immutable annotated tag such as `v0.5.1` at that commit and push it.
6. Observe `.github/workflows/release.yml` through all three stages:

   - clean Windows/Linux tests and native one-file builds;
   - packaged health probes plus manifest generation/signing; and
   - draft upload, five-asset download/audit, then stable publication.

7. Independently download the five release assets and run:

   ```text
   python scripts/release.py verify --directory <asset-directory> --tag v0.5.1
   ```

8. Test the update from the immediately previous stable version on both
   platforms. Do not move a published tag or replace consumed assets; publish a
   new patch version instead.

The private signing key is stored as the GitHub Actions repository secret
`CTRLSPEAK_UPDATE_SIGNING_KEY`. It is a base64-encoded raw Ed25519 private key.
Never print it, commit it, place it in logs, or copy it into an application
binary. The application contains only the public key.

## Build commands

Build natively on each target OS:

```text
python -m utils.build_exe
```

Expected local output is `dist/CtrlSpeak.exe` on Windows and
`dist/CtrlSpeak` on Linux. GitHub Actions renames copies to the public
platform-qualified release names before creating the manifest. Model weights
and CUDA runtimes remain external.

For a package smoke test, use a writable temporary output path:

```text
CtrlSpeak.exe --health-check-file <temporary-path>\health.json
CtrlSpeak --health-check-file <temporary-path>/health.json
```

The JSON must report product `ctrlspeak`, the exact tagged version, the running
executable path, and `frozen: true`.

## Failure recovery and diagnostics

The control center's **Copy diagnostics** action includes only version, runtime
classification, state, finite error category, release tag, manifest SHA-256,
and transaction ID. It excludes tokens, cookies, HTTP headers, transcript text,
clipboard contents, environment dumps, and signing secrets.

Transaction details and the helper log are retained in the relevant update
directory. Common categories include network/TLS/rate-limit failure, invalid
manifest/signature, missing or ambiguous target, unsafe redirect, range error,
incomplete/oversize download, hash/size mismatch, unwritable/manual install,
locked executable, helper launch failure, health timeout, and rollback failure.

Do not manually delete a transaction during an active install. If rollback
restored the previous executable, retain that transaction for diagnosis. If the
helper restored the file but could not relaunch it, start the stable executable
from its existing location and inspect the update log.

## Signing-key rotation

Key rotation requires at least two releases:

1. Ship a healthy client that trusts both the old and new public keys while the
   release is still signed by the old private key.
2. Confirm adoption and updater compatibility.
3. Start signing with the new key in a later release.
4. Remove the old public key only in a subsequent release after the supported
   upgrade population can verify the transition.

Never switch the GitHub secret and embedded public key in the same release
without a trust bridge; existing clients would reject it.
