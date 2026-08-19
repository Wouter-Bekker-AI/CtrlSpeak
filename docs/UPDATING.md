# CtrlSpeak v0.7.3 update and release guide

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

v0.5.2 fixes a post-update TLS handoff defect found during that first live
exercise. v0.5.1 inherited `SSL_CERT_FILE` and `REQUESTS_CA_BUNDLE` paths from
the previous one-file extraction while it still existed; after cleanup, later
checks could not load that old CA file. v0.5.2 recognizes and replaces inherited
PyInstaller-temporary CA paths and sanitizes them from future helper launches.
An affected v0.5.1 process needs one ordinary quit/reopen before it can discover
v0.5.2; after v0.5.2, this extra restart is not expected.

v0.5.3 is a quality-of-life patch. It adds an in-memory-only **Copy last
transcript** tray action that can recover a successful result even when text
insertion fails, and it lowers the nearly full-scale processing-chime attack to
a transparent -6 dBFS peak ceiling. The API contract is unchanged; the
version-matched maintained service reports v0.5.3.

v0.6.0 adds the capability-aware gateway role, the worker-only Ubuntu GPU
role, identity-scoped corrections, request-scoped OpenAI BYOK, and the explicit
GPU → OpenAI → tiny provider cascade. Server roles are deployed administratively;
the signed desktop binary continues to update through the existing GUI. The
production gateway currently runs on the dedicated OpenStack `CtrlSpeak`
instance; Nova provides only the private WireGuard transport.

v0.6.1 adds **Submit correction…** directly to the tray. The authenticated form
creates an immediately active user-scoped gateway rule, with an explicit
administrator-only option for a global rule. Submission runs off the Tk thread,
validates empty or unchanged mappings, and never logs the bearer token or rule
text.

v0.6.2 adds explicit Ubuntu-GPU-preferred and OpenAI-preferred cascades,
provider-only routes, and bounded fast failover through a cached Ubuntu worker
health check and circuit breaker. Cascading routes may continue after OpenAI
key/quota failures while single-provider routes preserve the exact error. The
Windows client can remember the user's OpenAI key in Windows Credential Manager
without adding it to settings, releases, logs, or the gateway.

v0.7.0 is the Midnight Signal interface release. It replaces the utilitarian
tray/control-center presentation with a coherent dark interface, a compact live
recording capsule, an elongated processing animation, clearer navigation and
quick actions, accessible state labels, saved audio-cue controls, and detailed
provider health/routing information. Provider and latency information is based
only on authenticated measured telemetry; missing measurements remain visibly
unavailable, and a fallback path appears only after the gateway reports it.
The API remains backward-compatible while adding optional probe, attempt,
inference, and total-routing millisecond fields.

v0.7.1 is the first Midnight Signal hotfix. It prevents the microphone recorder
and asynchronous UI cues from entering competing PortAudio initialization or
termination lifecycles on Windows. That race caused an unrecoverable native
`_portaudio` access violation immediately after right Ctrl was pressed; it did
not produce a Python traceback. The patch also corrects control-center, tray,
and overlay fidelity against the approved Midnight Signal visual direction.
The API contract remains backward-compatible. See
`docs/V0.7.1_HOTFIX_RELEASE.md` for the incident evidence and acceptance gates.

v0.7.2 is the Midnight Signal dismissal hotfix. It gives the quick panel a
visible **Hide panel** action and Escape dismissal, adds **Hide to tray** to the
full control center, and labels the native tray toggle **Show / hide quick
panel**. Hiding a surface does not
quit CtrlSpeak or interrupt transcription. The service and API schema remain
backward-compatible; their versions are synchronized with the desktop release.
See `docs/V0.7.2_HOTFIX_RELEASE.md` for its acceptance gates.

v0.7.3 is the Midnight Signal desktop-fidelity and clipboard hotfix. It replaces
the legacy correction popup with a responsive Midnight Signal form whose fixed
footer remains reachable on high-DPI and short work areas, restores the proper
CtrlSpeak microphone artwork to recording and processing overlays, and repairs
**Copy last transcript** ownership and pointer-sized Win32 clipboard calls on
64-bit Windows. The service and API schema remain backward-compatible; their
versions are synchronized with the desktop release. See
`docs/V0.7.3_HOTFIX_RELEASE.md` for its acceptance gates.

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

`APP_VERSION` in `utils/version.py` is the desktop release version source. The
server package version in `server/whisper_transcription/pyproject.toml`, runtime
`SERVICE_VERSION`, Windows file metadata in
`packaging/windows_version_info.txt`, the AppStream release, Git tag, manifest
version, and release title must all agree.

For each release:

1. Update the version and user-facing documentation/release notes.
2. Run:

   ```text
   python -m pytest -m core_headless
   python -m compileall main.py utils scripts tests
   git diff --check
   cd server/whisper_transcription
   python -m pytest -q tests
   cd ../..
   ```

3. Perform the physical Windows/Ubuntu checks in `docs/TESTING.md` appropriate
   to the change.
4. Back up and deploy service source only to roles affected by the release,
   using the administrative acceptance procedure below. For desktop-only
   v0.7.3, synchronize the gateway's release identity and verify the unchanged
   worker protocol. The desktop updater does not deploy servers.
5. Commit and push the reviewed `v0.7` branch.
6. Create the immutable annotated tag `v0.7.3` at that commit and push it.
7. Observe `.github/workflows/release.yml` through all three stages:

   - clean Windows/Linux tests and native one-file builds;
   - packaged health probes plus manifest generation/signing; and
   - draft upload, five-asset download/audit, then stable publication.

8. Independently download the five release assets and run:

   ```text
   python scripts/release.py verify --directory <asset-directory> --tag v0.7.3
   ```

9. Test the update from the immediately previous stable version on both
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

The standard helper selects `packaging/CtrlSpeak_v0.7.spec`. Historical specs
remain unchanged for reproducibility and are not release inputs.

## Gateway and worker deployment

Publishing a desktop release does not update systemd services. Before the
stable tag is published:

1. Confirm the exact dedicated gateway and Ubuntu worker targets and their
   service roles. Do not deploy the gateway role onto Nova; Nova remains the
   private WireGuard transport/Hermes host.
2. Create a timestamped rollback copy and consistent SQLite backup for every
   service whose source will be replaced. Preserve protected environment files,
   client/worker tokens, model caches, virtual environments, and runtime data
   outside Git.
3. Stage the `server/whisper_transcription` subtree for affected roles, install
   the role-specific dependency profile only if it changed, run the server
   tests, and verify the redacted runtime configuration before restart.
4. For v0.7.3, deploy the mechanically version-synchronized gateway and verify
   authenticated health/capabilities report `0.7.3`, with unchanged correction
   inventory, telemetry flags, private listener, and a real restricted-language
   route. The Ubuntu GPU worker requires no configuration change for protocol
   compatibility. If synchronizing its release identity and worker-side timing
   telemetry, deploy the same maintained v0.7.3 service source while preserving
   its CUDA model cache, virtual environment, token, data, unit, and drop-ins.
   Production was synchronized this way on 2026-08-19.
5. Verify GPU-worker-offline fast failover without weakening the 350 ms connect,
   500 ms probe, cache, or circuit-break limits. Roll back source and database
   together if schema or startup acceptance fails.

The Hermes `telegrampersonal` command adapter requires no update for v0.7 when
the existing `/v1/transcribe` contract and authentication remain compatible.
Do not overwrite its host-specific adapter with a generic repository copy: the
deployed Nova adapter forwards its request-scoped provider and OpenAI credential
without persisting either at the gateway.

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
