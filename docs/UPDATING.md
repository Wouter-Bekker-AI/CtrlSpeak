# Updating and releasing CtrlSpeak

## End-user update flow

Use the tray's **Check for updates** or the control center's **Updates** page.
The standard executable stays named `CtrlSpeak.exe` on Windows or `CtrlSpeak`
on Linux. Versions before 0.5 need one manual replacement; install a current
verified standard executable under the stable name.

1. Review the current and target versions and the expected download size.
   **Download and install** starts the transfer; you can cancel before handoff.
2. An exact supported base can use a signed differential patch. CtrlSpeak
   verifies the patch and reconstructs the complete new executable in a private
   transaction directory, not over the running file. If no patch applies, or
   reconstruction fails, it uses the verified full download. Direct patches
   are generated from up to three recent patch-aware signed releases (0.8.2+).
3. When the candidate is verified, finish any active dictation and choose
   **Restart and install**. The detached helper waits for exit, preserves the
   previous executable, replaces it atomically, and waits for the new app's
   startup health receipt. If startup fails, it restores the previous build.
4. Check the running version in the tray or control center. The Updates page
   reports the last installed transfer method, bytes and any fallback reason.

A small patch still needs disk space for the reconstructed executable, startup
extraction and rollback. Successful payload caches retain two recent installs
and the most recent explicit rollback to 0.7.7. Logs and audit metadata stay;
active, failed and unrecognized transactions are not automatically pruned.

**Updates → Roll back to 0.7.7** is a separate, explicit signed install for a
writable packaged copy. It retains settings and credentials but removes access
to newer dictation modes. See the [rollback details](V0.8.0_RELEASE.md#return-to-077).
Source checkouts can discover releases but must use Git to update themselves.

## Trust and safety model

The updater trusts the hard-coded GitHub repository `Wouter-Bekker-AI/CtrlSpeak`,
product `ctrlspeak`, channel `stable`, variant `standard`, and the embedded
Ed25519 public key. Supported release targets are Windows and Linux x86-64.

The exact public asset set is:

- `CtrlSpeak-windows-x86_64.exe` and `CtrlSpeak-linux-x86_64`
- `update-manifest.json`, `update-manifest.sig`, and `SHA256SUMS`
- only the `.vcdiff` files declared in the signed manifest, if any

A tag alone, draft, prerelease, wrong target, invalid signature, untrusted
download URL, or size/hash mismatch cannot be installed. Resume is bound to
one manifest and asset and requires a correct HTTP Range response. A patch
must match the complete base by version, size and hash; the reconstructed
candidate must match the complete signed target by size and hash.

Settings, logs, models, CUDA files, corrections and native credentials are
outside the executable and are not replaced. Transaction state is under
`%APPDATA%\CtrlSpeak\updates` or the XDG CtrlSpeak data directory. See the
[0.8.2 update evidence](V0.8.2_RELEASE.md) for format and migration details.

## Maintainer release workflow

1. Branch from `stable`. Align `APP_VERSION`, API runtime and package versions,
   transport/helper release identity, Windows resources, AppStream metadata,
   and release notes. Keep protocol versions unchanged unless the protocol
   really changes.
2. Run the [test gates](TESTING.md) and the native Windows/Linux checks relevant
   to the change. Validate the tag before pushing:
   `python scripts/release.py check-version --tag v<version>`.
3. Push reviewed source and create an immutable `v<version>` tag on the exact
   tested commit. The tag starts the
   [signed release workflow](../.github/workflows/release.yml); it is not a
   dry run. Preflight the configured fleet and ensure rollback is available.
4. CI builds natively, checks packaged health, generates and roundtrips
   patches, signs the manifest and uploads a draft. It downloads the draft
   again and verifies the exact public asset set before activation.
5. The workflow stages and verifies the **Ubuntu worker**, **dedicated
   gateway** and **external Nova STT helper** at the signed revision, with
   rollback on a failed mandatory activation/canary. It publishes only after
   those roles pass, then reconciles the Ubuntu desktop companion. A companion
   failure after publication requires separate reconciliation; check its
   actual result. Windows clients install through their GUI.
6. Verify the public assets independently and test an upgrade from the prior
   release against an isolated copy. Run
   `python scripts/release.py verify --directory <assets> --tag v<version> --exact-assets`.
   Preserve the fleet receipt and note any physical acceptance not performed.
   Advance `stable` to the verified source. Do not move published tags or
   replace released assets; issue a new patch version for corrections.

If build infrastructure fails before publication, rerun the failed jobs
first. When a fix to the workflow itself is required, tag the reviewed release
tooling as `v<version>-release-tools.<n>` and dispatch the release workflow
from that tooling tag with the existing product release tag as input.
Tooling tags do not start automatic releases or become update releases. This
preserves the production environment's tag-only policy. The workflow rebuilds
the exact product tag, checks both build provenance records before signing its
revision, and applies the same deployment/publication gates. It does not move
the product tag or replace published assets.

The private Ed25519 key stays in the GitHub Actions signing secret. Never put
it in source, logs or the binary. Key rotation needs a release that trusts both
keys before switching the signer; remove the old key only after adoption.

## Gateway, worker and Nova

The dedicated gateway owns routing, identities, corrections and audits. The
Ubuntu worker handles GPU transcription and optional S1; the helper on Nova
calls the gateway and is not a server. The release deploys through private
WireGuard using immutable Git checkouts and role-specific health checks. It
does not update or restart Hermes itself. Runtime state and credentials stay
outside the release checkouts. Do not copy host configuration into a public
repo. See the [service guide](../server/whisper_transcription/README.md) and
[API contract](API.md).

For recovery, inspect the deployment receipt and the selected revision on
each host before retrying anything. Do not start a second deployment over a
running workflow. Preserve the prior release, correction database, protected
environment, models and runtime dependencies. Roll back through the verified
selectors rather than overwriting the live checkout.

## Failure recovery and diagnostics

Use **Updates → Copy diagnostics** for version, runtime classification,
transaction state, redacted error details and release identity. It does not
include tokens, HTTP headers, transcript text, clipboard contents or signing
secrets. Transaction journals and helper logs live in the private updates
directory. Do not delete a transaction while installation is active. If the
helper restored the previous file but could not relaunch it, start the stable
filename in its existing location and inspect the retained logs.
