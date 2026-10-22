# Nova integration — CtrlSpeak 0.7.7

Keep the existing external STT command provider and dedicated gateway URL,
request-scoped OpenAI credential and GPU-preferred cleanup policy. Hermes code,
conversation state and the Telegram gateway do not need a restart for this update.

The API keeps `text`, optional validated `normalized_text`, correction IDs and
the existing schema-2 helper selection contract. Additional request/timing and
worker admission data is diagnostic; do not substitute it for transcript text.
S1 is still available only on eligible Ubuntu GPU requests. Desktop insertion,
formatting, Cancel and Retry changes do not change Telegram paragraph handling.

A busy worker can now be bypassed promptly under the chosen fallback strategy.
Do not manually poll/retry the same voice note in a loop, and do not add a second
cleanup pass. Report an actionable bounded failure if all permitted providers
fail; an explicit retry can incur another provider charge.

Deployment verified on 2026-10-22: gateway, GPU worker and this external helper
all use release revision `47bcc677bf9f194a2fd15d592d3e6d95d557cfe5`. GPU/S1,
OpenAI, gateway Tiny and helper schema-2 live canaries passed. The existing
Hermes Telegram process was not restarted. See the [release verification record](V0.7.7_RELEASE.md).

Suggested notice (not automatically sent to Nova):

> CtrlSpeak 0.7.7 improves busy-worker fallback and adds request/stage diagnostics.
> Your external helper was reconciled with the release without restarting Hermes.
> Keep the existing gateway, GPU-preferred routing, caller-owned API key and
> validated cleaned-text selection. Telegram formatting is unchanged.
