# Nova / Hermes integration handoff — CtrlSpeak 0.7.4

The dedicated gateway stays at `http://10.83.233.4:8765`. Authentication,
identity-scoped known words, correction submission, feedback and transcription
IDs are unchanged. The Ubuntu worker and gateway now run the same reviewed
Git revision. S1-mini runs only on the Ubuntu RTX 3060, not on gateway CPU.
The old gateway S1 service and health timer are deliberately disabled; do not
re-enable them as a repair. The replacement is Ubuntu's user service
`ctrlspeak-s1-mini`, publishing only loopback port 8081 for the worker.

## Required adapter change

The inspected Nova `telegrampersonal` profile still forces
`CTRLSPEAK_PROVIDER=openai-gpt-transcribe`. Its existing adapter does not send
`cleanup`; consequently it continues to work, but will not receive S1 cleanup.
No Nova adapter/profile was modified by this deployment.

Update the host-specific STT command adapter, preserving its credentials and
sidecar contract. Do not replace it with the generic repository script.

1. Send multipart `strategy=ubuntu-gpu-preferred` and `cleanup=true` to
   `/v1/transcribe`. Remove the explicit `provider` form field for this cascade.
2. Supply Nova's own saved OpenAI key in `X-CtrlSpeak-OpenAI-Key` on cascading
   requests too, not only when an explicit OpenAI provider is selected. Keep the
   normal gateway bearer authentication. Never log either header or install the
   OpenAI key at the gateway/worker.
3. Select `normalized_text` only if non-empty and the response reports
   `provider_used=ubuntu-gpu-large-v3-turbo`, `language=en`,
   `normalization.requested=true`, `applied=true`, `status=applied`,
   `device=cuda`, `location=ubuntu-worker`, and model `S1-mini by Superwhisper`.
   Otherwise select ordinary corrected `text`. Keep `raw_text` and `id` for
   correction/feedback use. A null cleaned result is normal, not a failed STT.
4. Keep the existing sidecar fields. Mark `normalization_applied` only when the
   validated cleaned result actually differs from the ordinary text. Emit a
   second, clearly labelled cleanup echo only then; otherwise emit one echo.
   Supply the selected transcript to the assistant exactly once.
5. Verify synthetic voice input with cleanup on/off and Ubuntu unavailable,
   including key forwarding on the cascade. Check that an S1 timeout does not
   discard successful ASR or trigger a second transcription. Report evidence,
   not just configuration intent.

GPU preferred means Ubuntu Whisper + optional S1, then OpenAI, then gateway
Tiny. OpenAI/Tiny never call S1 and return `normalized_text: null` with
`normalization.applied: false`. Ordinary gateway correction rules still apply.
Even the GPU path can skip cleanup for busy, unavailable, timed-out, non-English,
overlong, unchanged, or unsafe output, or a confirmed exact transcript override.
The API remains opt-in and backward-compatible when `cleanup` is omitted.

## Copyable instruction

Please update your host-specific CtrlSpeak STT adapter for 0.7.4: use
`strategy=ubuntu-gpu-preferred` and `cleanup=true`, remove the forced OpenAI
`provider`, and forward your own saved OpenAI key on cascade requests so fallback
works. Use `normalized_text` only when the response confirms applied S1-mini
cleanup on the Ubuntu CUDA worker; otherwise use ordinary corrected `text`.
OpenAI/Tiny never receive S1 cleanup. Preserve raw text/IDs and feedback, pass the
selected text to the agent once, and only show a second cleaned echo when cleanup
actually changed the text. Do not re-enable the retired gateway CPU S1 service
or its timer. Test both GPU and fallback paths and report results.
