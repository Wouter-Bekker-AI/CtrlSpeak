# Nova integration: CtrlSpeak 0.7.5

This release changes the external STT transport, not Hermes itself. Keep the
existing `local-whisper-api` command-provider configuration and its absolute
`hermes-stt-api {input_path} {language} {output_path}` command. No Hermes restart,
core-code patch, profile migration or active conversational turn reset is needed.

The configured command path selects an immutable CtrlSpeak Git release. Each
invocation reads the existing protected profile environment and uses a dedicated
helper virtual environment outside Hermes. It still requests
`strategy=ubuntu-gpu-preferred`, `cleanup=true`, with the requested language and
the existing request-scoped OpenAI key file. Never copy that key into the
gateway's common environment or print it in diagnostics.

Telegram's existing Ogg/Opus file is uploaded unchanged. If a caller supplies
PCM16 WAV, the helper compresses it losslessly to FLAC where smaller. The gateway
keeps it compressed when forwarding to the local GPU or OpenAI. The inference
backend decodes it. An already-lossy Telegram recording cannot become lossless
retroactively; this update introduces no additional loss.

The output contract remains one selected transcript in `{output_path}` plus
`{output_path}.ctrlspeak.json` schema 2. Only validated English Ubuntu CUDA S1
output is selected as cleaned text; otherwise use gateway-corrected `text`.
Preserve the feedback transcription ID and exact-override behavior. OpenAI and
gateway Tiny do not return S1-cleaned text. Do not echo both variants as separate
user messages or run a second cleanup call.

Release checks include real, non-personal FLAC and Opus canaries through the GPU
and OpenAI, a Tiny canary and helper/sidecar validation. The helper has a bounded
290-second overall deadline, cleans temporary compression files and emits only
safe byte-count/timing metadata. Failure is reported as failure, not empty success.

Message for Nova after the deployment is verified:

> CtrlSpeak is now on 0.7.5: your existing external STT helper handles lossless
> WAV-to-FLAC compression and unchanged Telegram Ogg/Opus uploads automatically;
> keep GPU-preferred routing and the current selected-text/feedback-sidecar
> contract, with S1 cleanup only on successful Ubuntu GPU requests. No Hermes
> code or configuration change is needed.
