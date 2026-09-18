# Nova integration — CtrlSpeak 0.7.6

The gateway, Ubuntu CUDA worker and external Nova STT helper are coordinated on
the immutable `v0.7.6` release, commit
`8acc98d2fdab1ea1c9a7746f439db088794c1ac3`. Hermes itself is not restarted or patched.

Keep the existing command-provider path, protected profile environment,
GPU-preferred routing, request-scoped OpenAI key and `cleanup=true`. Keep the
existing schema-2 selected-text/feedback-sidecar contract. The new desktop
checkbox **Preserve paragraphs and line breaks** is intentionally irrelevant
to Nova: Telegram text is not keyboard injection, so paragraphs stay intact.

The gateway now retains validated S1 output before dictionary corrections as
`s1_cleaned_text` and its returned cleaned variant as `normalized_text`, alongside
raw/corrected text. The former is diagnostic only. Continue choosing validated
`normalized_text` only when applied/provenance metadata permits it and no exact
override wins; otherwise use `text`. Do not run another cleanup call or replace
the selected text with the new diagnostic field.

Historical records do not gain a reconstructed cleaned transcript. New audits
contain sensitive transcript contents: inspect by the exact transcription ID
when needed, do not bulk-export them into ordinary diagnostics. Lossless audio
transport, provider fallback and correction behaviour are unchanged.

Synthetic GPU/OpenAI/Tiny and helper/sidecar checks passed; gateway audit values
match the actual responses. All 116 correction rules were retained. Hermes
remained on PID 2908907 throughout backend activation.

Optional message for Nova:

> CtrlSpeak is now on 0.7.6. Your external STT helper has been updated without
> restarting Hermes. Keep your existing GPU-preferred cleanup and selected-text
> contract; Telegram paragraph formatting is unchanged. The release adds safer
> desktop keyboard insertion and gateway storage of cleaned output for auditing.
> No Hermes configuration or code change is required.
