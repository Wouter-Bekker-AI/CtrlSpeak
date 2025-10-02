# Chat With Bot Integration

CtrlSpeak includes an optional "Chat with Bot" experience accessible from the management window. When enabled, it pairs the local transcription server with the vendored **SocialRobot** project to provide a voice conversation loop:

- **Speech to Text (STT)** - Uses CtrlSpeak's `/transcribe` endpoint. Audio captured by the VAD listener is sent to the running CtrlSpeak server (local or remote depending on mode). The server returns the recognized text.
- **Language Model (LLM)** - The recognized text is sent to the Ollama-compatible client inside SocialRobot. By default CtrlSpeak ships with a lightweight fallback response if no LLM endpoint is reachable, but you can supply your own by setting the `BOT_LLM_URL` and `BOT_LLM_MODEL` environment variables (or the matching CLI flags) before launching CtrlSpeak.
- **Text cleanup (background agent)** - SocialRobot now checks each reply and only calls `tools.message_management.force_plaintext` when markdown bullets or control characters are present. The scrubbed text populates the chat transcript, memory persistence, and Kokoro playback; otherwise the untouched reply flows straight through. The Profile Paragraphizer assets remain in `background_agents/tts_preprocessing_agent/`, but the helper is currently disabled in the runtime pipeline.
- **Text to Speech (TTS)** - The LLM response is converted to audio via Kokoro-ONNX. CtrlSpeak defaults to the formal male `am_michael` voice; override it with `BOT_VOICE` or the `--voice` flag.
- **Animated Face / Logo** - SocialRobot renders the default TrueAI transparent logo with amplitude-based scaling for visual feedback. Identity folders can still supply alternate assets under `third_party/social_robot/identities/<name>` when a different look is desired.
- **Text chat window** - Sessions now start in text mode. A PySide chat window shows the running transcript, accepts typed input, and exposes a microphone toggle. The window anchors itself to the bottom-right corner where the floating logo normally lives and applies `assets/icon.ico` to both the window chrome and the microphone button for consistent branding. When you click the microphone to enter voice mode the chat window slides to the top-right corner, hides the input, shows the floating logo again, and hands control back to the VAD/TTS pipeline. Clicking the button again (or ending the session) returns to text chat, restores the bottom-right placement, and leaves the conversation in on-screen text.
- **Documentation preload** - Before the assistant, Einstein, or default personas become interactive, CtrlSpeak runs `refresh_document_memory` for the selected identity to hash bundled Markdown docs, evict any stale `category="documentation"` entries from the vector store, and insert fresh chunks when the content changed or the 24-hour cooldown expired. Progress is reported only in the terminal while the chat window stays hidden.

## Identity Profiles

Personalities for the bot live under `third_party/social_robot/identities/<name>`. Each identity directory may contain:

- `identity.json` - configuration for the profile (voice, model, prompt settings, optional memory directory).
- `system_prompt.txt` (or another file referenced by the config) - the text injected as the LLM system prompt when the profile is loaded.
- Runtime memory lives under `${data_root}/bot_memory/<identity>/`, which CtrlSpeak creates automatically with `conversation/`, `screenshots/`, `chroma/`, and `traces/` subfolders. Packaged builds ignore any legacy `memory/` folders inside the repository tree so all writes land in AppData.
- Identity-specific defaults live under `${config_root}/identities/<identity>/memory.json`. CtrlSpeak seeds these files with `store_vector_memory: true`, `store_screenshots: true`, `retrieval_top_k: 5`, `retrieval_threshold: 0.75`, `max_vector_items: 5000`, `vector_ttl_days: null`, and `pii_redaction: false` so personas can independently tune retention and privacy.

The loader understands the following `identity.json` keys:

```json
{
  "name": "default",
  "description": "Friendly companion",
  "read_prompt_from_file": true,
  "prompt_file": "system_prompt.txt",
  "llm_model": "gemma3:1b",
  "llm_url": "http://localhost:11434/api/chat",
  "ollama_hardware": "gpu_only",
  "ollama_options": {
    "temperature": 0.9,
    "num_ctx": 4000,
    "num_gpu": 999,
    "main_gpu": 0,
    "gpu_split": [0.6, 0.4]
  },
  "tts": {
    "onnx_provider": "CUDAExecutionProvider",
    "device_id": 0
  },
  "voice": "am_michael",
  "require_text_cleaning": false,
  "vision": true,
  "tool": false,
  "memory_dir": "memory"
}
```

The `tts` block is optional, but the bundled identities include it so Kokoro explicitly requests the CUDA provider. Removing the block lets Kokoro fall back to its auto-detection logic (still preferring GPU providers whenever ONNX Runtime exposes them).

If `read_prompt_from_file` is `true`, the prompt file is read relative to the identity directory (unless an absolute path is provided). Omitting it falls back to the built-in default prompt.

When present, `ollama_options` is merged into the payload that SocialRobot sends to Ollama so you can tune generation parameters per identity. The optional `ollama_hardware` key controls how Ollama stages the weights:

- `cpu_only` – force the model to reside entirely in system RAM (`num_gpu` is set to zero).
- `cpu_and_gpu` – allow Ollama to combine VRAM and RAM (the Ollama default when no hardware preference is supplied).
- `gpu_only` – request that all available layers stay on the GPU (`num_gpu` is set to a high value, `main_gpu` defaults to `0`, and a `[0.6, 0.4]` `gpu_split` tells Ollama to place roughly 60% of the layers on GPU ID 0 and 40% on GPU ID 1 unless you override it).

CtrlSpeak applies the same options and hardware preference when it pre-warms the checkpoint via `/generate`, ensuring the residency chosen during warm-up matches the settings SocialRobot will use at runtime. When `gpu_only` is requested, the client refuses to continue if `/api/ps` reports CPU spillover so misconfigured VRAM budgets fail fast instead of silently degrading.

The optional `tts` object lets you steer Kokoro’s ONNX Runtime session:

- `onnx_provider` – Preferred provider name (for example `CUDAExecutionProvider`, `CPUExecutionProvider`, `DmlExecutionProvider`). Aliases such as `cuda` or `gpu` are normalized automatically. When omitted, Kokoro still attempts to use GPU providers first whenever ONNX Runtime reports them as available.
- `device_id` – Integer device index passed to GPU-capable providers. Use it to pin Kokoro to GPU 1 while Ollama consumes GPU 0, for example. Invalid or missing values are ignored.
- `providers` – Advanced escape hatch that accepts a JSON list mirroring the structure expected by `onnxruntime.InferenceSession` (for example `["CUDAExecutionProvider", {"name": "CPUExecutionProvider"}]` or `[{"name": "CUDAExecutionProvider", "options": {"device_id": 1}}]`).
- `provider_options` – Additional key/value pairs to merge into the provider options when you only need to override a handful of settings (for example `{ "cudnn_conv_algo_search": "DEFAULT" }`).

If you omit the block entirely, Kokoro continues to probe GPU providers automatically. When a requested provider is missing (for example because CUDA DLLs are not installed), the runtime logs which providers were skipped or why GPU initialisation failed before falling back to the default CPU session.

> **Note**
> The `memory_dir` field remains in legacy identity configs for compatibility, but CtrlSpeak always resolves runtime storage through the AppData helpers described above. Repository-relative memory paths are ignored so packaged builds stay read-only.

> **Tip**
> Update `${config_root}/identities/<identity>/memory.json` when you need to disable screenshot storage, change retrieval thresholds, adjust the vector-cap limit, apply a TTL, or enable the PII redactor for a specific persona.

### Bundled personas

CtrlSpeak ships with three ready-to-use personas:

- **assistant** – A Jarvis-inspired general helper backed by `gemma3:12b` with deterministic paragraph cleaning enabled.
- **default** – A lighter companion persona that uses `gemma3:1b` and keeps vision disabled by default.
- **einstein** – The deep-thinking and tool-planning specialist powered by `qwen3:14b`. CtrlSpeak appends `/think` to every Einstein turn (unless the user says `/no_think`) so Qwen3’s reasoning mode emits `<think>…</think>` plans before the final answer. The identity’s `identity.json` requests GPU-only execution, an 8 192 token context window, and the recommended sampling settings (`temperature` 0.6, `top_p` 0.95, `top_k` 20, `repeat_penalty` 1.1). Einstein also opts into `"hide_think": true`, which enables the `background_agents.manage_think` helper to strip `<think>` plans from the persisted chat history and Kokoro playback, drop the leading “Answer:” label before the visible reply, and immediately print a transient `Thinking...` placeholder in the chat window as soon as the user submits a message. Vision capture is disabled for this persona (`"vision": false`), so “look at my screen/clipboard” shortcuts only work with other identities. Stage the model in Ollama with a Modelfile equivalent to:

```
FROM hf.co/Qwen/Qwen3-14B-GGUF:Q4_K_M
PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 8192
PARAMETER num_gpu 999
SYSTEM You are Einstein, the deep-thinking tools agent for TrueAI. Thinking is always enabled unless a user explicitly includes /no_think.
TEMPLATE {{ .Prompt }}
```

Launch Ollama with `CUDA_VISIBLE_DEVICES=0,1` so both GPUs can host the checkpoint, then run `ollama create qwen3:14b -f ./Modelfile` followed by `ollama run qwen3:14b` to validate loading (or simply `ollama pull qwen3:14b` if you prefer the upstream defaults). SocialRobot automatically warms the model the first time you start Einstein and refuses to continue if Ollama reports CPU spillover while `ollama_hardware` is set to `gpu_only`.


The additional boolean keys control multimodal, cleaning, and future extensibility features:

- `require_text_cleaning` – Reserved for the paragraphizer workflow. The assets remain in place, but the helper is currently disabled so every identity relies on the deterministic plaintext scrub alone.
- `vision` – Enables image capture tooling documented in [`docs/tooling.md`](tooling.md). When `true`, SocialRobot listens for the spoken “look at my screen” and “look at my clipboard” commands, exposes matching context-menu actions on the floating logo, and routes captured images to the LLM. When `false`, the commands are ignored, the context-menu items are hidden, and no images are taken.
- `tool` – Reserved flag for forthcoming external tool integrations. It defaults to `false` today but can be toggled once tool calling is implemented.

CtrlSpeak now includes three bundled identities: `assistant` (vision enabled, text cleaning enabled), `default` (vision disabled, text cleaning disabled), and `einstein` (vision enabled, text cleaning enabled with Qwen3 reasoning defaults). All currently set `tool` to `false` until external tool integrations are wired up.

All face, mouth, and logo assets now live inside the identity directories; the legacy `third_party/social_robot/images/` placeholders have been removed so new personas should bundle their own art alongside `identity.json`. Likewise, shared prompt templates are deprecated—store any reusable system prompts with the identity that consumes them so packaging stays self-contained.

### Documentation ingestion workflow

The bundled personas keep their reference material up to date without overflowing the Chroma store:

1. `utils.bot_integration.start_bot` acquires the identity lock and calls both `refresh_document_memory(<identity>)` and `refresh_datetime_memory(<identity>)` for the bundled personas before SocialRobot launches. The documentation helper compares a SHA-256 hash of `README.md` plus the user-facing guides (`docs/bot_integration.md`, `docs/tooling.md`, `docs/user_flow.md`) against the tracker stored at `${data_root}/doc_memory/<identity>.json`, while the datetime helper snapshots the current local date, timezone, and locale under `${data_root}/datetime_memory/<identity>.json`. If each helper sees a matching hash inside the 24-hour cooldown, it logs a skip and continues immediately.
2. When the content or snapshot changed—or the cooldown elapsed—the helpers delete the relevant `category="documentation"` or `category="temporal_context"` rows, respect the identity’s `max_vector_items`, TTL, and PII-redaction preferences, and insert the fresh payloads (documentation arrives as multiple chunks, the datetime snapshot as a single chunk).
3. Terminal-only messages confirm the trigger (startup, keyword, hash change), report how many chunks were written, and indicate whether any older memories were evicted to honour the cap. The GUI and TTS layers remain silent.
4. Operators can say or type “update documentation” or “update datetime” (including via the Ctrl hotkey workflow) to force a refresh immediately. Forced runs bypass the cooldown so emergency edits or timezone changes land before the next user turn. The hotkey path refreshes the active identity, while in-session requests refresh whichever persona is currently running inside SocialRobot. Questions that mention the current date, time, or timezone automatically lower the retrieval threshold for the temporal-context chunk so the assistant responds with the stored snapshot instead of only referencing the tracker file.

Other identities can invoke the helper manually if they enable vector memory for documentation, but only the assistant, Einstein, and default personas do so automatically. Launching either persona also forces the LangGraph memory orchestrator on—even when `settings.json` never toggled the feature—so documentation retrieval always flows through the structured `assess_context → retrieve → plan_tools → call_tools → llm → persist` graph before the chat window becomes interactive.

Once the orchestrator is active, every user turn starts with a lightweight planner prompt that asks the LLM whether it wants additional context. The model must respond with `documentation`, `chat_history`, `date`, any comma-separated combination of those tokens, or `none`. A `none` answer skips the vector database entirely; otherwise the retrieval node limits its query to the requested buckets so documentation and temporal context are only attached when the model explicitly requests them. Questions about CtrlSpeak usage typically elicit `documentation`, while requests like “what’s my name?” surface `chat_history` so the assistant replays the correct conversation memories.

If the planner claims `none` but the user explicitly mentions the documentation, prior conversation history, or the current date/time, the orchestrator now overrides the plan based on those heuristics and still performs the relevant lookup. That safeguard catches prompts such as “look at our chat history” or “explain how to use this program” even when the planner misses the cue.

When the assistant or default persona receives an utterance that sounds like a “how do I use CtrlSpeak?” request (or when retrieval would otherwise return nothing), the LangGraph orchestrator relaxes the similarity check for `category="documentation"` memories and guarantees at least one documentation chunk appears in the retrieved context. Those passages are forwarded to the LLM inside a dedicated `Documentation excerpts` system message so the persona understands the text is canonical guidance and can quote it directly.

Every turn prints a terminal-only status line such as `[Memory] Vector store queried (plan=documentation, results=2, documentation=1, temporal=1).` or `[Memory] Vector store not queried (plan=none).` so operators can confirm both the planner’s decision and whether the vector database contributed context for the pending reply. The GUI and TTS surfaces remain silent.

You can switch identities from the command line with:

```powershell
python third_party/social_robot/main.py --identity ross
```

or by setting `BOT_IDENTITY=ross` before launching CtrlSpeak so the management window uses that persona.

## Voice keywords

SocialRobot loads [`tools/keywords.py`](tooling.md) at startup and registers one keyword set per identity directory. The following phrases are recognized out of the box:

- **look at my screen** – Captures a screenshot via `tools/vision.py` when the active identity has `vision: true`.
- **look at my clipboard** – Pulls the most recent image from the system clipboard and forwards it like a screenshot.
- **update documentation** – Forces the active identity (assistant or default) to reload Markdown documentation immediately, bypassing the cooldown and logging status only to the terminal.
- **update datetime** – Forces the active identity to refresh the temporal-context snapshot (current date, timezone, locale/country hints) immediately, bypassing the cooldown and logging status only to the terminal.
- **chat with `<identity>`** – Immediately relaunches SocialRobot with the target persona through the transcription server so the parent process coordinates the shutdown and restart (requests that target the already-active identity are ignored).
- **goodbye `<identity>`** – Immediately ends the current conversation when delivered through voice (including the stdin control channel) so the parent process controls the teardown. SocialRobot disables voice mode and unwinds the audio stack but leaves the chat window running so the user can close it manually; typed “goodbye …” phrases remain regular chat messages.

The push-to-talk workflow (hold the right Ctrl key while speaking) shares the same keyword registry. When VAD is idle because no bot session is active, saying “chat with assistant” through the hotkey launches that identity and skips text injection entirely. If another persona is already running, the helper first asks it to exit cooperatively via `utils.bot_integration.request_goodbye()` and falls back to `stop_bot()` only when the child process fails to exit within the timeout. Spoken “goodbye <identity>” commands are intercepted in the CtrlSpeak main process (the transcription server), which issues the same graceful-then-hard stop sequence used by the tray menu while leaving the chat window for the user to close. Each shutdown stage emits debug-level entries under `third_party.social_robot.main` in `${data_root}/logs/ctrlspeak.log` (for example, `%APPDATA%\CtrlSpeak\logs\ctrlspeak.log` on Windows), so you can see exactly which component executed when diagnosing a stalled goodbye. All of these phrases are matched fuzzily, so punctuation or light speech-to-text substitutions (for example, “chat was assistant”) still trigger the expected behaviour unless a document explicitly opts out.

> **Do not** move the goodbye/chat keyword detection back into SocialRobot or another worker thread. Keeping the logic in the CtrlSpeak main process is a hard requirement so the parent can enforce the proven shutdown path. Route any future conversation controls through the same helpers described above and send explicit stdin commands to the bot only after the parent has taken ownership of the request.

When you add new keywords, update `tools/keywords.py`, refresh [`docs/tooling.md`](tooling.md), and adjust the system prompts for any identities that should advertise the new commands. The assistant prompt bundled with CtrlSpeak now explicitly mentions the clipboard trigger and instructs the model to guide users toward the exact phrases when they hint at wanting a capture.

## LangGraph orchestration and persistence

Setting `use_langgraph_memory_orchestrator: true` in `settings.json` (or exporting `CTRLSPK_USE_LANGGRAPH_MEMORY_ORCHESTRATOR=1`) routes every conversation turn through `utils.memory_orchestrator`:

> **Bundled personas:** When you start the assistant, Einstein, or default personas, CtrlSpeak automatically flips this setting on (persisting it back to `settings.json`) so documentation ingestion and retrieval always use the LangGraph path even on pristine installs.

1. **assess_context** – ask the LLM which context buckets (`documentation`, `chat_history`, `date`, or `none`) it wants for the turn and cache the response.
2. **retrieve** – query Chroma for up to `retrieval_top_k` memories above `retrieval_threshold`, limited to the planner’s requested categories; empty stores or low scores short-circuit.
3. **plan_tools** – reserved for future branching/tooling (currently a pass-through node).
4. **call_tools** – executes planned tools (no-ops today).
5. **llm** – calls the Ollama client with the existing history plus any retrieved memory preamble.
6. **persist** – enqueues atomic JSONL appends and Chroma upserts on a background worker so Kokoro playback can start immediately.

Persistence enforces the 10 MB/5-file JSONL rotation, 5 000-vector cap with LRU eviction, optional TTL, and the `pii_redaction` toggle before embedding. Per-turn traces (`run_<timestamp>_<correlation>.json`) and CSV metrics (`retrieval_hits`, `avg_similarity`, `persist_latency_ms`, `evictions`, `lock_wait_ms`) accumulate under `${data_root}/bot_memory/<identity>/traces` for post-mortems. If initialization fails (missing AppData, Chroma errors, LangGraph import issues), SocialRobot prints “LangGraph orchestrator failure; falling back to legacy conversation.”, logs the exception, and resumes the synchronous history writer.

Run `python main.py --health [--health-identity <name>]` to probe the same pipeline. The command verifies data/log root writability, acquires/releases the identity lock, initializes the Chroma collection, and reports embedder metadata. Failures return a non-zero exit code and a JSON payload explaining the failing check.

## SocialRobot entrypoint responsibilities

`third_party/social_robot/main.py` remains the authoritative entrypoint for the Chat with Bot workflow. CtrlSpeak launches it as a subprocess from `utils.bot_integration.start_bot`, passes the resolved identity settings and CtrlSpeak STT URL, and relies on its stdin control channel for graceful shutdowns.【F:utils/bot_integration.py†L667-L760】【F:third_party/social_robot/main.py†L251-L399】 The module bootstraps speech recognition, Ollama chat streaming, Kokoro playback, optional preprocessing, and the animated face/logo UI before wiring the stdin listener that handles `goodbye` commands issued by the parent process.【F:third_party/social_robot/main.py†L146-L532】 Removing or partially deleting this file will break bot startup and teardown, so trim functionality only when you can update every caller and test path accordingly.

## Background agents

The speech rewrite pass lives under `background_agents/tts_preprocessing_agent`. The directory mirrors an identity folder:

- `identity.json` declares the Gemma 3 1B model, its Ollama URL, deterministic decoding options (temperature 0, top_p 1, repeat_penalty 1, mirostat disabled, fixed seed, and a `"\n\n"` stop), and the prompt files to load.
- `system_prompt.txt` contains the Profile Paragraphizer instructions that turn label/value profile dumps into a single paragraph beginning with “Here’s what I know about you.”
- `header_text.txt` remains available for legacy configurations that still set `preamble` to `header` or `both`, but the bundled identity now defaults to `system` and ignores the header.

`background_agents/tts_preprocessing_agent/background_agent.py` still constructs a non-streaming `OllamaClient` and exposes `load_tts_preprocessing_agent()` for future use. The Profile Paragraphizer prompt and identity metadata remain unchanged, but SocialRobot no longer calls the helper at runtime; deterministic scrubbing handles every reply instead.【F:background_agents/tts_preprocessing_agent/background_agent.py†L1-L199】

CtrlSpeak treats the agent as a first-class asset for identities that request it: `utils.bot_integration.start_bot` only stages the Gemma weights and pre-warms the checkpoint when the selected identity advertises `require_text_cleaning: true`, skipping the additional download and warm-up otherwise.【F:utils/bot_integration.py†L116-L214】【F:utils/bot_integration.py†L635-L790】 Packaged builds bundle `background_agents/` so the helper is present in single-file executables.【F:packaging/CtrlSpeak.spec†L42-L63】【F:packaging/CtrlSpeak_Watcher.spec†L42-L63】【F:utils/build_exe.py†L22-L82】

## Runtime Requirements

Using Chat with Bot requires:

1. CtrlSpeak running in **Client + Server** mode with the transcription server active.
2. The dependencies listed in the repository-level `requirements.txt` (pygame, Kokoro-ONNX, faster-whisper, etc.). Installing the root requirements is sufficient; no separate `third_party/social_robot/requirements.txt` file exists anymore.
3. Optional: a reachable Ollama server if you want real LLM responses. Without one, the bot echoes a graceful fallback derived from the user's input.

## Launching the Bot Manually

The bot can be exercised from the command line via `utils.bot_integration.run_bot_test`:

```powershell
py -3 run_test_script.py
```

This script starts the CtrlSpeak server with `--start-server-only`, waits for `/ping` to report healthy, streams the sample `assets/test_16k_mono.wav` through SocialRobot, prints the transcription and LLM response, and shuts everything down cleanly via `/kill`.

To integrate with your own automation, call:

```python
from utils.bot_integration import run_bot_test
run_bot_test("assets/test_16k_mono.wav", identity="default")
```

The helper reuses an existing server when you pass `stt_url=...`, and you can forward identity-specific overrides such as `prompt_file`, `system_prompt`, or `voice`.

## Configuring Kokoro TTS for GPU-first inference

Kokoro relies on ONNX Runtime. Because the root `requirements.txt` installs `onnxruntime-gpu==1.23.0`, the bundled identities request the CUDA provider by default and Kokoro automatically attempts to run on the GPU. When ONNX Runtime cannot honor that request—whether because the host has no GPU or the CUDA DLLs are missing—the engine falls back to the CPU and prints a warning so operators know TTS is no longer on the GPU. To customise that behaviour or pin Kokoro to a different device:

1. Set the identity’s `tts` block to request the provider you want (`"onnx_provider": "CUDAExecutionProvider"` and an optional `"device_id"`). The default personas already do this with `device_id: 0`.
2. Alternatively, export `BOT_TTS_PROVIDER`, `BOT_TTS_DEVICE_ID`, or `BOT_TTS_PROVIDERS` before launching Chat with Bot to override the identity defaults without touching JSON.
3. Launch the assistant and watch `nvidia-smi`—you should see a small memory bump on the specified GPU when TTS audio is generated. Successful GPU initialisation prints `-> Kokoro TTS using GPU providers: [...]` so you can confirm it at a glance.

If the chosen provider is unavailable (for example because the machine lacks `cufft64_11.dll` or cuDNN), CtrlSpeak logs which providers were skipped and emits a `Falling back to CPU` warning before proceeding. You can continue to control visibility with `CUDA_VISIBLE_DEVICES` or similar environment variables when you need to hide GPUs from ONNX Runtime entirely.

## Environment Overrides

| Variable | Purpose |
| --- | --- |
| `BOT_IDENTITY` | Name of the identity directory to load (default `default`). |
| `BOT_IDENTITIES_DIR` | Override the base directory that holds identity folders. |
| `BOT_PROMPT_FILE` | Explicit prompt file path to use when launching SocialRobot. |
| `BOT_SYSTEM_PROMPT` | Inline system prompt string (used when not reading from a file). |
| `BOT_MEMORY_DIR` | Override the memory directory for the active identity. |
| `BOT_LLM_URL` | Ollama endpoint (default `http://localhost:11434/api/chat`). |
| `BOT_LLM_MODEL` | Ollama model name (default `gemma3:1b`). |
| `BOT_VOICE` | Kokoro voice (default `hm_omega`). |
| `CTRLSPEAK_STT_URL` | Force SocialRobot to target a specific CtrlSpeak server. |
| `BOT_TTS_PROVIDER` | Override the Kokoro ONNX Runtime provider (e.g. `CUDAExecutionProvider`). |
| `BOT_TTS_DEVICE_ID` | GPU device index for providers that support it. |
| `BOT_TTS_PROVIDERS` | JSON list passed directly to `onnxruntime.InferenceSession` for Kokoro. |
| `BOT_TTS_PROVIDER_OPTIONS` | JSON object merged into the provider options when Kokoro is created. |

Each option also has a matching CLI flag in `third_party/social_robot/main.py` (for example `--identity`, `--prompt-file`, `--system-prompt`, `--memory-dir`).

## Provisioning Ollama for a new multi-GPU workstation

When you move to a fresh Windows machine and need Ollama to split work across multiple GPUs, perform the following host setup before launching CtrlSpeak or issuing API calls:

1. **Confirm the Ollama port and stop previous processes.** Check `%LOCALAPPDATA%\Ollama\server.log` (or run `netstat -ano | find "11434"`) to verify the daemon is bound to `0.0.0.0:11434`. End any lingering `ollama.exe`, `ollama_runners.exe`, or `wslrelay.exe` processes that might still own the port so the restart is clean.
2. **Set persistent GPU environment variables.** From an elevated PowerShell session, run:
   ```powershell
   setx /M CUDA_VISIBLE_DEVICES "0,1"
   setx /M OLLAMA_SCHED_SPREAD "1"
   ```
   These values survive reboots; log off (or reboot) so the Ollama service inherits them. Listing both device IDs keeps CUDA discovery deterministic, and `OLLAMA_SCHED_SPREAD=1` tells Ollama to balance layers instead of piling everything onto the first GPU.
3. **(Optional) Pre-stage a default split.** If you want Ollama to remember a custom fraction even without client overrides, create `%LOCALAPPDATA%\Ollama\server.json` with:
   ```json
   {
     "gpu_split": [0.6, 0.4]
   }
   ```
   Skip this file when you prefer to manage splits entirely from the API payload.
4. **Restart the Ollama service.** Run `ollama serve` (or launch the desktop UI) after the environment variables are in place. Watch `server.log` for fresh `inference compute` entries that enumerate both GPUs and confirm no port-in-use warnings appear.
5. **Validate GPU usage.** Issue a short test prompt (CLI or API) while monitoring `nvidia-smi`. You should see both cards pick up load, and the log’s `GPULayers` breakdown should mention GPU `0` and `1`. Adjust the per-request `gpu_split` array or `main_gpu` index if the distribution needs fine-tuning.

Once the host honors these defaults, CtrlSpeak’s `gpu_only` hardware mode—combined with the `[0.6, 0.4]` request-level `gpu_split`—keeps the model resident on both GPUs without additional machine-specific tweaks.

## Assistant Model Downloads

Launching Chat with Bot from the management window now verifies that the Ollama model required by the selected identity is available before SocialRobot starts. CtrlSpeak queries the Ollama API to see whether the model is already cached and, when it is missing, reuses the same welcome workflow that Whisper and CUDA downloads use: the intro video plays, fun facts rotate, and the lockout window reports live status with a cancel button. Once the download succeeds the window closes automatically and the bot process launches; cancellation or errors stop the launch and surface the failure in the lockout message so the user can try again after resolving the issue.【F:utils/bot_integration.py†L61-L213】【F:utils/models.py†L1308-L1484】

## Screenshot workflow

When you launch an identity with `vision: true` (for example the bundled **Assistant** persona) and say “look at my screen,” SocialRobot captures the current desktop. The helper now writes the PNG to `${data_root}/bot_memory/<identity>/screenshots/current.png` (along with a `metadata.json` descriptor) so each bot keeps only its most recent image on disk; new captures atomically replace the previous file instead of accumulating a gallery. Saying “look at my clipboard” (or choosing **Look at my Clipboard** from the floating logo) pulls the most recent image from the system clipboard via `tools/vision.py` and flows through the same single-image store. Keyword detection for both phrases lives in `tools/keywords.py`, which keeps the trigger vocabulary centralized as new commands are added. LangGraph inspects the user’s wording every turn: if the request references the stored image or the capture workflow just ran, the orchestrator automatically loads `current.png`, attaches the PNG to the Ollama request, and records only a file reference in the JSONL history. If a capture fails—because PyAutoGUI cannot access the display, the clipboard has no image, or dependencies are missing—the bot logs the issue and continues as a text-only exchange. Identities with `vision: false` skip these hooks entirely; the floating logo omits both context-menu options and voice commands fall back to a standard text-only exchange.

As soon as you pick an identity, CtrlSpeak now pings the configured Ollama endpoint with that profile’s model so the checkpoint is fully loaded before you speak. This avoids the first-turn lag that previously occurred while Ollama initialized the weights after receiving the initial utterance.

## Clearing stored memory

Use the **Clear Bot Memory** button in the management window when you need to wipe a persona’s stored context. After you choose an identity and confirm the prompt, CtrlSpeak removes that profile’s `${data_root}/bot_memory/<identity>/conversation/conversation.jsonl` log (plus any rotated archives) and deletes the `screenshots/`, `chroma/`, and `traces/` subdirectories so cached captures, vector embeddings, traces, and metrics are cleared. If nothing exists under the identity’s AppData tree, the dialog reports that there is nothing to clear.

## GUI Workflow

1. Start CtrlSpeak in Client + Server mode.
2. Right-click the tray icon, choose **Manage CtrlSpeak**, then use the Assistants card to review identity availability badges and click **Chat with Bot**.【F:utils/gui.py†L1424-L1455】
3. The management UI toggles the bot: click once to launch, again to stop. The active persona's badge switches to **active** while other identities remain marked **available**.【F:utils/gui.py†L1730-L1812】
4. The chat window opens in text mode. Type your first message or click the microphone button to switch to voice mode. Voice mode starts the VAD listener, shrinks the window to a read-only transcript, restores the floating logo, and speaks replies aloud.
5. When voice mode is active you can right-click the transparent logo to open its context menu. Choose **Look at my Screen** to capture the desktop or **Look at my Clipboard** to forward the latest snip stored in the clipboard. Both actions mirror the spoken commands and share the tooling documented in [`docs/tooling.md`](tooling.md). The menu still includes **Quit** when you need to close the bot quickly.

The SocialRobot process keeps running if you close the management window—you can reopen it later without interrupting the conversation. To shut the bot down, either click **Stop Chat with Bot** in the management window or choose **Quit** from the floating logo's context menu.

