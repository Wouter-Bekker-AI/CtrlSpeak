# Tooling Reference

This document is the authoritative index for reusable tooling that ships with CtrlSpeak. Every helper that exposes capabilities to AI agents, automation flows, or user-triggered actions must be documented here so contributors know what exists, how it works, and how to extend it safely.

CtrlSpeak distinguishes between interactive tooling and the system-managed services that keep the LangGraph workflow healthy. Modules under `tools/` are invoked directly by personas or users, while the background agents in `background_agents/` maintain temporal context, clean up `<think>` output, and (previously) refreshed documentation. The documentation pipeline is now paused while we prepare the Docling RAG integration, but the overall architecture remains described in the [Chat with Bot speech pipeline](../README.md#chat-with-bot-speech-pipeline) and [LangGraph orchestration and persistence](bot_integration.md#langgraph-orchestration-and-persistence) guides.

The guidance below applies to the `tools/` package and any future modules that belong to it. When you add, modify, or remove a tool you **must** update this document in the same pull request.

## Directory layout

```
ctrlspeak/
├── tools/
│   ├── __init__.py
│   ├── keywords.py
│   ├── message_management.py
│   ├── vision.py
│   └── goose_tool.py
```

- `tools/__init__.py` exposes the modules that make up the shared tooling surface. Import helpers via `from tools import vision` (or `keywords`) so the package can evolve without breaking downstream code.
- Additional tooling (for example, browser automation or document parsing) should live beside `vision.py` inside this directory. Each module must document its public API in this file before it is merged.
- `tools/goose_tool.py` wraps the Goose CLI so Einstein can delegate file inspection, edits, searches, and executions to a purpose-built automation agent.

## Goose automation helper (`tools/goose_tool.py`)

`goose_tool.py` exposes a single entry point that launches Goose in headless mode and returns the resulting transcript to CtrlSpeak:

| Function | Description |
| -------- | ----------- |
| `goose_query(prompt, *, model="qwen3:14b", mode="auto", provider="ollama", goose_exe="goose", stream=False)` | Runs `goose run` with the supplied natural-language prompt. By default the command executes non-streaming so the combined transcript is returned once Goose exits (the wrapper still mirrors stdout to the console). Set `stream=True` to forward output live through `subprocess.Popen`. The call raises `RuntimeError` when Goose exits non-zero or produces no output. |

### Usage guidelines

- The prompt must be a non-empty string. The optional `mode` argument is validated against Goose's supported values (`auto`, `smart_approve`, `approve`, `chat`).
- Goose runs without a persistent session (`--no-session`) and automatically enables the built-in `developer` tool so it can open shells, edit files, and manage approvals on Einstein's behalf.
- The wrapper sets `GOOSE_MODE` in the environment before launching Goose so downstream scripts honour the requested approval strategy.
- By default stdout is collected and echoed after Goose completes. Opt into live streaming with `stream=True` when real-time updates are necessary.

### LangGraph integration

- Einstein's tool belt now exposes a single function, `goose_tool_query(prompt, mode?, model?, provider?, goose_exe?, stream?)`. The orchestrator expects the model to describe the desired filesystem or shell task in natural language and let Goose execute it.
- The orchestrator validates that every tool call includes a prompt. Missing prompts are rejected and surfaced back to the model so it can refine the request.
- Goose executions always run with `stream=True` so the CLI's live output mirrors into the CtrlSpeak terminal. Once Goose exits the orchestrator stores its transcript (trimmed to 4 000 characters) as a `tool` role entry in the session history so later LLM calls can reference the result, while the chat window continues to show only the assistant's final reply.
- The LangGraph workflow no longer stages Goose plans heuristically. Each turn simply presents the tool schema to Einstein and honours whatever tool call it issues, keeping responsibility for when and how Goose is used entirely with the LLM.


## Vision tooling (`tools/vision.py`)

The `vision` module centralizes every capture routine that SocialRobot and other agents rely on. It keeps heavy dependencies lazy, respects the project’s logging rules, and writes persistent artifacts under the caller’s memory directory.

### Public API

| Function | Purpose |
| --- | --- |
| `capture_screenshot(target_dir: Optional[Path] = None, *, filename_prefix: str = "screenshot") -> VisionCapture` | Captures the active desktop using PyAutoGUI. On success it returns a `VisionCapture` with the base64 payload and an optional PNG written to `target_dir`. |
| `capture_clipboard_image(target_dir: Optional[Path] = None, *, filename_prefix: str = "clipboard") -> VisionCapture` | Reads the current clipboard via Pillow’s `ImageGrab.grabclipboard()`, extracts the first image payload, and persists it just like a screenshot. |
| `VisionCapture` | Dataclass wrapper that surfaces `image_b64`, `saved_path`, `source` (`"screen"` or `"clipboard"`), and an optional `error` string. The `.success` property is `True` when the capture produced an image. |

### Implementation details

- Dependencies (PyAutoGUI, Pillow, pygame) are imported inside the functions so importing `tools.vision` never loads GUI libraries unless a capture actually runs.
- Errors are reported through the shared logger obtained from `utils.config_paths.get_logger`, which means failures show up in `${data_root}/logs/ctrlspeak.log` (e.g., %APPDATA%\CtrlSpeak\logs\ctrlspeak.log on Windows) while callers can still recover gracefully.
- Successful captures store their files in `${data_root}/bot_memory/<identity>/screenshots` when the caller provides the identity’s memory directory. The LangGraph image-memory helpers collapse the folder down to a single `current.png` plus `metadata.json`, replacing the PNG atomically whenever a new capture arrives so identities keep only their latest image on disk.
- A lightweight shutter sound is played after every successful capture. The helper falls back silently when pygame is missing or audio initialization fails, so tests and headless environments stay stable.

### Usage pattern

```python
from pathlib import Path
from tools import vision

memory_dir = Path("/tmp/session")
result = vision.capture_screenshot(memory_dir / "screenshots")
if result.success:
    payload = result.image_b64  # Base64 string suitable for multimodal LLM APIs
    local_path = result.saved_path  # Absolute Path or None when persistence failed
else:
    print(f"Capture failed: {result.error}")
```

The SocialRobot integration uses the same workflow for both the **Look at my Screen** and **Look at my Clipboard** commands. Voice requests that match either phrase, stdin control messages, and the floating logo’s context menu all share this module.

### Testing guidance

- Unit tests live in `tests/core/test_tools_vision.py`. They mock PyAutoGUI and Pillow so the suite runs without a GUI or clipboard.
- When you extend the module, add corresponding tests under `tests/core/` and mark them with `@pytest.mark.core_headless` unless GUI access is unavoidable.
- Always run the compile smoke test and the `core_headless` pytest marker before submitting changes (see `docs/PROJECT_GROUND_RULES.md`).

## Keyword registry (`tools/keywords.py`)

The keyword registry keeps voice and command triggers in one place so assistants can react consistently as new tooling comes online. Each keyword specifies the phrase to match, the logical category it belongs to, and a payload string that downstream code uses to select an action.

### Public API

| Function | Purpose |
| --- | --- |
| `configure_identity_keywords(identities: Iterable[str]) -> None` | Populates conversation keywords (for example, “chat with vision”) using the available identity folder names. Must be called whenever the identity roster changes so voice triggers stay in sync. |
| `detect_vision_keyword(text: str) -> Optional[KeywordMatch]` | Returns the first vision keyword matched in `text`, or `None` when no trigger is present. |
| `detect_conversation_start_keyword(text: str) -> Optional[KeywordMatch]` | Detects `chat with <identity>` requests and returns the matching keyword metadata. |
| `detect_conversation_end_keyword(text: str) -> Optional[KeywordMatch]` | Detects `goodbye <identity>` requests for the active conversation. |
| `iter_keyword_matches(text: str, keywords: Iterable[Keyword]) -> Iterator[KeywordMatch]` | Generator that yields every keyword match found in the provided text. |
| `get_vision_keyword(payload: str) -> Optional[Keyword]` | Retrieves the configured keyword metadata for `"screen"`, `"clipboard"`, or future vision payloads. |
| `Keyword` / `KeywordMatch` | Lightweight dataclasses describing configured keywords and successful matches, including the compiled regex pattern for downstream substitutions. |

### Usage pattern

```python
from tools import keywords

match = keywords.detect_vision_keyword("look at my clipboard")
if match:
    if match.keyword.payload == "clipboard":
        handle_clipboard()
    elif match.keyword.payload == "screen":
        handle_screen()
```

SocialRobot consumes this module to decide whether the user asked for a screenshot or clipboard capture and to switch between identities when the user says “chat with Vision/Reception.” The CtrlSpeak transcription server inspects the same registry before forwarding speech to SocialRobot so both “chat with …” and “goodbye …” requests are handled in the parent process. This keeps shutdowns and relaunches consistent with the tray and hotkey controls. When you add new keywords, update the relevant identity system prompts (so assistants know which phrases to suggest) and refresh any UX documentation that references the trigger vocabulary.

### Keyword reference

The application responds to the following spoken or typed keywords. Each phrase is matched case-insensitively and with a fuzzy tolerance for punctuation or common speech-to-text substitutions. The clipboard trigger continues to accept near-miss variations such as “look at my slipboard” or “look at my clupboard,” and the same tolerance now applies to the conversation controls.

| Keyword | Category | Action |
| --- | --- | --- |
| `look at my screen` | Vision | Capture the user’s current desktop, store it as the identity’s latest image, and play the camera shutter sound. |
| `look at my clipboard` (including close variants like “look at my slipboard”) | Vision | Read the latest image from the system clipboard, replace the identity’s stored image, and play the camera shutter sound. |
| `update documentation` (also accepts “refresh documentation”) | Memory maintenance | Triggers the legacy documentation refresh helper. The command now logs that ingestion is disabled while Docling integration work is underway. |
| `update datetime` (accepts “update date time” or “refresh date time”) | Memory maintenance | Force the active identity to store the latest local date, timezone, and locale snapshot in vector memory, bypassing the 24-hour cooldown. |
| `chat with <identity>` | Conversation start | Relaunch the bot using the requested identity via the transcription server (ignored if that identity is already active). |
| `goodbye <identity>` | Conversation end | Play a farewell in the active persona’s voice and then shut down the conversation from the CtrlSpeak main process. |
| `quit control speak` | System | From the Lobby stage, speaks “goodbye” with the Reception persona’s voice before shutting down CtrlSpeak (ignored while a conversation is active). |

> **Note:** SocialRobot’s text chat window no longer treats typed “goodbye <identity>” phrases as keywords. Those messages are delivered to the bot verbatim; only spoken requests (or ones injected through the stdin control channel) trigger the shutdown helpers.

### Conversation keywords

`configure_identity_keywords()` keeps the voice trigger list synchronized with the identity folders. Once configured, the helpers recognize:

- `chat with <identity>` – immediately relaunches SocialRobot with the requested identity via the transcription server (no action is taken when the user asks for the already-active persona). Close variants like “chat was vision” or “chat was receptionist” are recognised automatically.
- `goodbye <identity>` – immediately ends the current conversation, first speaking a “goodbye” line with the active persona’s voice before the CtrlSpeak main process shuts the bot down. Light punctuation (for example, “goodbye, vision” or “goodbye receptionist”) remains valid.

The `<identity>` placeholder uses the directory names under `third_party/social_robot/personas/`. Call `configure_identity_keywords()` whenever you add or remove identities (for example, during application startup) to keep the registry current.

The same registry now powers the push-to-talk workflow: when the user holds the right Ctrl hotkey, CtrlSpeak transcribes the utterance and checks it against these keywords before typing anything. Phrases like “chat with vision” launch the corresponding bot immediately, “chat with reception” first stops any existing session before starting the Reception persona, and “goodbye <identity>” routes through the same shutdown helper used by the tray menu. Because the text insertion path never runs for handled keywords, make sure any new phrases you add here have matching automation hooks so the hotkey and transcription server remain in sync with voice-triggered behaviour.

## Transcript cleanup background agent (`background_agents/transcript_cleanup_agent/`)

CtrlSpeak normalises Whisper transcripts before keyword routing via the transcript cleanup background agent. The helper applies a configurable set of substitutions so the hotkey workflow, transcription server, and SocialRobot UI all see canonical phrases even when the speech-to-text model mishears them (for example, “clubboard” instead of “clipboard” or “chat with defunct” instead of “chat with default”).

### Public API

| Function | Purpose |
| --- | --- |
| `normalize_transcript(text: str) -> TranscriptCleanupResult` | Normalises the supplied transcript, returning the cleaned string and a tuple of `TranscriptCorrection` records describing each change. When the agent resources are unavailable the original text is returned unchanged. |
| `load_transcript_cleanup_agent(...) -> TranscriptCleanupAgent` | Loads the agent using the on-disk configuration bundle. Most callers should rely on `normalize_transcript` so resources are cached automatically. |

### Configuration bundle

- `identity.json` defines the correction thresholds, audit-log parameters, and pointers to the other resource files.
- `variant_map.json` lists canonical phrases mapped to their known misrecognitions. Add entries here when Whisper returns a consistent misspelling; the keys are treated as the desired replacement text.
- `supplemental_phrases.txt` seeds the fuzzy matcher with project terminology (for example, documented clipboard and chat commands) so similar phrases are normalised even without an explicit variant entry.
- `${data_root}/langgraph_agents/transcript_cleanup_agent/custom_variant_map.json` is a user-editable overlay that lives outside the packaged application. Operators can add or modify entries at runtime (even while CtrlSpeak is running) without touching the read-only resources bundled with the executable. The agent re-reads this file before each cleanup pass, merging the custom entries with the baked-in `variant_map.json` so new corrections apply immediately.

The loader merges these resources with the authoritative keyword list exposed by `tools.keywords`. During normalisation the agent first applies the explicit variants, then performs a fuzzy sweep over every tracked phrase using the thresholds in `identity.json`. All corrections are recorded in `${logs_root}/transcript_cleanup.log`, capped at 256 KiB with rotation so operators can audit what changed.

### Integration points

- `utils.system.handle_transcribed_text_from_hotkey` cleans transcripts before testing any keywords so push-to-talk users get the corrected behaviour immediately.
- `utils.system.handle_transcription_keyword` returns the cleaned text to HTTP clients when no keyword is triggered, ensuring downstream services receive the canonical phrases.
- `third_party.social_robot.main._handle_user_request` normalises every utterance before it reaches SocialRobot, printing the adjusted phrase in the console when a correction occurs so operators can see what changed.

### Testing

- Unit tests live in `tests/core/test_transcript_cleanup_agent.py`.
- Keyword routing tests under `tests/core/test_system_hotkey_keywords.py` and `tests/core/test_system_transcription_keywords.py` cover common misrecognitions (such as “chat with defunct”) to ensure the hotkey and transcription paths keep working as new variants are added.

### Testing guidance

- Unit tests live in `tests/core/test_tools_keywords.py` and validate phrase detection along with payload lookups.

## Message management (`tools/message_management.py`)

The message management helpers keep assistant replies safe for text-to-speech playback and storage by deterministically stripping Markdown artefacts.

### Public API

| Function | Purpose |
| --- | --- |
| `requires_force_plaintext(text: str, drop_chars: Iterable[str] = DEFAULT_DROP_CHARS, bullet_prefixes: Iterable[str] = DEFAULT_BULLET_PREFIXES) -> bool` | Returns `True` when `force_plaintext` would mutate the text. Checks for drop characters anywhere in the reply and for leading bullet markers on each line. |
| `force_plaintext(text: str, drop_chars: Iterable[str] = DEFAULT_DROP_CHARS, bullet_prefixes: Iterable[str] = DEFAULT_BULLET_PREFIXES) -> str` | Removes the configured drop characters, strips recognised bullet prefixes, collapses internal whitespace runs, and trims the result so TTS and persistence layers receive clean plaintext. |

### Default behaviour

- `DEFAULT_DROP_CHARS` removes `*`, `#`, `_`, `` ` ``, `>`, and `|` characters.
- `DEFAULT_BULLET_PREFIXES` normalise leading `-`, `+`, `•`, and `*` markers.
- Passing custom iterables allows callers to extend or tighten scrubbing rules when different formatting must be preserved.

### Testing guidance

- Companion tests live in `tests/tools/test_message_management.py` and cover both detection and scrubbing behaviours.

## Adding new tooling modules

1. Create a new module under `tools/` (for example `browser.py` or `parsing.py`).
2. Export it from `tools/__init__.py` so callers can import it consistently.
3. Document the new functionality in this file with API tables, usage examples, and testing notes.
4. Update any feature documentation (README, integration guides, UX walkthroughs) that depends on the new capability.
5. Ensure packaging scripts pick up the new module if they rely on explicit include lists.

Maintaining `docs/tooling.md` keeps CtrlSpeak’s tooling surface discoverable and prevents future refactors from duplicating functionality. Treat this file as the single source of truth for agent-facing helpers.

## Documentation ingestion helper (`background_agents/document_memory_agent.py`)

CtrlSpeak previously preloaded Markdown documentation into vector memory so the Vision, Einstein, and Reception identities always had the latest reference material before a conversation started. That workflow is on hold while we replace it with the Docling RAG agent. The `refresh_document_memory(identity, *, force=False, reason=None)` helper now acts as a compatibility shim: it prints `[DocMemory] Documentation ingestion disabled…` and returns `True` immediately so existing hotkeys, keywords, and tests continue to function without touching Chroma.

You can continue to invoke the helper when responding to the `update documentation` keyword or preparing an identity for launch, but the call only records the skip. Once the Docling pipeline lands we will restore meaningful ingestion behaviour.

The LangGraph orchestrator continues to offer retrieval planning and Goose tooling when explicitly enabled, but without documentation embeddings it focuses on conversation history management. Vector storage, eviction, and embedding behaviour remain documented in `utils.vector_memory.VectorMemoryStore` for projects that opt into custom memories.

## Date/time context helper (`background_agents/datetime_memory_agent.py`)

Alongside documentation, CtrlSpeak primes each identity with a snapshot of the host’s current date, timezone, and locale data so Vision can answer questions like “what day is it?” without re-querying the operating system. `refresh_datetime_memory(identity, *, force=False, reason=None)` orchestrates the workflow:

- Collects the local date (`YYYY-MM-DD` plus weekday and month names), the timezone abbreviation and UTC offset, and any locale or country code detectable from the OS environment.
- Serialises the snapshot into a single vector-memory chunk tagged with `category="temporal_context"`, `kind="current_date"`, and a `snapshot_hash` so duplicates can be detected.
- Deletes any prior temporal-context entries before inserting the new chunk, reusing the identity’s vector settings (`max_vector_items`, TTL, PII redaction) just like the documentation helper.
- Records the refresh timestamp and `snapshot_hash` under `${data_root}/datetime_memory/<identity>.json`, enforcing a 24-hour cooldown unless the data changes or a forced refresh is requested.
- Emits terminal-only status updates such as “Date/time injection complete…” while keeping the GUI silent.

The helper runs automatically for the Vision, Einstein, and Reception personas during startup, ensuring each session begins with a fresh temporal snapshot even while documentation ingestion is paused. Operators can force the refresh at any time via the `update datetime` keyword (including the fuzzy variant “update date time”) from the hotkey workflow or within an active SocialRobot session. Any user question that references the current date, time, day, or timezone automatically lowers the retrieval threshold for the temporal chunk, prompting the orchestrator to attach a `Temporal context` system message that contains the stored snapshot so the personas can answer directly.
