# Tooling Reference

This document is the authoritative index for reusable tooling that ships with CtrlSpeak. Every helper that exposes capabilities to AI agents, automation flows, or user-triggered actions must be documented here so contributors know what exists, how it works, and how to extend it safely.

The guidance below applies to the `tools/` package and any future modules that belong to it. When you add, modify, or remove a tool you **must** update this document in the same pull request.

## Directory layout

```
ctrlspeak/
├── tools/
│   ├── __init__.py
│   ├── keywords.py
│   └── vision.py
```

- `tools/__init__.py` exposes the modules that make up the shared tooling surface. Import helpers via `from tools import vision` (or `keywords`) so the package can evolve without breaking downstream code.
- Additional tooling (for example, browser automation or document parsing) should live beside `vision.py` inside this directory. Each module must document its public API in this file before it is merged.

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
| `configure_identity_keywords(identities: Iterable[str]) -> None` | Populates conversation keywords (for example, “chat with assistant”) using the available identity folder names. Must be called whenever the identity roster changes so voice triggers stay in sync. |
| `detect_vision_keyword(text: str) -> Optional[KeywordMatch]` | Returns the first vision keyword matched in `text`, or `None` when no trigger is present. |
| `detect_conversation_start_keyword(text: str) -> Optional[KeywordMatch]` | Detects `chat with <identity>` requests and returns the matching keyword metadata. |
| `detect_conversation_end_keyword(text: str) -> Optional[KeywordMatch]` | Detects `goodbye <identity>` requests for the active conversation. |
| `iter_keyword_matches(text: str, keywords: Iterable[Keyword]) -> Iterator[KeywordMatch]` | Generator that yields every keyword match found in the provided text. |
| `get_vision_keyword(payload: str) -> Optional[Keyword]` | Retrieves the configured keyword metadata for `"screen"`, `"clipboard"`, or future vision payloads. |
| `get_conversation_start_keyword(payload: str) -> Optional[Keyword]` | Returns the keyword definition for a conversation-start trigger associated with the provided identity payload. |
| `get_conversation_end_keyword(payload: str) -> Optional[Keyword]` | Returns the keyword definition for a conversation-end trigger associated with the provided identity payload. |
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

SocialRobot consumes this module to decide whether the user asked for a screenshot or clipboard capture and to switch between identities when the user says “chat with assistant/default.” The CtrlSpeak transcription server inspects the same registry before forwarding speech to SocialRobot so both “chat with …” and “goodbye …” requests are handled in the parent process. This keeps shutdowns and relaunches consistent with the tray and hotkey controls. When you add new keywords, update the relevant identity system prompts (so assistants know which phrases to suggest) and refresh any UX documentation that references the trigger vocabulary.

### Keyword reference

The application responds to the following spoken or typed keywords. Each phrase is matched case-insensitively and with a fuzzy tolerance for punctuation or common speech-to-text substitutions. The clipboard trigger continues to accept near-miss variations such as “look at my slipboard” or “look at my clupboard,” and the same tolerance now applies to the conversation controls.

| Keyword | Category | Action |
| --- | --- | --- |
| `look at my screen` | Vision | Capture the user’s current desktop, store it as the identity’s latest image, and play the camera shutter sound. |
| `look at my clipboard` (including close variants like “look at my slipboard”) | Vision | Read the latest image from the system clipboard, replace the identity’s stored image, and play the camera shutter sound. |
| `update documentation` (also accepts “refresh documentation”) | Memory maintenance | Force a documentation-ingestion pass for the active identity (assistant or default), bypassing the 24-hour cooldown. |
| `update datetime` (accepts “update date time” or “refresh date time”) | Memory maintenance | Force the active identity to store the latest local date, timezone, and locale snapshot in vector memory, bypassing the 24-hour cooldown. |
| `chat with <identity>` | Conversation start | Relaunch the bot using the requested identity via the transcription server (ignored if that identity is already active). |
| `goodbye <identity>` | Conversation end | Shut down the active conversation for the specified identity from the CtrlSpeak main process. |

> **Note:** SocialRobot’s text chat window no longer treats typed “goodbye <identity>” phrases as keywords. Those messages are delivered to the bot verbatim; only spoken requests (or ones injected through the stdin control channel) trigger the shutdown helpers.

### Conversation keywords

`configure_identity_keywords()` keeps the voice trigger list synchronized with the identity folders. Once configured, the helpers recognize:

- `chat with <identity>` – immediately relaunches SocialRobot with the requested identity via the transcription server (no action is taken when the user asks for the already-active persona). Close variants like “chat was assistant” are recognised automatically.
- `goodbye <identity>` – immediately ends the current conversation and shuts the bot down from the CtrlSpeak main process before SocialRobot processes the utterance. Light punctuation (for example, “goodbye, assistant”) remains valid.

The `<identity>` placeholder uses the directory names under `third_party/social_robot/identities/`. Call `configure_identity_keywords()` whenever you add or remove identities (for example, during application startup) to keep the registry current.

The same registry now powers the push-to-talk workflow: when the user holds the right Ctrl hotkey, CtrlSpeak transcribes the utterance and checks it against these keywords before typing anything. Phrases like “chat with assistant” launch the corresponding bot immediately, “chat with default” first stops any existing session before starting the default identity, and “goodbye <identity>” routes through the same shutdown helper used by the tray menu. Because the text insertion path never runs for handled keywords, make sure any new phrases you add here have matching automation hooks so the hotkey and transcription server remain in sync with voice-triggered behaviour.

### Testing guidance

- Unit tests live in `tests/core/test_tools_keywords.py` and validate phrase detection along with payload lookups.

## Adding new tooling modules

1. Create a new module under `tools/` (for example `browser.py` or `parsing.py`).
2. Export it from `tools/__init__.py` so callers can import it consistently.
3. Document the new functionality in this file with API tables, usage examples, and testing notes.
4. Update any feature documentation (README, integration guides, UX walkthroughs) that depends on the new capability.
5. Ensure packaging scripts pick up the new module if they rely on explicit include lists.

Maintaining `docs/tooling.md` keeps CtrlSpeak’s tooling surface discoverable and prevents future refactors from duplicating functionality. Treat this file as the single source of truth for agent-facing helpers.

## Documentation ingestion helper (`background_agents/document_memory_agent.py`)

CtrlSpeak preloads Markdown documentation into vector memory so the assistant and default identities always have the latest reference material before a conversation starts. The `refresh_document_memory(identity, *, force=False, reason=None)` helper orchestrates the workflow:

- Gathers `README.md` plus the curated user-facing documents `docs/bot_integration.md`, `docs/tooling.md`, and `docs/user_flow.md`, hashing the combined content to detect changes between runs.
- Chunks each source into ~1.2 kB segments, tagging metadata with `category="documentation"`, the relative `source` path, a `chunk` counter, and the shared `doc_hash`.
- Clears any existing documentation entries in the identity’s Chroma store before inserting the freshly generated chunks so stale copies never accumulate.
- Honors each identity’s memory settings (`max_vector_items`, `vector_ttl_days`, and `pii_redaction`) and skips work entirely when vector storage is disabled.
- Records the latest refresh timestamp and `doc_hash` in `${data_root}/doc_memory/<identity>.json`, enforcing a 24-hour cooldown unless the hash changes or a forced refresh is requested.
- Emits status messages only to the terminal (never the chat UI) so operators know when the background pass runs and how many chunks landed in the store.

Use this helper when gating assistant or default start-up or responding to the `update documentation` keyword. The tracker file under AppData keeps repeated launches quick when the docs have not changed.

When a user asks for help, CtrlSpeak biases retrieval toward documentation entries. The LangGraph orchestrator lowers the similarity threshold for `category="documentation"` memories whenever the utterance resembles a “how do I…” or “help me use the app” style question (and whenever regular retrieval returns no matches). Retrieved snippets are surfaced to the model inside a system message headed `Documentation excerpts`, so the bundled personas know those passages come from the official docs and should quote them verbatim when guiding users. Each turn also prints a `[Memory]` line summarizing whether the vector store was queried and how many documentation and temporal-context chunks contributed, giving operators immediate feedback that the ingest pipeline is feeding the conversation.

The documentation embeddings (and all other vector memories) use the deterministic `ctrlspeak-minhash` embedder implemented in `utils.vector_memory.VectorMemoryStore`. The helper hashes each chunk into a 32-dimensional vector and compares them with cosine similarity against every stored entry, returning up to the configured `retrieval_top_k` items (default **5**) for each turn. This hand-rolled similarity search keeps the runtime self-contained—no external embedding model downloads are required—while still enabling LangGraph to rank results and apply category-specific fallbacks.

## Date/time context helper (`background_agents/datetime_memory_agent.py`)

Alongside documentation, CtrlSpeak primes each identity with a snapshot of the host’s current date, timezone, and locale data so the assistant can answer questions like “what day is it?” without re-querying the operating system. `refresh_datetime_memory(identity, *, force=False, reason=None)` orchestrates the workflow:

- Collects the local date (`YYYY-MM-DD` plus weekday and month names), the timezone abbreviation and UTC offset, and any locale or country code detectable from the OS environment.
- Serialises the snapshot into a single vector-memory chunk tagged with `category="temporal_context"`, `kind="current_date"`, and a `snapshot_hash` so duplicates can be detected.
- Deletes any prior temporal-context entries before inserting the new chunk, reusing the identity’s vector settings (`max_vector_items`, TTL, PII redaction) just like the documentation helper.
- Records the refresh timestamp and `snapshot_hash` under `${data_root}/datetime_memory/<identity>.json`, enforcing a 24-hour cooldown unless the data changes or a forced refresh is requested.
- Emits terminal-only status updates such as “Date/time injection complete…” while keeping the GUI silent.

The helper runs automatically for the assistant and default personas during startup, meaning a session never opens before both documentation and temporal context are current. Operators can force the refresh at any time via the `update datetime` keyword (including the fuzzy variant “update date time”) from the hotkey workflow or within an active SocialRobot session. Any user question that references the current date, time, day, or timezone automatically lowers the retrieval threshold for the temporal chunk, prompting the orchestrator to attach a `Temporal context` system message that contains the stored snapshot so the assistant can answer directly.
