# Tooling Reference

This document is the authoritative index for reusable tooling that ships with CtrlSpeak. Every helper that exposes capabilities to AI agents, automation flows, or user-triggered actions must be documented here so contributors know what exists, how it works, and how to extend it safely.

CtrlSpeak distinguishes between interactive tooling and the system-managed services that keep the LangGraph workflow healthy. Modules under `tools/` are invoked directly by personas or users, while the background agents in `background_agents/` are orchestrator-managed services that refresh documentation, maintain temporal context, and clean up `<think>` output as described in the [Chat with Bot speech pipeline](../README.md#chat-with-bot-speech-pipeline) and [LangGraph orchestration and persistence](bot_integration.md#langgraph-orchestration-and-persistence) guides.

The guidance below applies to the `tools/` package and any future modules that belong to it. When you add, modify, or remove a tool you **must** update this document in the same pull request.

## Directory layout

```
ctrlspeak/
├── tools/
│   ├── __init__.py
│   ├── keywords.py
│   ├── message_management.py
│   ├── vision.py
│   └── workspace.py
```

- `tools/__init__.py` exposes the modules that make up the shared tooling surface. Import helpers via `from tools import vision` (or `keywords`) so the package can evolve without breaking downstream code.
- Additional tooling (for example, browser automation or document parsing) should live beside `vision.py` inside this directory. Each module must document its public API in this file before it is merged.
- `tools/workspace.py` exposes the editing toolkit reserved for Einstein. It enforces sandboxed file access, SHA-256 locking, JSON patching, and post-edit validation.

## Workspace editing toolkit (`tools/workspace.py`)

The workspace toolkit gives Einstein deterministic access to the repository when it needs to inspect or edit files. Every function returns a JSON-serialisable dictionary that includes an `ok` flag and a list of structured errors so tool calls can be audited easily. The toolkit is sandboxed to the repository root (or a caller-supplied override via `set_workspace_root` when running tests) and refuses any path that escapes that boundary.

### Discovery and inspection

| Function | Purpose |
| --- | --- |
| `get_system_info()` | Reports the host operating system, release, Python version, and convenience booleans (`is_windows`, `is_linux`, `is_macos`) so Einstein can plan platform-aware actions before touching the workspace. |
| `search_files(query: str, glob: str \| None = None, *, limit: int = 200)` | Returns relative paths whose names match `query`. An optional `glob` (for example `"**/*.py"`) narrows the search; results are truncated to `limit` entries. |
| `list_directory(path: str, *, pattern: str \| None = None, extensions: Iterable[str] \| None = None, recursive: bool = False, limit: int \| None = 200)` | Enumerates files inside `path`, optionally filtering by glob-style patterns or file extensions. When `recursive` is `True` the search walks subdirectories; results are truncated to `limit` entries and report whether the listing was shortened. |
| `stat_file(path: str)` | Reports existence, size (in bytes), and modification time (epoch seconds) for a target file. |
| `read_file(path: str, *, mode: Literal["text","json","yaml","toml","bytes"] = "text")` | Reads a file and returns both the decoded content and its SHA-256 digest. Structured modes (`json`, `yaml`, `toml`) parse the payload so Einstein can reason about object keys before planning a patch. |
| `analyze_python(path: str)` | Parses a Python module into an AST and returns the imported modules, top-level functions, classes, and a sorted `symbols` set. Useful for planning insertions without rewriting the file blindly. |

`search_files` and the other path-based helpers accept both POSIX (`/`) and Windows (`\`) separators. The toolkit now registers the user's home directory, drive anchors, and the standard Windows AppData folders in addition to the project workspace, so Einstein can inspect requests such as `Desktop\linux commands.txt` or `%APPDATA%\CtrlSpeak\logs\ctrlspeak.log` without tripping path guards. Callers can extend or replace this allow-list at runtime when further directories must be reachable.

### Editing primitives

| Function | Purpose |
| --- | --- |
| `dry_run_json_patch(path, patch, *, expect_sha256=None, schema_path=None)` | Applies an RFC 6902 patch in memory and returns the patched object, rendered JSON text, and the new SHA-256 digest. Rejects stale hashes and honours optional JSON Schema validation when `schema_path` is provided. |
| `apply_json_patch(path, patch, *, expect_sha256=None, schema_path=None)` | Writes the JSON patch to disk atomically after performing the same checks as the dry run. Always call the dry run first so Einstein can preview the change. |
| `dry_run_text_patch(path, unified_diff, *, expect_sha256=None)` | Applies a unified diff (as produced by `difflib.unified_diff`) to UTF-8 text in memory. Verifies diff context lines match the current file and returns the patched text and new digest. |
| `apply_text_patch(path, unified_diff, *, expect_sha256=None)` | Writes the text diff atomically. On failure it surfaces the same error payload returned by the dry run. |

All mutating calls require the latest SHA-256 digest via `expect_sha256` so optimistic locking can detect concurrent edits. The helpers raise a `sha_mismatch` error code when the digest is stale, prompting Einstein to re-read the file before retrying.

### Tool routing and LLM coordination

- Every turn begins with a **probe** call to Qwen’s tool-enabled chat endpoint. The orchestrator appends `/no_think` to the user text so Qwen responds with a terse decision, and the reply is inspected only for `tool_calls`.
- When the heuristics detect a read or directory request, they create a `ToolAction` with `source="langgraph"` and mark the probe as required. The ensuing Ollama payload sets `tool_choice="required"`, forcing Qwen to emit exactly one tool call before any natural-language reply.
- If no heuristic fires, LangGraph still issues the `/no_think` probe but keeps `tool_choice="auto"` so Qwen can choose a normal answer without touching the filesystem.
- The trigger phrases that drive these heuristics live alongside `_detect_file_read_request` and `_detect_directory_listing_request` in [`utils/memory_orchestrator.py`](../utils/memory_orchestrator.py). Expect variants such as “what is in …”, “content of …”, “show me …”, “read …”, and “list all the text files on my Desktop”; updating the code keeps the phrase list and documentation in sync.
- Once Qwen returns a tool call, the orchestrator executes the helper, logs the `[Tools]` telemetry, and feeds the JSON result back as a `role="tool"` message. Qwen can chain additional tools (read → patch, list → read, etc.) before emitting the final spoken answer.
- LangGraph no longer invokes workspace tools directly. Heuristic detections simply steer the probe, while every real filesystem operation now originates from Qwen’s `tool_calls`. Trace entries continue to label the origin as `source=langgraph` or `source=llm` for observability.
- The probe and tool loop surface three workspace functions:
  - `workspace_read_file(path, location_hint?, description?)` — plan a deterministic `stat_file` → optional `search_files` → `read_file` chain.
  - `workspace_apply_text_patch(path, diff, expect_sha256?, summary?)` — supply a minimal unified diff (with the latest digest when available) so the orchestrator can dry-run and apply the change safely.
  - `workspace_list_directory(path, pattern?, extensions?, recursive?, limit?, description?)` — enumerate directory contents for queries such as “list all the text files on my Desktop”.
- Tool probes stay silent in the chat UI; only the final Qwen reply reaches the user. The terminal output records each probe payload, tool execution, and retry so operators can follow along.
- `workspace_apply_text_patch` reuses `dry_run_text_patch` and `apply_text_patch` internally. Failed dry-runs surface contextual errors (`sha_mismatch`, diff context mismatch, malformed diff headers), while successful patches report the new SHA-256 digest.
- When the user references “that file” or repeats a filename, the orchestrator shares the most recent file snapshot (path, SHA-256, truncated content) with Qwen during the probe so pronoun-driven edits stay grounded in the latest read.
- `search_files` walks every allowed root (workspace, home directory, Desktop/Documents, `%APPDATA%`, drive anchors, and any roots registered via `register_allowed_root`) so tool calls can target both repository files and host-level paths.
- While the workflow remains under observation, every Ollama request that includes a `tools` payload is echoed to the terminal (`-> Tool-enabled request payload (testing only): …`). Remove or gate this log once the integration is fully stable.

### Sandbox boundaries

| Function | Purpose |
| --- | --- |
| `register_allowed_root(path)` | Adds a new filesystem root that Einstein may inspect. The default allow-list already includes the project workspace, the active user's home directory, drive anchors, and (on Windows) `%APPDATA%`, `%LOCALAPPDATA%`, and `%USERPROFILE%`, but this helper lets you add bespoke paths such as mounted network shares. |
| `list_allowed_roots()` | Returns the complete set of active roots so operators can verify which directories are visible to the toolkit. |
| `list_additional_roots()` | Lists only the non-workspace roots that have been explicitly registered. |
| `set_additional_allowed_roots(paths)` | Replaces the additional-root allow-list. Tests use this to sandbox operations to a temporary directory before restoring the defaults. |

### Validation and formatting

| Function | Purpose |
| --- | --- |
| `validate_json(path, *, schema_path=None)` | Parses JSON and, when `schema_path` is provided, validates it with `jsonschema`. Returns `ok: True` when the payload is well-formed (and schema-compliant). |
| `validate_yaml(path)` | Parses YAML via PyYAML’s `safe_load`. Fails gracefully when PyYAML is not installed. |
| `validate_python(path, *, mode="fast" \| "strict")` | Runs `py_compile`. In `strict` mode it also invokes `ruff check --select E9,F63,F7,F82` when Ruff is available, returning a `lint_failed` error if syntax or runtime errors are detected. |
| `format_file(path)` | Formats Python sources with Black when the dependency is available. Returns `changed: False` when no formatter ran or when Black reported no changes. |

### Usage workflow

1. **Locate the target:** `search_files` and `stat_file` help Einstein confirm the file name and guard against typos.
2. **Inspect the current state:** `read_file` (optionally with `mode="json"`/`"yaml"`) or `analyze_python` builds an accurate picture before drafting edits.
3. **Plan with patches:** Generate the minimal RFC 6902 or unified diff necessary for the change. Always call the matching `dry_run_*` variant with the last known SHA-256 digest.
4. **Validate before writing:** Only call `apply_*` when the dry run succeeds. Immediately follow the write with `validate_json`/`validate_python`/`validate_yaml` as appropriate, then `format_file` to keep style consistent.
5. **Audit the response:** Every tool response includes the computed `new_sha256` so Einstein can feed it into the next mutation or confirm that no changes were needed.

When writing tests for new tooling behaviours, call `set_workspace_root(tmp_path)` to sandbox operations to the pytest temporary directory and avoid touching the real repository.

> **Python compatibility note:** The toolkit prefers the standard-library `tomllib` module introduced in Python 3.11 when parsing TOML files. On Python 3.10 and earlier, install [`tomli`](https://pypi.org/project/tomli/) so the same APIs remain available.

> **Runtime observability:** The toolkit prints `[Tools] …` status messages for every discovery and edit action (for example, `stat_file`, `search_files`, `read_file`, and path expansions such as `[Tools] Expanded candidate path …`). These logs appear in the CtrlSpeak terminal so operators can follow Einstein’s plan without the interim chatter leaking into the user-facing chat transcript.

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
| `update documentation` (also accepts “refresh documentation”) | Memory maintenance | Force a documentation-ingestion pass for the active identity (assistant, Einstein, or default), bypassing the 24-hour cooldown. |
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

CtrlSpeak preloads Markdown documentation into vector memory so the assistant, Einstein, and default identities always have the latest reference material before a conversation starts. The `refresh_document_memory(identity, *, force=False, reason=None)` helper orchestrates the workflow:

- Gathers `README.md` plus the curated user-facing documents `docs/bot_integration.md`, `docs/tooling.md`, and `docs/user_flow.md`, hashing the combined content to detect changes between runs.
- Chunks each source into ~1.2 kB segments, tagging metadata with `category="documentation"`, the relative `source` path, a `chunk` counter, and the shared `doc_hash`.
- Clears any existing documentation entries in the identity’s Chroma store before inserting the freshly generated chunks so stale copies never accumulate.
- Honors each identity’s memory settings (`max_vector_items`, `vector_ttl_days`, and `pii_redaction`) and skips work entirely when vector storage is disabled.
- Records the latest refresh timestamp and `doc_hash` in `${data_root}/doc_memory/<identity>.json`, enforcing a 24-hour cooldown unless the hash changes or a forced refresh is requested.
- Emits status messages only to the terminal (never the chat UI) so operators know when the background pass runs and how many chunks landed in the store.

Use this helper when gating assistant, Einstein, or default start-up or responding to the `update documentation` keyword. The tracker file under AppData keeps repeated launches quick when the docs have not changed.

Before any retrieval runs, the LangGraph orchestrator applies lightweight heuristics to the user text to decide which context buckets are required. The heuristics cover documentation (instruction manuals and CtrlSpeak usage notes), chat history (personal conversation memories), and temporal context (current date/time). When they trigger, the resulting plan is used immediately without calling the LLM. Only when the heuristics return an empty plan does the orchestrator ask the planner prompt to choose between `documentation`, `chat_history`, `date`, or `none`; the final decision ORs the heuristic guess with any LLM suggestion so guidance-driven requests still bias toward documentation even if the model stays silent. For Einstein, that fallback planner prompt automatically appends `/no_think` so Qwen3 returns a terse bucket selection without emitting a `<think>` block. A `none` outcome skips the vector store entirely, while any other choice constrains retrieval to the requested categories so documentation and temporal snippets are injected only when relevant. Retrieved snippets surface inside a system message headed `Documentation excerpts`, signalling to the bundled personas that those passages come from the official docs and should be quoted verbatim when guiding users. Each turn also prints a `[Memory]` line that now records the requested plan, whether the heuristics or the LLM produced it (`method=heuristic` or `method=llm`), and how many documentation and temporal-context chunks contributed, giving operators immediate feedback that the ingest pipeline is feeding the conversation.

The documentation embeddings (and all other vector memories) use the deterministic `ctrlspeak-minhash` embedder implemented in `utils.vector_memory.VectorMemoryStore`. The helper hashes each chunk into a 32-dimensional vector and compares them with cosine similarity against every stored entry, returning up to the configured `retrieval_top_k` items (default **5**) for each turn. This hand-rolled similarity search keeps the runtime self-contained—no external embedding model downloads are required—while still enabling LangGraph to rank results and apply category-specific fallbacks.

## Date/time context helper (`background_agents/datetime_memory_agent.py`)

Alongside documentation, CtrlSpeak primes each identity with a snapshot of the host’s current date, timezone, and locale data so the assistant can answer questions like “what day is it?” without re-querying the operating system. `refresh_datetime_memory(identity, *, force=False, reason=None)` orchestrates the workflow:

- Collects the local date (`YYYY-MM-DD` plus weekday and month names), the timezone abbreviation and UTC offset, and any locale or country code detectable from the OS environment.
- Serialises the snapshot into a single vector-memory chunk tagged with `category="temporal_context"`, `kind="current_date"`, and a `snapshot_hash` so duplicates can be detected.
- Deletes any prior temporal-context entries before inserting the new chunk, reusing the identity’s vector settings (`max_vector_items`, TTL, PII redaction) just like the documentation helper.
- Records the refresh timestamp and `snapshot_hash` under `${data_root}/datetime_memory/<identity>.json`, enforcing a 24-hour cooldown unless the data changes or a forced refresh is requested.
- Emits terminal-only status updates such as “Date/time injection complete…” while keeping the GUI silent.

The helper runs automatically for the assistant, Einstein, and default personas during startup, meaning a session never opens before both documentation and temporal context are current. Operators can force the refresh at any time via the `update datetime` keyword (including the fuzzy variant “update date time”) from the hotkey workflow or within an active SocialRobot session. Any user question that references the current date, time, day, or timezone automatically lowers the retrieval threshold for the temporal chunk, prompting the orchestrator to attach a `Temporal context` system message that contains the stored snapshot so the personas can answer directly.
