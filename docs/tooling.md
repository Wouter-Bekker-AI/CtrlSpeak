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
- Errors are reported through the shared logger obtained from `utils.config_paths.get_logger`, which means failures show up in `%APPDATA%\CtrlSpeak\logs\ctrlspeak.log` while callers can still recover gracefully.
- Successful captures store their files in `<memory>/screenshots/` when the caller provides a directory. Filenames follow the pattern `<prefix>_<UTC timestamp>.png`, making it easy to correlate history entries with on-disk artifacts.
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

match = keywords.detect_vision_keyword("please look at my clipboard")
if match:
    if match.keyword.payload == "clipboard":
        handle_clipboard()
    elif match.keyword.payload == "screen":
        handle_screen()
```

SocialRobot consumes this module to decide whether the user asked for a screenshot or clipboard capture, to switch between identities when the user says “chat with assistant/default,” and to close an active session when the user says “goodbye assistant/default.” When you add new keywords, update the relevant identity system prompts (so assistants know which phrases to suggest) and refresh any UX documentation that references the trigger vocabulary.

### Conversation keywords

`configure_identity_keywords()` keeps the voice trigger list synchronized with the identity folders. Once configured, the helpers recognize:

- `chat with <identity>` – immediately relaunches SocialRobot with the requested identity (no action is taken when the user asks for the already-active persona).
- `goodbye <identity>` – immediately ends the current conversation and shuts the bot down.

The `<identity>` placeholder uses the directory names under `third_party/social_robot/identities/`. Call `configure_identity_keywords()` whenever you add or remove identities (for example, during application startup) to keep the registry current.

The same registry now powers the push-to-talk workflow: when the user holds the right Ctrl hotkey, CtrlSpeak transcribes the utterance and checks it against these keywords before typing anything. Phrases like “chat with assistant” launch the corresponding bot immediately, “chat with default” first stops any existing session before starting the default identity, and “goodbye <identity>” routes through the same shutdown helper used by the tray menu. Because the text insertion path never runs for handled keywords, make sure any new phrases you add here have matching automation hooks so the hotkey remains in sync with voice-triggered behaviour.

### Testing guidance

- Unit tests live in `tests/core/test_tools_keywords.py` and validate phrase detection along with payload lookups.

## Adding new tooling modules

1. Create a new module under `tools/` (for example `browser.py` or `parsing.py`).
2. Export it from `tools/__init__.py` so callers can import it consistently.
3. Document the new functionality in this file with API tables, usage examples, and testing notes.
4. Update any feature documentation (README, integration guides, UX walkthroughs) that depends on the new capability.
5. Ensure packaging scripts pick up the new module if they rely on explicit include lists.

Maintaining `docs/tooling.md` keeps CtrlSpeak’s tooling surface discoverable and prevents future refactors from duplicating functionality. Treat this file as the single source of truth for agent-facing helpers.
