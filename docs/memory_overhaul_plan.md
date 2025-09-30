# Memory Orchestration Roadmap

This roadmap captures the staged rebuild of CtrlSpeak's long-term memory system. The effort is explicitly migration-free: new installations and identities start clean in platform AppData directories, and any legacy `memory/` assets in the repository tree are ignored. All acceptance criteria outlined below are now implemented; the document remains as a reference for the architecture and accompanying tests.

---

## Part 1 – Platform Foundations *(current work)*

### Goals
- Establish platform-aware helpers for `data_root`, `config_root`, and `logs_root`.
- Guarantee global conventions around atomic writes, per-identity exclusivity, and JSONL rotation.
- Ensure every runtime artifact (models, CUDA runtimes, automation outputs, vector stores, screenshots, logs, traces) resolves through the data root while settings remain in the config root.

### Acceptance Criteria
- **Windows:** runtime data and config resolve under `%APPDATA%\CtrlSpeak`.
- **Linux/macOS:** `data_root = ${XDG_DATA_HOME:-~/.local/share}/CtrlSpeak`, `config_root = ${XDG_CONFIG_HOME:-~/.config}/CtrlSpeak`.
- `logs_root = ${data_root}/logs` with rotating handlers wired to that directory.
- Helper module eagerly creates `models/`, `cuda/`, `bot_memory/`, `logs/`, and `temp/` under the data root.
- Temporary files used for atomic writes are created inside the target directory and promoted with `os.replace`.

### Implementation Steps
1. Extend `utils/config_paths.py` with `get_data_dir()`, `get_config_dir()`, and `get_logs_dir()` based on the platform rules above; ensure the data root pre-creates required subdirectories.
2. Update runtime helpers (models, CUDA caches, automation artifacts, logging configuration, temporary recordings, screenshots, traces, future vector stores) to source paths from `get_data_dir()` and `get_logs_dir()`.
3. Keep `settings.json` and identity overrides in the config root, guarded by a module-level settings lock.
4. Add convenience wrappers for common subpaths (e.g., `get_bot_memory_dir(identity)` in Part 2) that delegate to the new helpers.

### Test Coverage
- Environment matrix where `XDG_DATA_HOME`/`XDG_CONFIG_HOME` are unset/set to confirm directories resolve correctly.
- Regression guard that nothing writes under `~/.config` except config artifacts.
- CLI single-instance lock tests confirm lock files land under the data root.

---

## Part 2 – Identity Storage & Concurrency *(next slice)*

### 2.1 SocialRobot Memory Relocation (Fresh Start)
- Identities persist under `${data_root}/bot_memory/<identity>/` with `conversation/`, `screenshots/`, `chroma/`, and `traces/` subfolders.
- Update integration helpers and GUI actions to operate solely on AppData-backed paths; packaged builds never write into the repository tree.

### 2.2 Atomic File Operations
- Introduce `utils/io_atomic.py` with `atomic_write_text` and `atomic_rotate` helpers.
- Use the helpers for settings and conversation persistence, ensuring temp files live beside targets and are removed on failure.

### 2.3 Cross-Process Identity Locks
- Implement `${data_root}/.locks/<identity>.lock` via `portalocker` (Windows + POSIX).
- Parent acquires the lock before spawning SocialRobot or LangGraph; the child fails fast with “identity in use” if the lock is held.
- Document the user-facing error and release semantics (including crash recovery).

### Tests
- Launching a bot creates the full directory tree in `data_root` and “Clear memory” touches only AppData paths.
- Simulated disk-full/permission errors leave no partial files.
- Multiprocess contention test verifies the second process exits with the documented error and locks are cleared after abnormal termination.

---

## Part 3 – Orchestrator, Vector Memory & Observability *(final slice)*

### 3.1 LangGraph Orchestrator (Stream-Safe Defaults)
- Graph nodes cover `retrieve → plan_tools → call_tools → llm → persist`.
- Retrieval skips when the store is empty or similarity falls below the configured threshold (default cosine ≥ 0.75, top-k = 5).
- Embedding/upsert work runs asynchronously so TTS streaming is never blocked.
- Feature flag `use_langgraph_memory_orchestrator` defaults to `false` for gradual rollout.

### 3.2 Chroma Lifecycle & Retention
- Chroma client lives under `${bot_memory}/chroma`, sharing the identity lock to enforce single-writer semantics.
- Embedder metadata drives `_vN` collections when versions change; maintenance hooks expose compact/vacuum and drop-and-recreate flows via CLI/UI.
- Vector store caps at 5 000 items with LRU eviction and optional per-document TTL.

### 3.3 Privacy, Observability & Health
- Conversation logs rotate at 10 MB with five files retained (JSONL format).
- Optional regex-based PII redaction (emails, phone numbers, ID-like tokens) runs before embedding; toggled per identity.
- LangGraph nodes emit structured traces with a `correlation_id`, retry budget (one for vector upserts), and counters for retrieval hits, average similarity, persist latency, eviction count, and lock wait durations.
- `ctrlspeak health` command verifies data-root writability, lock acquisition, Chroma initialization, and embedder metadata alignment.

### Tests
- Empty-store retrieval and similarity gating behave as expected while async persistence keeps TTS streaming responsive.
- Embedder upgrades yield new `_vN` collections with old data readable.
- Health diagnostics report actionable failures with a stable JSON schema.
- Documentation reflects retention/privacy defaults, locking behavior, platform paths, and the health command workflow.

---

Deliverables for Parts 2 and 3 remain unchanged from the original roadmap; they will be executed in order after Part 1 lands.
