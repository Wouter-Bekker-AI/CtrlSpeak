# CtrlSpeak Developer TODO

This living backlog tracks follow-up work the developer wants to schedule for CtrlSpeak. Update it whenever new ideas surface or when items are completed.

## Pending Work
- Add a "look at my cursor" / "look at my mouse" keyword that captures a 640×480 image centered on the cursor and passes it to the Vision persona.
- Integrate a Qwen3 model with tooling support, starting with Gemini CLI interactions and web search actions.
- Extend automated tests to cover additional LangGraph tool nodes once external integrations land (current coverage focuses on retrieval gating, async persistence, and embedder upgrades).
- Build a parsing tool that can read local files to gather contextual information for Vision.
- Consolidate end-user documentation into a single "CtrlSpeak user manual" once the current guides are stable so only one file needs to be injected into vector memory.
- Extend Einstein's tooling surface even further:
  - Add a guarded command runner for quick local checks (for example, `ruff` or targeted pytest subsets) so validation stays automated.
  - Perform internet searches (e.g., Bing or Brave API) with clear rate limiting and attribution so Einstein can reference current information.
  - Stage commit messages or PR summaries automatically once edits pass validation.

## Suggested Work
- Break the monolithic `main.main` startup path into smaller units to improve testability and reduce the need for integration-only coverage.
- Expand coverage for `utils.winio` text-injection fallbacks so AnyDesk/console scenarios are exercised in automated tests.
- Capture per-stage timings and status metrics from the automation flow to make workstation diagnostics easier to interpret.
- Add LangGraph trace visualizations in the management UI to help operators understand which tools fired during a session.
- Create contributor documentation that walks through adding a new Einstein tool end-to-end, including safety reviews and test expectations.

## Future Expansion
- Add an automatic update mechanism so deployed installations can receive new builds and agent definitions without manual intervention.
- Integrate licensing support by connecting CtrlSpeak to a license server that can validate seats and enforce entitlements.
- Package and distribute the application through an app store or hosted download site to simplify delivery and onboarding.
- Provide a sandboxed Python execution service (e.g., Pyodide or Firejail-backed runner) so Einstein can generate and execute short scripts safely when answering complex questions.
- Expose a user-facing tool registry that lets operators define custom Create/Add tools (with schema validation and permission scopes) that Einstein can discover at runtime.
- Add a "Send Bug Report" action that packages recent entries from `CtrlSpeak-error.log` and submits them to a configurable support endpoint.

