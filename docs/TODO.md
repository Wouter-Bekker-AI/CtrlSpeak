# CtrlSpeak Developer TODO

This living backlog tracks follow-up work the developer wants to schedule for CtrlSpeak. Update it whenever new ideas surface or when items are completed.

## Pending Work
- Add a "look at my cursor" / "look at my mouse" keyword that captures a 640×480 image centered on the cursor and passes it to the assistant agent.
- Gate cursor- and vision-related keywords so they only trigger when the active assistant has vision capabilities.
- Integrate a Qwen3 model with tooling support, starting with Gemini CLI interactions and web search actions.
- Extend automated tests to cover additional LangGraph tool nodes once external integrations land (current coverage focuses on retrieval gating, async persistence, and embedder upgrades).
- Build a parsing tool that can read local files to gather contextual information for the assistant.

## Suggested Work
- Break the monolithic `main.main` startup path into smaller units to improve testability and reduce the need for integration-only coverage.
- Expand coverage for `utils.winio` text-injection fallbacks so AnyDesk/console scenarios are exercised in automated tests.
- Capture per-stage timings and status metrics from the automation flow to make workstation diagnostics easier to interpret.

## Future Expansion
- Add an automatic update mechanism so deployed installations can receive new builds and agent definitions without manual intervention.
- Integrate licensing support by connecting CtrlSpeak to a license server that can validate seats and enforce entitlements.
- Package and distribute the application through an app store or hosted download site to simplify delivery and onboarding.

