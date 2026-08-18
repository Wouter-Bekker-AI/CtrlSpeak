from __future__ import annotations

from dataclasses import replace
import sys
import types
from types import SimpleNamespace

import pytest

import tkinter as tk


# The core-headless suite intentionally installs a minimal tkinter module.
# Midnight Signal imports messagebox for the real corrections UI, so provide
# the inert namespace needed only to collect these pure layout tests.
if not hasattr(tk, "messagebox"):
    tk.messagebox = types.SimpleNamespace()  # type: ignore[attr-defined]
    sys.modules.setdefault("tkinter.messagebox", tk.messagebox)  # type: ignore[arg-type, attr-defined]
if not hasattr(tk, "ROUND"):
    tk.ROUND = "round"  # type: ignore[attr-defined]
if not hasattr(tk, "ARC"):
    tk.ARC = "arc"  # type: ignore[attr-defined]

from utils.midnight_overlay import (
    MidnightSignalOverlay,
    MonitorBounds,
    fit_overlay_text,
    overlay_geometry,
    overlay_geometry_spec,
    processing_capsule_layout,
    recording_capsule_layout,
    result_capsule_layout,
    waveform_bar_levels,
)
from utils.midnight_signal_ui import (
    MidnightSignalManagementMixin,
    MidnightTrayFlyout,
    PROVIDER_CARD_SPECS,
    ROUTING_COLUMN_WEIGHTS,
    capability_config_identity,
    language_summary,
    management_content_width,
    management_scroll_units,
    management_viewport_requires_scroll,
    provider_order_for_strategy,
    provider_signal_levels,
    strategy_choices_from_capabilities,
)
from utils import midnight_signal_ui as midnight_ui
from utils.ui_state import TranscriptionUiSession


pytestmark = pytest.mark.core_headless


class RecordingCanvas:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def _record(self, name: str, args: tuple[object, ...], kwargs: dict[str, object]) -> int:
        self.calls.append((name, args, kwargs))
        return len(self.calls)

    def create_line(self, *args, **kwargs) -> int:
        return self._record("line", args, kwargs)

    def create_arc(self, *args, **kwargs) -> int:
        return self._record("arc", args, kwargs)

    def create_oval(self, *args, **kwargs) -> int:
        return self._record("oval", args, kwargs)

    def create_text(self, *args, **kwargs) -> int:
        return self._record("text", args, kwargs)


class FakeVar:
    def __init__(self, value: object = "") -> None:
        self.value = value

    def get(self):
        return self.value

    def set(self, value: object) -> None:
        self.value = value


class FakeWidget:
    def __init__(self) -> None:
        self.options: dict[str, object] = {}

    def configure(self, **kwargs) -> None:
        self.options.update(kwargs)


class FakePackWidget(FakeWidget):
    def __init__(self, name: str, log: list[str] | None = None) -> None:
        super().__init__()
        self.name = name
        self.log = log
        self.visible = False

    def pack_forget(self) -> None:
        self.visible = False

    def pack(self, **_kwargs) -> None:
        self.visible = True
        if self.log is not None:
            self.log.append(self.name)


def _recording_snapshot():
    session = TranscriptionUiSession(clock=lambda: 10.0)
    session.begin_recording()
    return replace(
        session.snapshot(),
        elapsed_ms=4200.0,
        elapsed_label="4.2s",
        level_dbfs=-18.0,
        level_label="-18.0 dBFS",
        level_fraction=0.7,
    )


def _processing_snapshot():
    session = TranscriptionUiSession(clock=lambda: 10.0)
    session.begin_recording()
    session.begin_processing()
    return replace(session.snapshot(), elapsed_ms=1800.0, elapsed_label="1.8s")


def test_dashboard_contract_is_three_fifths_route_and_two_fifths_summary() -> None:
    assert ROUTING_COLUMN_WEIGHTS == (3, 2)
    assert [spec[0] for spec in PROVIDER_CARD_SPECS] == [
        "ubuntu-gpu-large-v3-turbo",
        "openai-gpt-transcribe",
        "gateway-tiny",
    ]
    assert all(model and location for _provider, _title, model, location in PROVIDER_CARD_SPECS)


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        (
            "ubuntu-gpu-preferred",
            ("ubuntu-gpu-large-v3-turbo", "openai-gpt-transcribe", "gateway-tiny"),
        ),
        (
            "openai-preferred",
            ("openai-gpt-transcribe", "ubuntu-gpu-large-v3-turbo", "gateway-tiny"),
        ),
        ("ubuntu-gpu-only", ("ubuntu-gpu-large-v3-turbo",)),
        ("openai-only", ("openai-gpt-transcribe",)),
        ("gateway-tiny-only", ("gateway-tiny",)),
    ],
)
def test_numbered_provider_order_matches_selected_strategy(strategy, expected) -> None:
    assert provider_order_for_strategy(strategy) == expected


def test_gateway_default_is_unnumbered_until_gateway_chain_is_known() -> None:
    assert provider_order_for_strategy("server-default") is None
    capabilities = {
        "default_strategy": "openai-preferred",
        "strategies": [
            {
                "id": "openai-preferred",
                "providers": [
                    "openai-gpt-transcribe",
                    "ubuntu-gpu-large-v3-turbo",
                    "nova-tiny-whisper",
                ],
            }
        ],
    }
    assert provider_order_for_strategy("server-default", capabilities) == (
        "openai-gpt-transcribe",
        "ubuntu-gpu-large-v3-turbo",
        "gateway-tiny",
    )


def test_gateway_default_choice_is_always_prepended_to_advertised_strategies() -> None:
    choices = strategy_choices_from_capabilities(
        {"strategies": [{"id": "ubuntu-gpu-preferred", "providers": []}]}
    )
    assert choices[0] == ("server-default", "Gateway default")
    assert choices[1][0] == "ubuntu-gpu-preferred"


@pytest.mark.parametrize(
    ("display", "expected"),
    [
        (
            "GPU preferred",
            ["ubuntu-gpu-large-v3-turbo", "openai-gpt-transcribe", "gateway-tiny"],
        ),
        (
            "OpenAI preferred",
            ["openai-gpt-transcribe", "ubuntu-gpu-large-v3-turbo", "gateway-tiny"],
        ),
        ("Ubuntu GPU only", ["ubuntu-gpu-large-v3-turbo"]),
        ("OpenAI only", ["openai-gpt-transcribe"]),
        ("Emergency tiny only", ["gateway-tiny"]),
    ],
)
def test_management_cards_rerender_and_relabel_for_every_strategy(
    monkeypatch,
    display: str,
    expected: list[str],
) -> None:
    order: list[str] = []
    painted: list[tuple[str, int | None]] = []
    manager = object.__new__(MidnightSignalManagementMixin)
    manager.ms_strategy_display_var = FakeVar(display)
    manager.provider_strategy_var = FakeVar("server-default")
    manager._capabilities = {}
    manager.ms_route_order_var = FakeVar()
    manager.ms_provider_cards = {
        provider_id: FakePackWidget(provider_id, order)
        for provider_id, _title, _model, _location in PROVIDER_CARD_SPECS
    }
    manager.ms_provider_badges = {
        provider_id: SimpleNamespace(provider_id=provider_id)
        for provider_id, _title, _model, _location in PROVIDER_CARD_SPECS
    }
    manager.ms_route_telemetry_card = FakePackWidget("telemetry", order)
    monkeypatch.setattr(
        midnight_ui,
        "_paint_route_badge",
        lambda badge, number, **_kwargs: painted.append((badge.provider_id, number)),
    )

    manager._apply_route_strategy_visuals()

    assert order[:-1] == expected
    assert order[-1] == "telemetry"
    assert [provider_id for provider_id, number in painted if number is not None] == expected
    assert [number for provider_id, number in painted if provider_id in expected] == list(
        range(1, len(expected) + 1)
    )
    assert "Selected route" in manager.ms_route_order_var.value


def test_unresolved_gateway_default_renders_all_cards_without_numbers(
    monkeypatch,
) -> None:
    order: list[str] = []
    painted: list[int | None] = []
    manager = object.__new__(MidnightSignalManagementMixin)
    manager.ms_strategy_display_var = FakeVar("Gateway default")
    manager.provider_strategy_var = FakeVar("server-default")
    manager._capabilities = {}
    manager.ms_route_order_var = FakeVar()
    manager.ms_provider_cards = {
        provider_id: FakePackWidget(provider_id, order)
        for provider_id, _title, _model, _location in PROVIDER_CARD_SPECS
    }
    manager.ms_provider_badges = {
        provider_id: object()
        for provider_id, _title, _model, _location in PROVIDER_CARD_SPECS
    }
    manager.ms_route_telemetry_card = FakePackWidget("telemetry", order)
    monkeypatch.setattr(
        midnight_ui,
        "_paint_route_badge",
        lambda _badge, number, **_kwargs: painted.append(number),
    )

    manager._apply_route_strategy_visuals()

    assert order[:-1] == [spec[0] for spec in PROVIDER_CARD_SPECS]
    assert painted == [None, None, None]
    assert "server decides" in manager.ms_route_order_var.value


def test_management_scroll_contract_covers_constrained_high_dpi_viewport() -> None:
    assert management_viewport_requires_scroll(829, 706) is True
    assert management_viewport_requires_scroll(650, 706) is False
    assert management_scroll_units(120) == -3
    assert management_scroll_units(-120) == 3
    assert management_scroll_units(0) == 0
    assert management_content_width(1094, 962) == 1094
    assert management_content_width(720, 962) == 962


def test_provider_signal_and_language_summaries_are_deterministic() -> None:
    levels = provider_signal_levels(2)
    assert levels == provider_signal_levels(2)
    assert len(levels) == 17
    assert all(0.0 <= level <= 1.0 for level in levels)
    assert levels != provider_signal_levels(3)
    assert language_summary(()) == "Automatic"
    assert language_summary(("en", "af")) == "EN · AF"


def _api_config(*, strategy: str = "ubuntu-gpu-preferred", token: str = "token-a"):
    return SimpleNamespace(
        backend="api",
        api_url="http://gateway.test:8000",
        api_token=token,
        provider_strategy=strategy,
        allowed_output_languages=("en",),
    )


def _manager_for_capability_callbacks() -> MidnightSignalManagementMixin:
    manager = object.__new__(MidnightSignalManagementMixin)
    manager.is_open = lambda: True
    manager._capability_generation = 2
    manager._capabilities = {}
    manager._provider_profiles = ()
    manager.ms_route_summary_var = FakeVar("initial")
    manager.provider_strategy_var = FakeVar("ubuntu-gpu-preferred")
    manager.ms_strategy_combo = FakeWidget()
    manager.ms_route_strategy_combo = FakeWidget()
    manager.ms_provider_signals = {}
    manager.ms_provider_vars = {
        provider_id: (FakeVar(), FakeVar(), FakeVar(), FakeVar())
        for provider_id, _title, _model, _location in PROVIDER_CARD_SPECS
    }
    manager._apply_route_strategy_visuals = lambda: None
    return manager


def test_out_of_order_capability_success_and_error_cannot_overwrite_new_state(
    monkeypatch,
) -> None:
    config = _api_config()
    identity = capability_config_identity(config)
    monkeypatch.setattr(midnight_ui, "get_backend_config", lambda: config)
    monkeypatch.setattr(
        midnight_ui.sysmod,
        "transcription_ui_session",
        SimpleNamespace(snapshot=lambda: SimpleNamespace(provider=None)),
    )
    manager = _manager_for_capability_callbacks()
    current = {
        "version": "new",
        "providers": [],
        "strategies": [{"id": "ubuntu-gpu-preferred", "providers": []}],
    }
    stale = {"version": "old", "providers": []}

    manager._finish_capabilities(current, 21.0, 2, identity)
    assert "Gateway new" in manager.ms_route_summary_var.value
    assert manager.ms_strategy_combo.options["values"][0] == "Gateway default"

    manager._finish_capabilities(stale, 999.0, 1, identity)
    manager._finish_capabilities_error(1, identity)
    assert "Gateway new" in manager.ms_route_summary_var.value

    manager._finish_capabilities_error(2, identity)
    assert manager._capabilities == {}
    assert manager._provider_profiles == ()
    assert "unavailable" in manager.ms_route_summary_var.value.casefold()


def test_backend_save_invalidates_capability_chain_before_rerender(
    monkeypatch,
) -> None:
    observations: list[dict[str, object]] = []
    manager = object.__new__(MidnightSignalManagementMixin)
    manager.provider_strategy_var = FakeVar()
    manager.ms_strategy_display_var = FakeVar("Gateway default")
    manager._apply_backend = lambda: None
    manager._capability_generation = 0
    manager._corrections_generation = 0
    manager._correction_mutation_generation = 0
    manager._correction_mutations_pending = 0
    manager._correction_mutation_errors = []
    manager._correction_refresh_notice = None
    manager._corrections_loading = False
    manager._capabilities = {"default_strategy": "openai-preferred"}
    manager._provider_profiles = (object(),)
    manager.ms_backend_status_var = FakeVar()
    manager.backend_status_var = FakeVar("saved")
    manager._apply_route_strategy_visuals = lambda: observations.append(
        dict(manager._capabilities)
    )
    manager.window = SimpleNamespace(after=lambda *_args: None)
    monkeypatch.setattr(midnight_ui, "get_backend_config", lambda: _api_config())

    manager._save_backend_midnight()

    assert observations == [{}]
    assert manager._provider_profiles == ()


class ImmediateThread:
    def __init__(self, *, target, **_kwargs) -> None:
        self.target = target

    def start(self) -> None:
        self.target()


class NoAfterWindow:
    def after(self, *_args, **_kwargs):
        raise AssertionError("worker must not call Tk/window.after")


@pytest.mark.parametrize("failure", [False, True])
def test_correction_list_worker_queues_captured_success_or_error(
    monkeypatch,
    failure: bool,
) -> None:
    queued: list[tuple[object, tuple[object, ...]]] = []
    monkeypatch.setattr(midnight_ui.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(
        midnight_ui.sysmod,
        "enqueue_management_task",
        lambda callback, *args: queued.append((callback, args)),
    )
    manager = object.__new__(MidnightSignalManagementMixin)
    manager.is_open = lambda: True
    manager.window = NoAfterWindow()
    manager._corrections_loading = False
    manager._corrections_generation = 0
    manager._correction_rules = []
    manager.ms_correction_status_var = FakeVar()
    manager._render_corrections = lambda: None

    class Client:
        @staticmethod
        def list_corrections():
            if failure:
                raise RuntimeError("list failed")
            return [{"id": "rule-1", "source_phrase": "Innova", "replacement_phrase": "Nova", "enabled": True}]

    manager._correction_client = lambda: Client()
    manager._refresh_corrections()
    assert len(queued) == 1
    callback, args = queued.pop()
    callback(*args)
    if failure:
        assert "list failed" in str(manager.ms_correction_status_var.value)
    else:
        assert manager._correction_rules[0]["id"] == "rule-1"


@pytest.mark.parametrize("failure", [False, True])
def test_correction_mutation_worker_queues_captured_success_or_error(
    monkeypatch,
    failure: bool,
) -> None:
    queued: list[tuple[object, tuple[object, ...]]] = []
    refreshed: list[bool] = []
    monkeypatch.setattr(midnight_ui.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(
        midnight_ui.sysmod,
        "enqueue_management_task",
        lambda callback, *args: queued.append((callback, args)),
    )
    manager = object.__new__(MidnightSignalManagementMixin)
    manager.is_open = lambda: True
    manager.window = NoAfterWindow()
    manager._corrections_loading = False
    manager._corrections_generation = 0
    manager._correction_mutation_generation = 0
    manager._correction_mutations_pending = 0
    manager._correction_mutation_errors = []
    manager._correction_refresh_notice = None
    manager.ms_correction_status_var = FakeVar()
    manager._correction_client = lambda: object()
    manager._refresh_corrections = lambda: refreshed.append(True)

    def action(_client) -> None:
        if failure:
            raise RuntimeError("mutation failed")

    manager._run_correction_mutation("Saving correction", action)
    while queued:
        callback, args = queued.pop(0)
        callback(*args)
    assert refreshed == [True]
    if failure:
        assert "mutation failed" in str(manager._correction_refresh_notice)
    else:
        assert manager._correction_refresh_notice is None


def test_correction_mutations_are_fifo_and_refresh_only_after_all_settle(
    monkeypatch,
) -> None:
    queued: list[tuple[object, tuple[object, ...]]] = []
    committed: list[str] = []
    refreshed: list[tuple[str, ...]] = []
    monkeypatch.setattr(midnight_ui.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(
        midnight_ui.sysmod,
        "enqueue_management_task",
        lambda callback, *args: queued.append((callback, args)),
    )
    manager = object.__new__(MidnightSignalManagementMixin)
    manager.is_open = lambda: True
    manager._corrections_loading = False
    manager._corrections_generation = 0
    manager._correction_mutation_generation = 0
    manager._correction_mutations_pending = 0
    manager._correction_mutation_errors = []
    manager._correction_refresh_notice = None
    manager.ms_correction_status_var = FakeVar()
    manager._correction_client = lambda: object()
    manager._refresh_corrections = lambda: refreshed.append(tuple(committed))

    manager._run_correction_mutation(
        "First", lambda _client: committed.append("first")
    )
    # The first worker has observed Empty but its idle callback has not yet run;
    # this reproduces the enqueue/worker-exit race deterministically.
    manager._run_correction_mutation(
        "Second", lambda _client: committed.append("second")
    )
    while queued:
        callback, args = queued.pop(0)
        callback(*args)

    assert committed == ["first", "second"]
    assert refreshed == [("first", "second")]


@pytest.mark.parametrize(
    ("capability_status", "has_key", "expected"),
    [
        ("available_with_key", False, "Available with key · key required"),
        ("available_with_key", True, "Available with key · configured"),
        ("unavailable", True, "Unavailable · configured"),
    ],
)
def test_flyout_openai_copy_preserves_capability_truth_and_key_state(
    monkeypatch,
    capability_status: str,
    has_key: bool,
    expected: str,
) -> None:
    config = _api_config()
    identity = capability_config_identity(config)
    monkeypatch.setattr(midnight_ui, "get_runtime_backend_config", lambda: config)
    monkeypatch.setattr(
        midnight_ui,
        "get_session_openai_api_key",
        lambda: "configured-key" if has_key else None,
    )
    flyout = object.__new__(MidnightTrayFlyout)
    flyout.is_open = lambda: True
    flyout._capability_generation = 1
    flyout._capabilities = {}
    flyout._provider_rows = {"openai-gpt-transcribe": (FakeVar(), FakeVar())}
    flyout._apply_route_strategy_visuals = lambda: None
    payload = {
        "providers": [
            {
                "id": "openai-gpt-transcribe",
                "status": capability_status,
                "model": "gpt-4o-transcribe",
            }
        ]
    }
    flyout._apply_capabilities(payload, 25.0, 1, identity)
    assert flyout._provider_rows["openai-gpt-transcribe"][0].value == expected
    assert "Ready" not in expected


def test_flyout_rejects_out_of_order_capability_success_and_error(
    monkeypatch,
) -> None:
    config = _api_config()
    identity = capability_config_identity(config)
    monkeypatch.setattr(midnight_ui, "get_runtime_backend_config", lambda: config)
    monkeypatch.setattr(midnight_ui, "get_session_openai_api_key", lambda: None)
    flyout = object.__new__(MidnightTrayFlyout)
    flyout.is_open = lambda: True
    flyout._capability_generation = 2
    flyout._capabilities = {}
    flyout._provider_rows = {
        "openai-gpt-transcribe": (FakeVar("initial"), FakeVar("initial"))
    }
    flyout._apply_route_strategy_visuals = lambda: None

    flyout._apply_capabilities(
        {
            "version": "new",
            "providers": [
                {
                    "id": "openai-gpt-transcribe",
                    "status": "available_with_key",
                }
            ],
        },
        20.0,
        2,
        identity,
    )
    state_var, timing_var = flyout._provider_rows["openai-gpt-transcribe"]
    assert state_var.value == "Available with key · key required"
    assert timing_var.value == "Gateway 20 ms"

    flyout._apply_capabilities(
        {
            "version": "stale",
            "providers": [
                {"id": "openai-gpt-transcribe", "status": "unavailable"}
            ],
        },
        999.0,
        1,
        identity,
    )
    flyout._apply_capabilities_error(1, identity)
    assert state_var.value == "Available with key · key required"
    assert timing_var.value == "Gateway 20 ms"

    flyout._apply_capabilities_error(2, identity)
    assert flyout._capabilities == {}
    assert state_var.value == "Unavailable · key required"
    assert timing_var.value == "—"


def test_flyout_success_resets_rows_omitted_by_gateway(monkeypatch) -> None:
    config = _api_config()
    identity = capability_config_identity(config)
    monkeypatch.setattr(midnight_ui, "get_runtime_backend_config", lambda: config)
    monkeypatch.setattr(midnight_ui, "get_session_openai_api_key", lambda: "key")
    flyout = object.__new__(MidnightTrayFlyout)
    flyout.is_open = lambda: True
    flyout._capability_generation = 1
    flyout._capabilities = {"version": "old"}
    flyout._provider_rows = {
        "ubuntu-gpu-large-v3-turbo": (FakeVar("Ready"), FakeVar("Probe 8 ms")),
        "openai-gpt-transcribe": (FakeVar("Ready"), FakeVar("Gateway 10 ms")),
        "gateway-tiny": (FakeVar("Ready"), FakeVar("Gateway 10 ms")),
    }
    flyout._apply_route_strategy_visuals = lambda: None

    flyout._apply_capabilities(
        {
            "version": "new",
            "providers": [
                {
                    "id": "openai-gpt-transcribe",
                    "status": "available_with_key",
                }
            ],
        },
        15.0,
        1,
        identity,
    )

    assert flyout._provider_rows["ubuntu-gpu-large-v3-turbo"][0].value == "Not advertised"
    assert flyout._provider_rows["ubuntu-gpu-large-v3-turbo"][1].value == "—"
    assert flyout._provider_rows["gateway-tiny"][0].value == "Not advertised"
    assert flyout._provider_rows["gateway-tiny"][1].value == "—"


def test_recording_capsule_has_ordered_regions_and_real_ten_bar_waveform() -> None:
    layout = recording_capsule_layout(548, 88)
    assert (
        layout.mic_x
        < layout.divider_one_x
        < layout.timer_x
        < layout.divider_two_x
        < layout.waveform_left
        < layout.waveform_right
        < layout.status_x
        < layout.dbfs_x
    )
    samples = [index / 99 for index in range(100)]
    bars = waveform_bar_levels(samples)
    assert len(bars) == 10
    assert all(0.08 <= level <= 1.0 for level in bars)
    assert len(set(round(level, 3) for level in bars)) > 1

    overlay = object.__new__(MidnightSignalOverlay)
    overlay.waveform_provider = lambda: samples
    overlay.device_label_provider = lambda: "System default"
    canvas = RecordingCanvas()
    overlay._draw_recording(canvas, 548, 88, _recording_snapshot())

    waveform_lines = [
        call
        for call in canvas.calls
        if call[0] == "line" and "waveform" in call[2].get("tags", ())
    ]
    assert len(waveform_lines) == 10
    text = [call[2].get("text") for call in canvas.calls if call[0] == "text"]
    assert {"4.2s", "LISTENING", "Release Right Ctrl to transcribe", "-18.0 dBFS"} <= set(text)


def test_processing_capsule_uses_long_double_orbit_and_never_a_provider_orbit(
    monkeypatch,
) -> None:
    layout = processing_capsule_layout(548, 88)
    assert layout.orbit_rx / layout.orbit_ry > 3.0
    assert layout.divider_x < layout.orbit_cx < layout.status_x < layout.elapsed_x

    overlay = object.__new__(MidnightSignalOverlay)
    overlay.reduced_motion = True
    overlay._started = 0.0
    monkeypatch.setattr(
        MidnightSignalOverlay,
        "_processing_strategy_label",
        staticmethod(lambda _state: "GPU preferred"),
    )
    canvas = RecordingCanvas()
    overlay._draw_processing(canvas, 548, 88, _processing_snapshot())

    orbit_ovals = [
        call
        for call in canvas.calls
        if call[0] == "oval" and "processing-orbit" in call[2].get("tags", ())
    ]
    assert len(orbit_ovals) == 2
    for _kind, coordinates, _options in orbit_ovals:
        x1, y1, x2, y2 = (float(value) for value in coordinates)
        assert (x2 - x1) / (y2 - y1) > 3.0
    assert any(
        "processing-comet" in call[2].get("tags", ()) for call in canvas.calls
    )
    text = [str(call[2].get("text")) for call in canvas.calls if call[0] == "text"]
    assert {"TRANSCRIBING", "GPU preferred", "1.8s", "Cancel from tray"} <= set(text)
    assert not any(name in " ".join(text) for name in ("Ubuntu GPU", "OpenAI", "Gateway Tiny"))


def test_overlay_geometry_is_bounded_and_selects_high_dpi_compact_layout() -> None:
    nominal_bounds = MonitorBounds(0, 0, 1920, 1080)
    nominal = overlay_geometry(nominal_bounds, 1.0)
    assert (nominal.width, nominal.height, nominal.compact) == (548, 88, False)
    assert nominal_bounds.left <= nominal.x
    assert nominal.x + nominal.width <= nominal_bounds.right
    assert nominal_bounds.top <= nominal.y
    assert nominal.y + nominal.height <= nominal_bounds.bottom

    constrained_bounds = MonitorBounds(100, 50, 900, 650)
    constrained = overlay_geometry(constrained_bounds, 2.0)
    assert constrained.compact is True
    assert constrained.width < round(548 * 2.0)
    assert constrained_bounds.left <= constrained.x
    assert constrained.x + constrained.width <= constrained_bounds.right
    assert constrained_bounds.top <= constrained.y
    assert constrained.y + constrained.height <= constrained_bounds.bottom

    short_bounds = MonitorBounds(-300, -20, 0, 40)
    short = overlay_geometry(short_bounds, 3.0)
    assert short.height == short_bounds.height
    assert short_bounds.left <= short.x <= short_bounds.right - short.width
    assert short_bounds.top <= short.y <= short_bounds.bottom - short.height
    assert "+-" not in overlay_geometry_spec(short)
    assert overlay_geometry_spec(short).endswith(f"{short.x:+d}{short.y:+d}")


def test_compact_recording_and_processing_keep_required_truthful_copy(
    monkeypatch,
) -> None:
    overlay = object.__new__(MidnightSignalOverlay)
    overlay.waveform_provider = lambda: [0.2] * 100
    overlay.device_label_provider = lambda: None
    overlay.reduced_motion = True
    overlay._started = 0.0
    monkeypatch.setattr(
        MidnightSignalOverlay,
        "_processing_strategy_label",
        staticmethod(lambda _state: "GPU preferred"),
    )

    recording_canvas = RecordingCanvas()
    overlay._draw_recording(recording_canvas, 360, 88, _recording_snapshot())
    recording_text = " ".join(
        str(call[2].get("text"))
        for call in recording_canvas.calls
        if call[0] == "text"
    )
    assert "LISTENING" in recording_text
    assert "release Right Ctrl" in recording_text
    assert "4.2s" in recording_text
    assert "-18.0 dBFS" in recording_text
    assert len(
        [
            call
            for call in recording_canvas.calls
            if call[0] == "line" and "waveform" in call[2].get("tags", ())
        ]
    ) == 10

    processing_canvas = RecordingCanvas()
    overlay._draw_processing(processing_canvas, 360, 88, _processing_snapshot())
    processing_text = " ".join(
        str(call[2].get("text"))
        for call in processing_canvas.calls
        if call[0] == "text"
    )
    assert "TRANSCRIBING" in processing_text
    assert "GPU preferred" in processing_text
    assert "1.8s" in processing_text
    assert not any(
        provider in processing_text
        for provider in ("Ubuntu GPU", "OpenAI", "Gateway Tiny")
    )


@pytest.mark.parametrize("phase", ["success", "error", "cancelled"])
@pytest.mark.parametrize("width", [548, 360])
def test_terminal_detail_never_enters_reserved_elapsed_column(
    phase: str,
    width: int,
) -> None:
    long_detail = (
        "This complete safe detail remains available in the tray and control "
        "centre while the overlay renders a compact single line"
    )
    session = TranscriptionUiSession(clock=lambda: 10.0)
    session.begin_recording()
    if phase != "cancelled":
        session.begin_processing()
    if phase == "success":
        session.complete(elapsed_ms=4321.0)
    elif phase == "error":
        session.fail("providers_exhausted", elapsed_ms=4321.0)
    else:
        session.cancel()
    snapshot = replace(session.snapshot(), detail=long_detail, elapsed_label="4.3s")

    overlay = object.__new__(MidnightSignalOverlay)
    canvas = RecordingCanvas()
    if phase == "cancelled":
        overlay._draw_cancelled(canvas, width, 88, snapshot)
    else:
        overlay._draw_result(
            canvas, width, 88, snapshot, success=(phase == "success")
        )

    layout = result_capsule_layout(width, 88)
    assert layout.detail_right < layout.elapsed_left
    detail_calls = [
        call
        for call in canvas.calls
        if call[0] == "text" and "result-detail" in call[2].get("tags", ())
    ]
    assert len(detail_calls) == 1
    _kind, coordinates, options = detail_calls[0]
    detail_x = float(coordinates[0])
    rendered_width = float(options["width"])
    assert detail_x + rendered_width <= layout.detail_right
    assert detail_x + rendered_width < layout.elapsed_left
    assert options["text"] != long_detail
    assert str(options["text"]).endswith("…")
    assert any(
        call[0] == "text" and call[2].get("text") == "4.3s"
        for call in canvas.calls
    )
    # Rendering is intentionally non-destructive; the complete safe copy stays
    # in the immutable snapshot consumed by the tray/control centre.
    assert snapshot.detail == long_detail


def test_overlay_text_fitting_is_deterministic_and_single_line() -> None:
    original = "A very long\nresult detail with    repeated whitespace"
    first = fit_overlay_text(original, pixel_width=82)
    assert first == fit_overlay_text(original, pixel_width=82)
    assert "\n" not in first
    assert "  " not in first
    assert first.endswith("…")
