import ctypes
import json
import sys
import subprocess
import textwrap
from types import SimpleNamespace

import pytest

from utils.dictation_text import prepare_dictation_text, control_character_counts
from utils import config_paths, system
from utils.transcription_backend import TranscriptionResult, save_backend_config, get_backend_config

pytestmark = pytest.mark.core_headless


@pytest.mark.parametrize('text,expected', [
    ('First.\r\n\r\nSecond.\n', 'First. Second.'),
    ('a\rb\tc\vd\fe\x85f\u2028g\u2029h', 'a b c d e f g h'),
    ('\x00\x03\x08hello\x1b\x7f\x9b', 'hello'),
    ('café 中文 🌱 👩\u200d💻 e\u0301', 'café 中文 🌱 👩\u200d💻 e\u0301'),
    ('hello\ud800\u202eevil\u202c', 'helloevil'),
    ('literal \\n is not a newline', 'literal \\n is not a newline'),
])
def test_safe_single_line(text, expected):
    assert prepare_dictation_text(text) == expected
    assert control_character_counts(prepare_dictation_text(text)) == {}


def test_formatting_retains_only_internal_breaks_not_tabs_or_final_submit():
    assert prepare_dictation_text('\nFirst.\r\n\r\n\tSecond.\x1b\x08\n', preserve_formatting=True) == 'First.\n\n Second.'
    assert prepare_dictation_text('\r\n\t\x1b', preserve_formatting=True) == ''


def test_all_control_characters_removed_in_safe_mode_and_policy_is_idempotent():
    text = 'A' + ''.join(map(chr, list(range(32)) + list(range(127, 160)))) + 'B'
    for preserve in (False, True):
        result = prepare_dictation_text(text, preserve_formatting=preserve)
        assert set(control_character_counts(result)) <= ({'U+000A'} if preserve else set())
        assert prepare_dictation_text(result, preserve_formatting=preserve) == result


def test_formatting_setting_default_roundtrip_and_validation():
    assert get_backend_config({}).gpu_cleanup_preserve_formatting is False
    args = dict(backend='api', api_url='http://gateway', api_token=None, feedback_capture_method='disabled')
    assert save_backend_config(**args, gpu_cleanup_enabled=True, gpu_cleanup_preserve_formatting=True).gpu_cleanup_preserve_formatting
    assert save_backend_config(**args).gpu_cleanup_preserve_formatting
    assert get_backend_config({}).gpu_cleanup_preserve_formatting
    with pytest.raises(ValueError):
        save_backend_config(**args, gpu_cleanup_preserve_formatting='true')


@pytest.mark.parametrize('enabled,formatting,applied,preserve', [
    (True, False, True, False), (True, True, True, True),
    (False, True, True, False), (True, True, False, False),
])
def test_injection_audits_and_feedback_use_delivered_text(tmp_path, monkeypatch, enabled, formatting, applied, preserve):
    config_paths.settings.update(gpu_cleanup_enabled=enabled, gpu_cleanup_preserve_formatting=formatting)
    result = TranscriptionResult(text='First.\n\nSecond.\r\n\x1b', transcription_id='tx', raw_text='first second',
        corrected_text='first second', metadata={'normalized_text': 'First.\n\nSecond.\r\n\x1b',
            'client_cleanup_applied': applied, 'Authorization': 'never-log-secret'})
    calls, tracked = [], []
    monkeypatch.setattr(system, 'insert_text_into_focus', lambda text, **kw: calls.append((text, kw)))
    monkeypatch.setattr(system, 'track_feedback_injection', tracked.append)
    system.inject_transcription_result(result)
    expected = 'First.\n\nSecond.' if preserve else 'First. Second.'
    assert calls == [(expected, {'preserve_formatting': True} if preserve else {})]
    assert tracked[0].text == expected
    assert system.get_last_transcript() == expected
    rawlog = (config_paths.get_logs_dir() / 'dictation-audit.jsonl').read_text()
    assert 'never-log-secret' not in rawlog
    records = [json.loads(line) for line in rawlog.splitlines()]
    assert [r['outcome'] for r in records] == ['attempted', 'adapter_returned']
    assert records[0]['attempt_id'] == records[1]['attempt_id']
    assert records[1]['insertion_text'] == expected
    assert records[1]['selected_text'] == result.text


def test_failed_insertion_is_not_logged_as_success(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError('adapter failed')
    monkeypatch.setattr(system, 'insert_text_into_focus', fail)
    monkeypatch.setattr(system, 'track_feedback_injection', lambda _: pytest.fail('must not track failed insertion'))
    with pytest.raises(RuntimeError):
        system.inject_transcription_result(TranscriptionResult(text='hi'))
    records = [json.loads(line) for line in (config_paths.get_logs_dir() / 'dictation-audit.jsonl').read_text().splitlines()]
    assert records[-1]['outcome'] == 'failed_or_partial'


def test_audit_rotation_is_bounded_and_logging_failure_does_not_block(monkeypatch):
    from utils import transcript_audit
    directory = config_paths.get_logs_dir()
    audit = directory / 'dictation-audit.jsonl'
    result = TranscriptionResult(text='safe')
    for _ in range(4):
        audit.write_text('x' * (2 * 1024 * 1024))
        transcript_audit.record_injection(result, 'safe', preserve_formatting=False,
                                          attempt_id='test', outcome='adapter_returned')
    assert sorted(p.name for p in directory.glob('dictation-audit.jsonl*')) == [
        'dictation-audit.jsonl', 'dictation-audit.jsonl.1', 'dictation-audit.jsonl.2']
    def fail():
        raise OSError('no disk')
    monkeypatch.setattr(transcript_audit, 'get_logs_dir', fail)
    transcript_audit.record_injection(result, 'safe', preserve_formatting=False,
                                      attempt_id='test', outcome='attempted')


@pytest.mark.skipif(not sys.platform.startswith('win'), reason='Win32 SendInput structure')
@pytest.mark.parametrize('preserve', [False, True])
def test_windows_sendinput_exact_events_no_hidden_enter_or_truncated_unicode(monkeypatch, preserve):
    from utils import windows_input as win
    events = []
    def send(count, pointer, size):
        key = ctypes.cast(pointer, ctypes.POINTER(win.INPUT)).contents.union.ki
        events.append((key.wVk, key.wScan, key.dwFlags))
        return 1
    monkeypatch.setattr(win, '_send_input', send)
    monkeypatch.setattr(win, 'get_focused_control', lambda: 1)
    monkeypatch.setattr(win, 'ensure_foreground', lambda _: None)
    monkeypatch.setattr(win.time, 'sleep', lambda _: None)
    # U+1000D used to truncate to CR in a 16-bit wScan.
    text = 'A\n\nB\t\x1b\x08🌱\U0001000d\n'
    assert win.send_unicode_input(text, preserve_formatting=preserve)
    expected = prepare_dictation_text(text, preserve_formatting=preserve).replace('\n', '\r').encode('utf-16-le')
    units = [int.from_bytes(expected[i:i+2], 'little') for i in range(0, len(expected), 2)]
    assert events == [event for unit in units for event in [(0, unit, win.KEYEVENTF_UNICODE), (0, unit, win.KEYEVENTF_UNICODE | win.KEYEVENTF_KEYUP)]]
    assert not any(vk in (0x0d, 0x09, 0x08, 0x1b) for vk, _, _ in events)
    if not preserve:
        assert not any(scan < 32 or 127 <= scan < 160 for _, scan, _ in events)
    assert events[-1][1] != 13


@pytest.mark.skipif(not sys.platform.startswith('win'), reason='Win32 paths')
@pytest.mark.parametrize('remote,direct_success,send_success', [(True, False, True), (True, False, False), (False, True, False), (False, False, False)])
def test_windows_all_insertion_routes_receive_sanitized_text(monkeypatch, remote, direct_success, send_success):
    from utils import windows_input as win
    received = []
    monkeypatch.setattr(win, 'get_focused_control', lambda: 1)
    monkeypatch.setattr(win, 'is_console_window', lambda _: False)
    monkeypatch.setattr(win, 'is_remote_host_window', lambda _: remote)
    monkeypatch.setattr(win, 'is_anydesk_window', lambda _: remote)
    monkeypatch.setattr(win, 'send_unicode_input', lambda text, **kw: received.append(text) or send_success)
    monkeypatch.setattr(win, 'try_direct_text_insert', lambda text, hwnd: received.append(text) or direct_success)
    for name in ('try_sendinput_paste', 'try_clipboard_paste'):
        monkeypatch.setattr(win, name, lambda text: received.append(text) or False)
    monkeypatch.setattr(win.pyautogui, 'write', lambda text: received.append(text), raising=False)
    win.insert_text_into_focus('First.\nSecond.\t\x1b\r')
    assert received and set(received) == {'First. Second.'}


@pytest.mark.parametrize('preserve', [False, True])
def test_linux_clipboard_sanitized_without_submit(monkeypatch, preserve):
    from utils import linux_input as lin
    calls = []
    monkeypatch.setattr(lin, 'ensure_desktop_automation_supported', lambda: None)
    monkeypatch.setattr(lin, 'clipboard_tool_available', lambda: True)
    monkeypatch.setattr(lin, 'clipboard_contains_non_text_data', lambda: False)
    monkeypatch.setattr(lin, 'get_clipboard_text', lambda: 'prior')
    monkeypatch.setattr(lin, 'set_clipboard_text', lambda text: calls.append(('paste', text)) or True)
    monkeypatch.setattr(lin, 'restore_clipboard_text', lambda _: None)
    monkeypatch.setattr(lin, 'get_pyautogui', lambda: SimpleNamespace(hotkey=lambda *keys: calls.append(keys)))
    monkeypatch.setattr(lin.time, 'sleep', lambda _: None)
    lin.insert_text_into_focus('a\nb\t\x03\n', preserve_formatting=preserve)
    assert calls == [('paste', 'a\nb' if preserve else 'a b'), ('ctrl', 'v')]


@pytest.mark.skipif(not sys.platform.startswith('win'), reason='real withdrawn Windows Tk; no Linux display required')
def test_real_transcription_page_toggles_nested_formatting_checkbox():
    # A separate interpreter avoids the headless suite's tkinter stubs. The
    # real page is built withdrawn: no user focus, clipboard or typing changes.
    probe = textwrap.dedent('''
        import tkinter as tk
        from tkinter import ttk
        from types import SimpleNamespace
        from utils.midnight_signal_ui import MidnightSignalManagementMixin, apply_midnight_signal_theme
        from utils.config_paths import settings, DEFAULT_SETTINGS
        settings.update(DEFAULT_SETTINGS)
        root = tk.Tk()
        root.withdraw()
        errors = []
        root.report_callback_exception = lambda *args: errors.append(str(args))
        try:
            apply_midnight_signal_theme(root)
            page = ttk.Frame(root)
            view = SimpleNamespace(
                backend_var=tk.StringVar(value='Remote API'),
                provider_strategy_var=tk.StringVar(value='ubuntu-gpu-preferred'),
                gpu_cleanup_var=tk.BooleanVar(value=False),
                gpu_formatting_var=tk.BooleanVar(value=False),
                feedback_capture_var=tk.StringVar(value='disabled'),
                backend_status_var=tk.StringVar(value='test'),
                _save_backend_midnight=lambda: None,
                _set_output_language_selection=lambda values: None)
            MidnightSignalManagementMixin._build_transcription_page(view, page)
            def children(widget):
                for child in widget.winfo_children():
                    yield child
                    yield from children(child)
            option = next(w for w in children(page) if isinstance(w, ttk.Checkbutton)
                          and w.cget('text') == 'Preserve paragraphs and line breaks')
            holder = option.master
            assert holder.winfo_manager() == ''
            view.gpu_cleanup_var.set(True)
            root.update_idletasks()
            assert holder.winfo_manager() == 'grid'
            assert option.winfo_reqheight() > 0
            option.invoke()
            assert view.gpu_formatting_var.get() is True
            view.gpu_cleanup_var.set(False)
            assert holder.winfo_manager() == ''
            assert not errors, errors
            print('real Tk formatting checkbox visibility verified')
        finally:
            root.destroy()
    ''')
    result = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, result.stderr
