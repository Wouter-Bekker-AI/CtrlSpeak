"""PySide chat window that exposes independent VAD and TTS toggles."""

from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QEvent, QPoint, QSize, Qt, Signal, QTimer
from PySide6.QtGui import QGuiApplication, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class ChatWindow(QWidget):
    """Simple chat UI for SocialRobot conversations."""

    send_text = Signal(str)
    vad_toggle_requested = Signal(bool)
    tts_toggle_requested = Signal(bool)
    closed = Signal()
    geometry_changed = Signal()

    _append_html = Signal(str)
    _invoke_callable = Signal(object)

    def __init__(
        self,
        identity_display: str,
        persona_icon_path: Path,
        *,
        deaf_icon_path: Optional[Path] = None,
        speak_icon_path: Optional[Path] = None,
        mute_icon_path: Optional[Path] = None,
        parent: Optional[QWidget] = None,
        theme: str = "dark",
    ) -> None:
        super().__init__(parent)
        self._identity_display = identity_display
        self.setObjectName("chatWindowRoot")
        self.setAutoFillBackground(True)

        self._persona_icon = self._load_icon(persona_icon_path)
        self._deaf_icon = self._load_icon(deaf_icon_path)
        self._speak_icon = self._load_icon(speak_icon_path)
        self._mute_icon = self._load_icon(mute_icon_path)

        if not self._persona_icon.isNull():
            self.setWindowIcon(self._persona_icon)

        self.setWindowTitle(f"Chat with {identity_display}")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)

        self._vad_enabled = False
        self._tts_enabled = False
        normalized_theme = (theme or "dark").strip().lower()
        self._theme = normalized_theme if normalized_theme in {"light", "dark"} else "dark"
        self._status_color = "#5f6368"
        self._title_label: Optional[QLabel] = None

        self._build_ui()

        self._append_html.connect(self._append_to_history)
        self._invoke_callable.connect(self._dispatch_callable)
        self._update_send_enabled()
        self._apply_vad_styles()
        self._apply_tts_styles()
        self._apply_theme()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(12)
        title_label = QLabel(f"<b>Chat with {html.escape(self._identity_display)}</b>")
        title_label.setTextFormat(Qt.TextFormat.RichText)
        title_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        header.addWidget(title_label)
        self._title_label = title_label

        self._vad_button = QPushButton()
        self._vad_button.setObjectName("iconButton")
        self._vad_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._vad_button.setCheckable(True)
        self._vad_button.setFlat(False)
        self._vad_button.setDefault(False)
        self._vad_button.setAutoDefault(False)
        self._vad_button.setAccessibleName("Toggle microphone listener")
        self._vad_button.clicked.connect(self._on_vad_button_clicked)
        header.addWidget(self._vad_button)

        self._tts_button = QPushButton()
        self._tts_button.setObjectName("iconButton")
        self._tts_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._tts_button.setCheckable(True)
        self._tts_button.setFlat(False)
        self._tts_button.setDefault(False)
        self._tts_button.setAutoDefault(False)
        self._tts_button.setAccessibleName("Toggle text-to-speech playback")
        self._tts_button.clicked.connect(self._on_tts_button_clicked)
        header.addWidget(self._tts_button)

        layout.addLayout(header)

        self._history = QTextBrowser(self)
        self._history.setReadOnly(True)
        self._history.setOpenExternalLinks(False)
        self._history.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._history.setMinimumHeight(320)
        layout.addWidget(self._history)

        self._controls_widget = QWidget(self)
        self._controls_widget.setObjectName("controlsPanel")
        controls_layout = QHBoxLayout(self._controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(8)

        self._input = QLineEdit(self._controls_widget)
        self._default_placeholder = "Type a message…"
        self._voice_placeholder = "Type a message (sent as voice)…"
        self._input.setPlaceholderText(self._default_placeholder)
        self._input.returnPressed.connect(self._on_return_pressed)
        self._input.textChanged.connect(self._update_send_enabled)
        controls_layout.addWidget(self._input)

        self._send_button = QPushButton("Send", self._controls_widget)
        self._send_button.setObjectName("sendButton")
        self._send_button.setDefault(True)
        self._send_button.clicked.connect(self._on_send_clicked)
        controls_layout.addWidget(self._send_button)

        layout.addWidget(self._controls_widget)

        self.resize(520, 640)

    # ------------------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().showEvent(event)
        self._move_to_active_corner()
        QTimer.singleShot(0, lambda: self._apply_titlebar_theme(ensure_activation=True))

    # ------------------------------------------------------------------
    def append_user_message(self, text: str, *, medium: Optional[str] = None) -> None:
        if not text:
            return
        speaker = self._resolve_user_speaker_label(medium)
        self._append_message(speaker, text)

    def append_bot_message(self, text: str) -> None:
        if not text:
            return
        self._append_message("Bot", text)

    def append_status_message(self, speaker: str, text: str) -> None:
        """Display a non-persistent status update in the transcript."""

        if not text:
            return
        escaped_text = html.escape(text).replace("\n", "<br>")
        escaped_speaker = html.escape(speaker)
        color = self._status_color
        self._append_html.emit(
            f"<span style=\"color:{color};\"><b>{escaped_speaker}:</b> {escaped_text}</span>"
        )

    def _append_message(self, speaker: str, text: str) -> None:
        escaped_text = html.escape(text).replace("\n", "<br>")
        escaped_speaker = html.escape(speaker)
        self._append_html.emit(f"<b>{escaped_speaker}:</b> {escaped_text}")

    def _resolve_user_speaker_label(self, medium: Optional[str]) -> str:
        if not medium:
            return "You"
        normalized = medium.strip().lower()
        if normalized == "voice":
            return "You (voice)"
        if normalized == "text":
            return "You (text)"
        return "You"

    def _append_to_history(self, html_text: str) -> None:
        self._history.append(html_text)
        cursor = self._history.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self._history.setTextCursor(cursor)

    def _dispatch_callable(self, callback: object) -> None:
        if not callable(callback):
            return
        try:
            callback()
        except Exception:
            # Propagate exceptions to stderr so failures remain visible next to
            # the bot logs. Re-raise to avoid swallowing unexpected errors.
            import traceback

            traceback.print_exc()
            raise

    def clear_input(self) -> None:
        self._input.clear()

    def focus_input(self) -> None:
        self._input.setFocus(Qt.FocusReason.ActiveWindowFocusReason)

    def transcript_widget(self) -> QTextBrowser:
        return self._history

    # ------------------------------------------------------------------
    def set_vad_enabled(self, enabled: bool) -> None:
        if self._vad_enabled == enabled:
            return
        self._vad_enabled = enabled
        self._apply_vad_styles()

    def set_tts_enabled(self, enabled: bool) -> None:
        if self._tts_enabled == enabled:
            return
        self._tts_enabled = enabled
        self._apply_tts_styles()

    def set_theme(self, theme: str) -> None:
        normalized = (theme or "").strip().lower()
        if normalized not in {"light", "dark"}:
            normalized = "dark"
        if normalized == self._theme:
            return
        self._theme = normalized
        self._apply_theme()

    def set_theme_async(self, theme: str) -> None:
        self.invoke(lambda: self.set_theme(theme))

    def _apply_vad_styles(self) -> None:
        if self._vad_enabled:
            self._vad_button.setChecked(True)
            self._vad_button.setToolTip("Disable microphone listener")
            if self._persona_icon.isNull():
                self._vad_button.setText("🎙")
                self._vad_button.setIcon(QIcon())
            else:
                self._vad_button.setText("")
                self._vad_button.setIcon(self._persona_icon)
                self._vad_button.setIconSize(QSize(24, 24))
            self._input.setPlaceholderText(self._voice_placeholder)
        else:
            self._vad_button.setChecked(False)
            self._vad_button.setToolTip("Enable microphone listener")
            if self._deaf_icon.isNull():
                self._vad_button.setText("🙉")
                self._vad_button.setIcon(QIcon())
            else:
                self._vad_button.setText("")
                self._vad_button.setIcon(self._deaf_icon)
                self._vad_button.setIconSize(QSize(24, 24))
            self._input.setPlaceholderText(self._default_placeholder)

    def _apply_tts_styles(self) -> None:
        if self._tts_enabled:
            self._tts_button.setChecked(True)
            self._tts_button.setToolTip("Mute persona audio")
            if self._speak_icon.isNull():
                self._tts_button.setText("🔊")
                self._tts_button.setIcon(QIcon())
            else:
                self._tts_button.setText("")
                self._tts_button.setIcon(self._speak_icon)
                self._tts_button.setIconSize(QSize(24, 24))
        else:
            self._tts_button.setChecked(False)
            self._tts_button.setToolTip("Enable persona audio")
            if self._mute_icon.isNull():
                self._tts_button.setText("🔇")
                self._tts_button.setIcon(QIcon())
            else:
                self._tts_button.setText("")
                self._tts_button.setIcon(self._mute_icon)
                self._tts_button.setIconSize(QSize(24, 24))
            QApplication.processEvents()
            self.focus_input()
        self._move_to_active_corner()

    def _apply_theme(self) -> None:
        if self._theme == "dark":
            self._status_color = "#9aa0a6"
            self.setStyleSheet(
                """
                QWidget#chatWindowRoot {
                    background-color: #121212;
                    color: #e8eaed;
                }
                QTextBrowser {
                    background-color: #1e1e1e;
                    color: #e8eaed;
                    border: 1px solid #3c4043;
                    border-radius: 8px;
                    padding: 8px;
                }
                QLineEdit {
                    background-color: #1e1e1e;
                    color: #e8eaed;
                    border: 1px solid #3c4043;
                    border-radius: 6px;
                    padding: 6px;
                }
                QLineEdit::placeholder {
                    color: #9aa0a6;
                }
                QPushButton#sendButton {
                    background-color: #8ab4f8;
                    color: #202124;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-weight: 600;
                }
                QPushButton#sendButton:disabled {
                    background-color: #3c4043;
                    color: #9aa0a6;
                }
                QPushButton#iconButton {
                    background-color: #2d2f31;
                    color: #e8eaed;
                    border: 1px solid #3c4043;
                    border-radius: 6px;
                    padding: 6px;
                    min-width: 36px;
                    min-height: 36px;
                }
                QPushButton#iconButton:checked {
                    background-color: #5f6368;
                }
                QWidget#controlsPanel {
                    background-color: transparent;
                }
                """
            )
            if self._title_label is not None:
                self._title_label.setStyleSheet("color: #e8eaed;")
        else:
            self._status_color = "#5f6368"
            self.setStyleSheet(
                """
                QWidget#chatWindowRoot {
                    background-color: #f1f3f4;
                    color: #202124;
                }
                QTextBrowser {
                    background-color: #ffffff;
                    color: #202124;
                    border: 1px solid #dadce0;
                    border-radius: 8px;
                    padding: 8px;
                }
                QLineEdit {
                    background-color: #ffffff;
                    color: #202124;
                    border: 1px solid #dadce0;
                    border-radius: 6px;
                    padding: 6px;
                }
                QLineEdit::placeholder {
                    color: #5f6368;
                }
                QPushButton#sendButton {
                    background-color: #1a73e8;
                    color: #ffffff;
                    border-radius: 6px;
                    padding: 8px 16px;
                    font-weight: 600;
                }
                QPushButton#sendButton:disabled {
                    background-color: #dadce0;
                    color: #9aa0a6;
                }
                QPushButton#iconButton {
                    background-color: #ffffff;
                    color: #202124;
                    border: 1px solid #dadce0;
                    border-radius: 6px;
                    padding: 6px;
                    min-width: 36px;
                    min-height: 36px;
                }
                QPushButton#iconButton:checked {
                    background-color: #e8f0fe;
                    border-color: #1a73e8;
                }
                QWidget#controlsPanel {
                    background-color: transparent;
                }
                """
            )
            if self._title_label is not None:
                self._title_label.setStyleSheet("color: #202124;")
        self._apply_titlebar_theme(ensure_activation=True)

    def _move_to_active_corner(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        if available.isNull():
            return
        frame = self.frameGeometry()
        margin = 24
        target_point = QPoint(available.right() - margin, available.bottom() - margin)
        frame.moveBottomRight(target_point)
        self.move(frame.topLeft())

    def moveEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().moveEvent(event)
        self.geometry_changed.emit()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        self.geometry_changed.emit()

    def changeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().changeEvent(event)
        if event is None:
            return
        if event.type() == QEvent.Type.WindowStateChange:
            self.geometry_changed.emit()

    # ------------------------------------------------------------------
    def _on_send_clicked(self) -> None:
        text = self._input.text().strip()
        if not text:
            return
        self.clear_input()
        self.send_text.emit(text)

    def _on_return_pressed(self) -> None:
        self._on_send_clicked()

    def _update_send_enabled(self) -> None:
        self._send_button.setEnabled(bool(self._input.text().strip()))

    def _on_vad_button_clicked(self) -> None:
        self.vad_toggle_requested.emit(not self._vad_enabled)

    def _on_tts_button_clicked(self) -> None:
        self.tts_toggle_requested.emit(not self._tts_enabled)

    # ------------------------------------------------------------------
    def invoke(self, callback: Callable[[], None]) -> None:
        """Schedule ``callback`` to run on the UI thread."""

        if callable(callback):
            try:
                self._invoke_callable.emit(callback)
            except RuntimeError:
                # The widget may already be deleted during shutdown; ignore the
                # signal failure so cleanup can continue gracefully.
                return

    def set_vad_enabled_async(self, enabled: bool) -> None:
        self.invoke(lambda: self.set_vad_enabled(enabled))

    def set_tts_enabled_async(self, enabled: bool) -> None:
        self.invoke(lambda: self.set_tts_enabled(enabled))

    def close_async(self) -> None:
        self.invoke(self.close)

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self.closed.emit()
        super().closeEvent(event)

    # ------------------------------------------------------------------
    @staticmethod
    def _load_icon(path: Optional[Path]) -> QIcon:
        if path is None:
            return QIcon()
        try:
            icon = QIcon(str(path))
        except Exception:
            return QIcon()
        return icon

    # ------------------------------------------------------------------
    def _apply_titlebar_theme(self, *, ensure_activation: bool = False) -> None:
        """Toggle the native title bar between light and dark modes when possible."""

        if sys.platform != "win32":
            return

        window = self.windowHandle()
        if window is None:
            QTimer.singleShot(0, lambda: self._apply_titlebar_theme(ensure_activation=ensure_activation))
            return

        try:
            import ctypes
        except Exception:
            return

        try:
            dwmapi = ctypes.windll.dwmapi
        except Exception:
            return

        hwnd = int(window.winId())
        use_dark = ctypes.c_int(1 if self._theme == "dark" else 0)
        attribute_ids = (20, 19)  # Windows 11/late Windows 10, early Windows 10

        for attribute_id in attribute_ids:
            try:
                result = dwmapi.DwmSetWindowAttribute(
                    hwnd,
                    attribute_id,
                    ctypes.byref(use_dark),
                    ctypes.sizeof(use_dark),
                )
            except Exception:
                return
            if result == 0:
                break
        else:
            return

        # Request rounded corners when available (Windows 11).
        try:
            corner_preference_attr = 33  # DWMWA_WINDOW_CORNER_PREFERENCE
            dwmwcp_round = ctypes.c_int(2)  # DWMWCP_ROUND
            dwmapi.DwmSetWindowAttribute(
                hwnd,
                corner_preference_attr,
                ctypes.byref(dwmwcp_round),
                ctypes.sizeof(dwmwcp_round),
            )
        except Exception:
            pass

        if ensure_activation and self.isVisible():
            try:
                window.requestActivate()
            except Exception:
                pass
            self.raise_()
            self.activateWindow()



__all__ = ["ChatWindow"]
