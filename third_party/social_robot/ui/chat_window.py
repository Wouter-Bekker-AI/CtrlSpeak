"""PySide chat window that exposes independent VAD and TTS toggles."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import QPoint, QSize, Qt, Signal
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
    export_profile_requested = Signal()
    closed = Signal()

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
    ) -> None:
        super().__init__(parent)
        self._identity_display = identity_display

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

        self._build_ui()

        self._append_html.connect(self._append_to_history)
        self._invoke_callable.connect(self._dispatch_callable)
        self._update_send_enabled()
        self._apply_vad_styles()
        self._apply_tts_styles()

    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        title_label = QLabel(f"<b>Chat with {html.escape(self._identity_display)}</b>")
        title_label.setTextFormat(Qt.TextFormat.RichText)
        title_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        header.addWidget(title_label)

        self._vad_button = QPushButton()
        self._vad_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._vad_button.setCheckable(True)
        self._vad_button.setFlat(False)
        self._vad_button.setDefault(False)
        self._vad_button.setAutoDefault(False)
        self._vad_button.setAccessibleName("Toggle microphone listener")
        self._vad_button.clicked.connect(self._on_vad_button_clicked)
        header.addWidget(self._vad_button)

        self._tts_button = QPushButton()
        self._tts_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._tts_button.setCheckable(True)
        self._tts_button.setFlat(False)
        self._tts_button.setDefault(False)
        self._tts_button.setAutoDefault(False)
        self._tts_button.setAccessibleName("Toggle text-to-speech playback")
        self._tts_button.clicked.connect(self._on_tts_button_clicked)
        header.addWidget(self._tts_button)

        self._export_button = QPushButton("Export profile")
        self._export_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_button.setToolTip("Save the remembered profile details to disk")
        self._export_button.setAccessibleName("Export profile snapshot")
        self._export_button.clicked.connect(self.export_profile_requested.emit)
        header.addWidget(self._export_button)

        layout.addLayout(header)

        self._history = QTextBrowser(self)
        self._history.setReadOnly(True)
        self._history.setOpenExternalLinks(False)
        self._history.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._history.setMinimumHeight(320)
        layout.addWidget(self._history)

        self._controls_widget = QWidget(self)
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
        self._send_button.setDefault(True)
        self._send_button.clicked.connect(self._on_send_clicked)
        controls_layout.addWidget(self._send_button)

        layout.addWidget(self._controls_widget)

        self.resize(520, 640)

    # ------------------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().showEvent(event)
        self._move_to_active_corner()

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
        self._append_html.emit(
            f"<span style=\"color:#5f6368;\"><b>{escaped_speaker}:</b> {escaped_text}</span>"
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
            self._history.setMinimumHeight(220)
            self.resize(420, 420)
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
            self._history.setMinimumHeight(320)
            self.resize(520, 640)
            QApplication.processEvents()
            self.focus_input()
        self._move_to_active_corner()

    def _move_to_active_corner(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        if available.isNull():
            return
        frame = self.frameGeometry()
        margin = 24
        if self._tts_enabled:
            target_point = QPoint(available.right() - margin, available.top() + margin)
            frame.moveTopRight(target_point)
        else:
            target_point = QPoint(available.right() - margin, available.bottom() - margin)
            frame.moveBottomRight(target_point)
        self.move(frame.topLeft())

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
            self._invoke_callable.emit(callback)

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


__all__ = ["ChatWindow"]
