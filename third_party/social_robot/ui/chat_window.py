"""PySide chat window that toggles between text and voice modes."""

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
    """Simple chat UI for SocialRobot text-first conversations."""

    send_text = Signal(str)
    voice_mode_requested = Signal(bool)
    export_profile_requested = Signal()
    closed = Signal()

    _append_html = Signal(str)
    _invoke_callable = Signal(object)

    def __init__(
        self,
        identity_display: str,
        icon_path: Path,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._voice_mode = False
        self._identity_display = identity_display

        self._icon = QIcon(str(icon_path))
        if not self._icon.isNull():
            self.setWindowIcon(self._icon)

        self.setWindowTitle(f"Chat with {identity_display}")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._build_ui()

        self._append_html.connect(self._append_to_history)
        self._update_send_enabled()
        self._apply_mode_styles()

        self._invoke_callable.connect(self._dispatch_callable)

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

        self._mode_button = QPushButton()
        if self._icon.isNull():
            self._mode_button.setText("🎤")
        else:
            self._mode_button.setIcon(self._icon)
            self._mode_button.setIconSize(QSize(24, 24))
        self._mode_button.setToolTip("Switch to voice mode")
        self._mode_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._mode_button.setCheckable(True)
        self._mode_button.setFlat(False)
        self._mode_button.setDefault(False)
        self._mode_button.setAutoDefault(False)
        self._mode_button.setAccessibleName("Toggle microphone mode")
        self._mode_button.clicked.connect(self._on_mode_button_clicked)
        header.addWidget(self._mode_button)

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
    def set_voice_mode(self, enabled: bool) -> None:
        if self._voice_mode == enabled:
            return
        self._voice_mode = enabled
        self._apply_mode_styles()

    def _apply_mode_styles(self) -> None:
        if self._voice_mode:
            self._mode_button.setChecked(True)
            self._mode_button.setStyleSheet(
                "QPushButton { background-color: #1a73e8; color: white; border-radius: 4px; }"
            )
            self._mode_button.setToolTip("Switch to text chat")
            if self._icon.isNull():
                self._mode_button.setText("⌨")
            else:
                self._mode_button.setText("")
            self._controls_widget.setVisible(True)
            self._input.setPlaceholderText(self._voice_placeholder)
            self._history.setMinimumHeight(220)
            self.resize(420, 420)
        else:
            self._mode_button.setChecked(False)
            self._mode_button.setStyleSheet("")
            self._mode_button.setToolTip("Switch to voice mode")
            if self._icon.isNull():
                self._mode_button.setText("🎤")
            else:
                self._mode_button.setText("")
            self._controls_widget.setVisible(True)
            self._input.setPlaceholderText(self._default_placeholder)
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
        # Keep a small margin from the edges to match the floating logo placement.
        margin = 24
        if self._voice_mode:
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

    def _on_mode_button_clicked(self) -> None:
        target = not self._voice_mode
        self.voice_mode_requested.emit(target)

    # ------------------------------------------------------------------
    def invoke(self, callback: Callable[[], None]) -> None:
        """Schedule ``callback`` to run on the UI thread."""

        if callable(callback):
            self._invoke_callable.emit(callback)

    def set_voice_mode_async(self, enabled: bool) -> None:
        self.invoke(lambda: self.set_voice_mode(enabled))

    def close_async(self) -> None:
        self.invoke(self.close)

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self.closed.emit()
        super().closeEvent(event)


__all__ = ["ChatWindow"]
