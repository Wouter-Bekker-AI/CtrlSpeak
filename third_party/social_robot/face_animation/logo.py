# logo.py
# Transparent, draggable frameless window that shows a PNG with alpha.

import sys
import threading
from pathlib import Path
from typing import Callable, Optional

try:
    from PySide6.QtCore import Qt, QPoint, Signal, Slot, QObject, QTimer
    from PySide6.QtGui import QAction, QGuiApplication, QPixmap
    from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QMenu, QWidget
except ImportError:
    print("ERROR: PySide6 is not installed or failed to import.")
    print("Fix: pip install PySide6")
    sys.exit(1)


class FloatingLogo(QWidget):
    def __init__(
        self,
        pixmap: QPixmap,
        stay_on_top: bool,
        on_look_at_screen: Optional[Callable[[], None]] = None,
        on_look_at_clipboard: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__()
        self._drag_pos: QPoint | None = None
        self.original_pixmap = pixmap
        self._on_look_at_screen = on_look_at_screen
        self._on_look_at_clipboard = on_look_at_clipboard
        self._on_top_timer: Optional[QTimer] = None

        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
        if stay_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self.label = QLabel(self)
        self.label.setPixmap(pixmap)
        self.label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)

        self._look_action = QAction("Look at my Screen", self)
        self._look_action.triggered.connect(self._trigger_look_at_screen)
        self._look_action.setEnabled(on_look_at_screen is not None)
        self._clipboard_action = QAction("Look at my Clipboard", self)
        self._clipboard_action.triggered.connect(self._trigger_look_at_clipboard)
        self._clipboard_action.setEnabled(on_look_at_clipboard is not None)
        self._quit_action = QAction("Quit", self)
        self._quit_action.triggered.connect(QApplication.instance().quit)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._open_menu)

        if stay_on_top:
            self._on_top_timer = QTimer(self)
            self._on_top_timer.setInterval(500)
            self._on_top_timer.timeout.connect(self._refresh_on_top)
            self._on_top_timer.start()

    @Slot(float)
    def set_scale(self, scale: float) -> None:
        width = max(1, int(self.original_pixmap.width() * scale))
        height = max(1, int(self.original_pixmap.height() * scale))
        scaled_pm = self.original_pixmap.scaled(
            width,
            height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.label.setPixmap(scaled_pm)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(width, height)

    def set_capture_callbacks(
        self,
        *,
        screen: Optional[Callable[[], None]],
        clipboard: Optional[Callable[[], None]],
    ) -> None:
        self._on_look_at_screen = screen
        self._on_look_at_clipboard = clipboard
        self._look_action.setEnabled(screen is not None)
        self._clipboard_action.setEnabled(clipboard is not None)

    def set_look_at_screen_callback(
        self, callback: Optional[Callable[[], None]]
    ) -> None:
        self.set_capture_callbacks(
            screen=callback,
            clipboard=self._on_look_at_clipboard,
        )

    def set_look_at_clipboard_callback(
        self, callback: Optional[Callable[[], None]]
    ) -> None:
        self.set_capture_callbacks(
            screen=self._on_look_at_screen,
            clipboard=callback,
        )

    def _trigger_look_at_screen(self) -> None:
        if self._on_look_at_screen is None:
            return
        try:
            self._on_look_at_screen()
        except Exception:
            # Keep failures visible in the terminal alongside the bot logs.
            import traceback

            traceback.print_exc()

    def _trigger_look_at_clipboard(self) -> None:
        if self._on_look_at_clipboard is None:
            return
        try:
            self._on_look_at_clipboard()
        except Exception:
            import traceback

            traceback.print_exc()

    def _open_menu(self, pos) -> None:
        menu = QMenu(self)
        if self._on_look_at_screen is not None:
            menu.addAction(self._look_action)
        if self._on_look_at_clipboard is not None:
            menu.addAction(self._clipboard_action)
        menu.addAction(self._quit_action)
        menu.exec_(self.mapToGlobal(pos))

    def _refresh_on_top(self) -> None:
        if not self.isVisible() or self.isHidden():
            return
        if not (self.windowFlags() & Qt.WindowType.WindowStaysOnTopHint):
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
            self.show()
        self.raise_()

    def mousePressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = (
                event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            )
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self._drag_pos is not None and (event.buttons() & Qt.MouseButton.LeftButton):
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = None
            event.accept()

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if event.key() == Qt.Key.Key_Escape:
            QApplication.instance().quit()
        else:
            super().keyPressEvent(event)


class LogoAnimator(QObject):
    update_signal = Signal(float)

    def __init__(self, logo_path: Path, scale: float = 0.3, on_top: bool = True) -> None:
        super().__init__()
        self.logo_path = logo_path
        self.initial_scale = scale
        self._base_scale = scale
        self._min_amplitude_delta = 0.012
        self._max_amplitude_delta = 0.05
        self._amplitude_factor = 0.18
        self._amplitude_delta = 0.05
        self._current_amplitude = 0.0
        self.on_top = on_top
        self.app: Optional[QApplication] = None
        self.widget: Optional[FloatingLogo] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self.original_pixmap: Optional[QPixmap] = None
        self._on_look_at_screen: Optional[Callable[[], None]] = None
        self._on_look_at_clipboard: Optional[Callable[[], None]] = None

    def setup_widget(
        self,
        on_look_at_screen: Optional[Callable[[], None]] = None,
        on_look_at_clipboard: Optional[Callable[[], None]] = None,
    ) -> None:
        if on_look_at_screen is not None:
            self._on_look_at_screen = on_look_at_screen
        if on_look_at_clipboard is not None:
            self._on_look_at_clipboard = on_look_at_clipboard

        self.app = QApplication.instance() or QApplication(sys.argv)

        pixmap = QPixmap(str(self.logo_path))
        if pixmap.isNull():
            print(f"Failed to load logo image: {self.logo_path}")
            return
        self.original_pixmap = pixmap

        self._recompute_amplitude_delta()

        # Calculate max window size based on the maximum amplitude-driven scale bump.
        max_scale = self._base_scale + self._max_amplitude_delta
        max_width = max(1, int(self.original_pixmap.width() * max_scale))
        max_height = max(1, int(self.original_pixmap.height() * max_scale))

        self.widget = FloatingLogo(
            self.original_pixmap,
            self.on_top,
            on_look_at_screen=self._on_look_at_screen,
            on_look_at_clipboard=self._on_look_at_clipboard,
        )
        self.widget.setMinimumSize(1, 1)
        self.widget.setMaximumSize(max_width, max_height)
        self.update_signal.connect(self.widget.set_scale)
        self.widget.set_scale(self.initial_scale)

        screen = QGuiApplication.primaryScreen()
        if screen:
            screen_geo = screen.availableGeometry()
            # Position in bottom right corner with some padding
            self.widget.move(
                screen_geo.right() - self.widget.width() - 20,
                screen_geo.bottom() - self.widget.height() - 20,
            )

        self.widget.hide()

    def _apply_scale(self, scale: float) -> None:
        if self.widget is None:
            return
        clamped = max(0.01, min(scale, 1.0))
        self.widget.set_scale(clamped)

    def set_base_scale(self, scale: float) -> None:
        clamped = max(0.05, min(scale, 1.0))
        self._base_scale = clamped
        self._recompute_amplitude_delta()
        self._apply_scale(clamped + self._current_amplitude * self._amplitude_delta)

    def center_on_widget(self, target: QWidget) -> None:
        if self.widget is None or target is None:
            return
        if not target.isVisible():
            return
        frame = target.frameGeometry()
        if frame.isNull():
            return
        x = frame.center().x() - (self.widget.width() // 2)
        y = frame.center().y() - (self.widget.height() // 2)
        self.widget.move(x, y)

    def anchor_to_widget_bottom_right(self, target: QWidget, margin: int = 24) -> None:
        if self.widget is None or target is None:
            return
        if not target.isVisible():
            return
        rect = target.rect()
        bottom_right = target.mapToGlobal(rect.bottomRight())
        x = bottom_right.x() - self.widget.width() - margin
        y = bottom_right.y() - self.widget.height() - margin
        screen = QGuiApplication.primaryScreen()
        if screen:
            available = screen.availableGeometry()
            if not available.isNull():
                x = max(available.left(), min(x, available.right() - self.widget.width()))
                y = max(available.top(), min(y, available.bottom() - self.widget.height()))
        self.widget.move(x, y)

    def move_to_screen_corner(self, padding: int = 20) -> None:
        if self.widget is None:
            return
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        screen_geo = screen.availableGeometry()
        if screen_geo.isNull():
            return
        x = screen_geo.right() - self.widget.width() - padding
        y = screen_geo.bottom() - self.widget.height() - padding
        self.widget.move(x, y)

    def run(self) -> None:
        if not self.app:
            print("Error: setup_widget must be called before run.")
            return
        self.app.exec()

    def update_amplitude(self, amplitude: float) -> None:
        clamped = max(0.0, min(amplitude, 1.0))
        self._current_amplitude = clamped
        scale = self._base_scale + (clamped * self._amplitude_delta)
        self.update_signal.emit(scale)

    def _recompute_amplitude_delta(self) -> None:
        target = self._base_scale * self._amplitude_factor
        self._amplitude_delta = max(
            self._min_amplitude_delta,
            min(self._max_amplitude_delta, target),
        )

    def stop(self) -> None:
        if self.widget is not None:
            self.widget.hide()
        if self.app:
            self.app.quit()

    def show_widget(self) -> None:
        if self.widget is not None:
            self.widget.show()
            self.widget.raise_()

    def hide_widget(self) -> None:
        if self.widget is not None:
            self.widget.hide()

    def set_look_at_screen_callback(
        self, callback: Optional[Callable[[], None]]
    ) -> None:
        self._on_look_at_screen = callback
        if self.widget is not None:
            self.widget.set_capture_callbacks(
                screen=callback,
                clipboard=self._on_look_at_clipboard,
            )

    def set_look_at_clipboard_callback(
        self, callback: Optional[Callable[[], None]]
    ) -> None:
        self._on_look_at_clipboard = callback
        if self.widget is not None:
            self.widget.set_capture_callbacks(
                screen=self._on_look_at_screen,
                clipboard=callback,
            )
