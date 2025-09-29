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
    ) -> None:
        super().__init__()
        self._drag_pos: QPoint | None = None
        self.original_pixmap = pixmap
        self._on_look_at_screen = on_look_at_screen
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

    def set_look_at_screen_callback(
        self, callback: Optional[Callable[[], None]]
    ) -> None:
        self._on_look_at_screen = callback
        self._look_action.setEnabled(callback is not None)

    def _trigger_look_at_screen(self) -> None:
        if self._on_look_at_screen is None:
            return
        try:
            self._on_look_at_screen()
        except Exception:
            # Keep failures visible in the terminal alongside the bot logs.
            import traceback

            traceback.print_exc()

    def _open_menu(self, pos) -> None:
        menu = QMenu(self)
        menu.addAction(self._look_action)
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
        self.on_top = on_top
        self.app: Optional[QApplication] = None
        self.widget: Optional[FloatingLogo] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self.original_pixmap: Optional[QPixmap] = None
        self._look_callback: Optional[Callable[[], None]] = None

    def setup_widget(
        self, on_look_at_screen: Optional[Callable[[], None]] = None
    ) -> None:
        if on_look_at_screen is not None:
            self._look_callback = on_look_at_screen

        self.app = QApplication.instance() or QApplication(sys.argv)

        pixmap = QPixmap(str(self.logo_path))
        if pixmap.isNull():
            print(f"Failed to load logo image: {self.logo_path}")
            return
        self.original_pixmap = pixmap

        # Calculate max window size based on max scale (e.g., 0.4)
        max_scale = self.initial_scale + 0.1  # initial_scale (0.3) + max amplitude effect (0.1)
        max_width = max(1, int(self.original_pixmap.width() * max_scale))
        max_height = max(1, int(self.original_pixmap.height() * max_scale))

        self.widget = FloatingLogo(
            self.original_pixmap,
            self.on_top,
            on_look_at_screen=self._look_callback,
        )
        self.widget.setFixedSize(max_width, max_height)
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

        self.widget.show()

    def run(self) -> None:
        if not self.app:
            print("Error: setup_widget must be called before run.")
            return
        self.app.exec()

    def update_amplitude(self, amplitude: float) -> None:
        scale = self.initial_scale + (amplitude * 0.1)
        self.update_signal.emit(scale)

    def stop(self) -> None:
        if self.app:
            self.app.quit()

    def set_look_at_screen_callback(
        self, callback: Optional[Callable[[], None]]
    ) -> None:
        self._look_callback = callback
        if self.widget is not None:
            self.widget.set_look_at_screen_callback(callback)
