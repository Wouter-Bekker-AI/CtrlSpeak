# logo.py
# Transparent, draggable frameless window that shows a PNG with alpha.

import sys
import threading
from pathlib import Path
from typing import Optional

try:
    from PySide6.QtCore import Qt, QPoint, Signal, Slot, QObject
    from PySide6.QtGui import QGuiApplication, QPixmap, QAction
    from PySide6.QtWidgets import QApplication, QWidget, QLabel, QHBoxLayout, QMenu
except ImportError:
    print("ERROR: PySide6 is not installed or failed to import.")
    print("Fix: pip install PySide6")
    sys.exit(1)

class FloatingLogo(QWidget):
    def __init__(self, pixmap: QPixmap, stay_on_top: bool):
        super().__init__()
        self._drag_pos: QPoint | None = None
        self.original_pixmap = pixmap

        flags = Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool
        if stay_on_top:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self.label = QLabel(self)
        self.label.setPixmap(pixmap)
        self.label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.label)

        self._quit_action = QAction("Quit", self)
        self._quit_action.triggered.connect(QApplication.instance().quit)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._open_menu)

    @Slot(float)
    def set_scale(self, scale: float):
        w = max(1, int(self.original_pixmap.width() * scale))
        h = max(1, int(self.original_pixmap.height() * scale))
        scaled_pm = self.original_pixmap.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(scaled_pm)
        # Center the label within the fixed-size window
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def _open_menu(self, pos):
        m = QMenu(self)
        m.addAction(self._quit_action)
        m.exec_(self.mapToGlobal(pos))

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            e.accept()

    def mouseMoveEvent(self, e):
        if self._drag_pos is not None and (e.buttons() & Qt.MouseButton.LeftButton):
            self.move(e.globalPosition().toPoint() - self._drag_pos)
            e.accept()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = None
            e.accept()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape:
            QApplication.instance().quit()
        else:
            super().keyPressEvent(e)

class LogoAnimator(QObject):
    update_signal = Signal(float)

    def __init__(self, logo_path: Path, scale: float = 0.3, on_top: bool = True):
        super().__init__()
        self.logo_path = logo_path
        self.initial_scale = scale
        self.on_top = on_top
        self.app: Optional[QApplication] = None
        self.widget: Optional[FloatingLogo] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self.original_pixmap: Optional[QPixmap] = None

    def setup_widget(self):
        self.app = QApplication.instance() or QApplication(sys.argv)

        pm = QPixmap(str(self.logo_path))
        if pm.isNull():
            print(f"Failed to load logo image: {self.logo_path}")
            return
        self.original_pixmap = pm

        # Calculate max window size based on max scale (e.g., 0.4)
        max_scale = self.initial_scale + 0.1  # initial_scale (0.3) + max amplitude effect (0.1)
        max_w = max(1, int(self.original_pixmap.width() * max_scale))
        max_h = max(1, int(self.original_pixmap.height() * max_scale))

        self.widget = FloatingLogo(self.original_pixmap, self.on_top)
        self.widget.setFixedSize(max_w, max_h)
        self.update_signal.connect(self.widget.set_scale)
        self.widget.set_scale(self.initial_scale)

        screen = QGuiApplication.primaryScreen()
        if screen:
            screen_geo = screen.availableGeometry()
            # Position in bottom right corner with some padding
            self.widget.move(screen_geo.right() - self.widget.width() - 20, screen_geo.bottom() - self.widget.height() - 20)

        self.widget.show()

    def run(self):
        if not self.app:
            print("Error: setup_widget must be called before run.")
            return
        self.app.exec()

    def update_amplitude(self, amplitude: float):
        scale = self.initial_scale + (amplitude * 0.1)
        self.update_signal.emit(scale)

    def stop(self):
        if self.app:
            self.app.quit()