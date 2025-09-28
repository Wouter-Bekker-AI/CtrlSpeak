"""Animated face projection for the social robot."""

from __future__ import annotations

import math
import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path

import pygame


ASSET_ROOT = Path(__file__).resolve().parents[1] / "images" / "Desktop"
FACE_IMAGE_DEFAULT = str(ASSET_ROOT / "face.png")
MOUTH_IMAGE_DEFAULT = str(ASSET_ROOT / "mouth.png")


@dataclass
class FaceSettings:
    """Settings for the animated face display."""
    window_size: Tuple[int, int] = (1920, 1080)
    display_size: Optional[Tuple[int, int]] = None
    rotation_degrees: int = 0
    background_color: Tuple[int, int, int] = (8, 10, 26)
    face_image_path: str = FACE_IMAGE_DEFAULT
    mouth_image_path: str = MOUTH_IMAGE_DEFAULT
    face_offset: Tuple[int, int] = (-150, 0)
    face_scale: float = 0.8
    face_image_rotation: float = 180.0
    # Uncomment for Desktop (Uncommented by default)
    mouth_anchor: Tuple[float, float] = (0.5, 0.28)
    # Uncomment for Robot
    #mouth_anchor: Tuple[float, float] = (0.28, 0.5)
    mouth_base_scale: Tuple[float, float] = (1.5, 1.5)
    mouth_min_scale: float = 0.6
    mouth_max_scale: float = 1.35
    breath_amplitude: float = 6.0
    fps: int = 60


class FaceAnimator:
    """Real-time face renderer using pre-rendered art with lip-sync support."""

    def __init__(self, settings: Optional[FaceSettings] = None) -> None:
        self.settings = settings or FaceSettings()
        self.render_size = self.settings.window_size
        rotation = self.settings.rotation_degrees % 360
        if self.settings.display_size is not None:
            self.window_size = self.settings.display_size
        elif rotation in (90, 270):
            self.window_size = (self.render_size[1], self.render_size[0])
        else:
            self.window_size = self.render_size
        self._rotation_angle = rotation

        self._amplitude_queue: "queue.Queue[Tuple[float, float]]" = queue.Queue()
        self._running = threading.Event()
        self._running.set()

        self._current_mouth_level = 0.0
        self._target_mouth_level = 0.0
        self._last_mouth_update = time.monotonic()

        self._breath_phase = 0.0

        self._clock: Optional[pygame.time.Clock] = None
        self._screen: Optional[pygame.Surface] = None
        self._canvas: Optional[pygame.Surface] = None
        self._face_surface: Optional[pygame.Surface] = None
        self._mouth_surface: Optional[pygame.Surface] = None
        self._scaled_face_size: Optional[Tuple[int, int]] = None
        self._mouth_base_size: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # Animation helpers
    def _update_breath(self, dt: float) -> None:
        self._breath_phase += dt * 0.6

    def _update_mouth(self, dt: float) -> None:
        now = time.monotonic()
        while not self._amplitude_queue.empty():
            timestamp, amplitude = self._amplitude_queue.get_nowait()
            self._target_mouth_level = max(0.0, min(amplitude, 1.0))
            self._last_mouth_update = timestamp

        if now - self._last_mouth_update > 0.35:
            self._target_mouth_level *= 0.85

        smoothing = min(1.0, dt * 14.0)
        self._current_mouth_level += (self._target_mouth_level - self._current_mouth_level) * smoothing

    # ------------------------------------------------------------------
    # Public API
    def update_amplitude(self, amplitude: float) -> None:
        self._amplitude_queue.put((time.monotonic(), amplitude))

    def stop(self) -> None:
        self._running.clear()

    # ------------------------------------------------------------------
    def _load_assets(self) -> None:
        settings = self.settings
        try:
            face_image = pygame.image.load(settings.face_image_path).convert_alpha()
        except Exception as exc:  # pragma: no cover - runtime asset failure
            raise RuntimeError(
                f"Failed to load face image '{settings.face_image_path}': {exc}"
            ) from exc

        try:
            mouth_image = pygame.image.load(settings.mouth_image_path).convert_alpha()
        except Exception as exc:  # pragma: no cover - runtime asset failure
            raise RuntimeError(
                f"Failed to load mouth image '{settings.mouth_image_path}': {exc}"
            ) from exc

        scaled_face_width = max(1, int(face_image.get_width() * settings.face_scale))
        scaled_face_height = max(1, int(face_image.get_height() * settings.face_scale))
        self._scaled_face_size = (scaled_face_width, scaled_face_height)
        self._face_surface = pygame.transform.smoothscale(face_image, self._scaled_face_size).convert_alpha()

        base_mouth_width = max(
            1,
            int(mouth_image.get_width() * settings.face_scale * settings.mouth_base_scale[0]),
        )
        base_mouth_height = max(
            1,
            int(mouth_image.get_height() * settings.face_scale * settings.mouth_base_scale[1]),
        )
        self._mouth_base_size = (base_mouth_width, base_mouth_height)
        self._mouth_surface = pygame.transform.smoothscale(mouth_image, self._mouth_base_size).convert_alpha()

    def _draw_face(self) -> None:
        assert self._screen is not None
        assert self._canvas is not None
        assert self._face_surface is not None
        assert self._mouth_surface is not None
        assert self._scaled_face_size is not None
        assert self._mouth_base_size is not None
        surface = self._canvas
        w, h = surface.get_size()
        settings = self.settings

        surface.fill(settings.background_color)

        breath_offset = int(math.sin(self._breath_phase) * settings.breath_amplitude)

        face_layer = self._face_surface.copy()

        openness = max(0.0, min(self._current_mouth_level, 1.0)) ** 0.7
        scale_y = settings.mouth_min_scale + (settings.mouth_max_scale - settings.mouth_min_scale) * openness
        scale_y = max(0.05, scale_y)

        mouth_width = self._mouth_base_size[0]
        mouth_height = max(1, int(self._mouth_base_size[1] * scale_y))
        mouth_surface = pygame.transform.smoothscale(
            self._mouth_surface, (mouth_width, mouth_height)
        ).convert_alpha()

        def _resolve_anchor(value: float, size: int) -> int:
            if -1.0 <= value <= 1.0:
                return int(size * value)
            return int(value * settings.face_scale)

        anchor_x = _resolve_anchor(settings.mouth_anchor[0], self._scaled_face_size[0])
        anchor_y = _resolve_anchor(settings.mouth_anchor[1], self._scaled_face_size[1])
        mouth_rect = mouth_surface.get_rect(center=(anchor_x, anchor_y))
        face_layer.blit(mouth_surface, mouth_rect)

        rotated_face = pygame.transform.rotozoom(face_layer, -settings.face_image_rotation, 1.0)
        dest_rect = rotated_face.get_rect()
        dest_rect.center = (
            w // 2 + settings.face_offset[0],
            h // 2 + settings.face_offset[1] + breath_offset,
        )

        surface.blit(rotated_face, dest_rect)

        final_surface = surface
        if self._rotation_angle % 360 != 0:
            final_surface = pygame.transform.rotate(surface, self._rotation_angle)

        if final_surface.get_size() != self.window_size:
            final_surface = pygame.transform.smoothscale(final_surface, self.window_size)

        self._screen.fill(settings.background_color)
        dest_rect = final_surface.get_rect(center=(self.window_size[0] // 2, self.window_size[1] // 2))
        self._screen.blit(final_surface, dest_rect)

    # ------------------------------------------------------------------
    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("Robot Face")
        self._screen = pygame.display.set_mode(self.window_size)
        self._canvas = pygame.Surface(self.render_size).convert_alpha()
        self._clock = pygame.time.Clock()
        self._load_assets()

        while self._running.is_set():
            dt = self._clock.tick(self.settings.fps) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.stop()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    self.stop()

            self._update_breath(dt)
            self._update_mouth(dt)

            self._draw_face()
            pygame.display.flip()

        pygame.quit()


__all__ = ["FaceAnimator", "FaceSettings"]
