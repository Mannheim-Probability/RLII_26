from __future__ import annotations

from pathlib import Path
from collections import deque
from typing import Any, Optional

import numpy as np
import pygame

# Pygame's normal display module owns one window. For the separate reward
# interface we use pygame's SDL2 window API when available.
try:
    from pygame._sdl2.video import (
        Window as SDL2Window,
        Renderer as SDL2Renderer,
        Texture as SDL2Texture,
    )
    SDL2_MULTIWINDOW_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    SDL2Window = None
    SDL2Renderer = None
    SDL2Texture = None
    SDL2_MULTIWINDOW_AVAILABLE = False


class SupplyChainRendererNS:
    """
    Pygame renderer for SupplyChainEnv.

    The renderer is intentionally separated from the RL environment:
    - It READS the environment state.
    - It NEVER changes inventory, demand, shipments, rewards, etc.
    - PPO continues to observe the normal Dict observation.
    - render_mode="human" opens a pygame window.
    - render_mode="rgb_array" returns an RGB numpy array for video recording.

    Optional PNG assets
    -------------------
    Put the following files into an `assets/` directory next to this file:

        assets/
            supplier.png
            warehouse.png
            truck.png
            package.png
            customer.png

    Missing files are not an error. The renderer draws simple fallback graphics.

    Expected environment attributes
    -------------------------------
    Required:
        env.inventory
        env.capacity
        env.demand
        env.max_order
        env.max_lead_time
        env.step_count
        env._shipments

    Nice to have:
        env.last_info
        env.order_history
        env._get_in_transit()

    Recommended shipment dictionary:
        {
            "id": 3,
            "qty": 20,
            "remaining": 2,
            "initial_lead_time": 4,
        }

    If `initial_lead_time` is missing, max_lead_time is used as a fallback.
    """

    WIDTH = 1280
    HEIGHT = 1040
    SCENE_HEIGHT = 700
    SCENE_OFFSET_Y = 335
    HISTORY_LENGTH = 60

    # Separate reward window.
    REWARD_WINDOW_WIDTH = 960
    REWARD_WINDOW_HEIGHT = 460
    REWARD_MOVING_AVERAGE = 100

    SPEED_PRESETS = (
        {
            "key": "very_slow",
            "label": "I",
            "render_every_n_steps": 1,
            "animate_frames": 20,
            "fps": 20,
            "limit_fps": True,
        },
        {
            "key": "slow",
            "label": "II",
            "render_every_n_steps": 1,
            "animate_frames": 10,
            "fps": 30,
            "limit_fps": True,
        },
        {
            "key": "normal",
            "label": "III",
            "render_every_n_steps": 1,
            "animate_frames": 4,
            "fps": 60,
            "limit_fps": True,
        },
        {
            "key": "fast",
            "label": "IV",
            "render_every_n_steps": 1,
            "animate_frames": 1,
            "fps": 60,
            "limit_fps": True,
        },
        {
            "key": "very_fast",
            "label": "V",
            "render_every_n_steps": 5,
            "animate_frames": 1,
            "fps": 60,
            "limit_fps": False,
        },
        {
            "key": "very_very_fast",
            "label": "VI",
            "render_every_n_steps": 25,
            "animate_frames": 1,
            "fps": 240,
            "limit_fps": False,
        },
        {
            "key": "very_very_very_fast",
            "label": "VII",
            "render_every_n_steps": 50,
            "animate_frames": 1,
            "fps": 350,
            "limit_fps": False,
        },
    )

    # Main scene coordinates
    SUPPLIER_CENTER = (115, 300)
    ROAD_START_X = 215
    ROAD_END_X = 555
    ROAD_Y = 310

    WAREHOUSE_RECT = pygame.Rect(580, 165, 320, 325)

    CUSTOMER_CENTER = (1200, 300)

    def __init__(
        self,
        render_mode: str,
        fps: int = 60,
        asset_dir: Optional[str | Path] = None,
        animate_frames: int = 1,
        render_every_n_steps: int = 5,
        limit_fps: bool = False,
        speed_preset: Optional[str] = None,
        show_reward_window: bool = True,
    ):
        if render_mode not in {"human", "rgb_array"}:
            raise ValueError(
                "render_mode must be 'human' or 'rgb_array'"
            )

        self.render_mode = render_mode
        self.fps = max(1, int(fps))

        # Speed controls:
        # - animate_frames=1: only one pygame frame per rendered environment step.
        # - render_every_n_steps=5: draw only every fifth RL step.
        # - limit_fps=False: do not intentionally sleep between frames.
        #
        # History for the top chart is still recorded on EVERY call to render().
        self.animate_frames = max(1, int(animate_frames))
        self.render_every_n_steps = max(1, int(render_every_n_steps))
        self.limit_fps = bool(limit_fps)

        self._speed_index = self._find_matching_speed_preset()
        self._force_redraw = True
        self._render_disabled_by_user = False

        self.speed_button_rect = pygame.Rect(
            self.WIDTH - 335,
            14,
            300,
            52,
        )

        self.speed_down_button_rect = pygame.Rect(
            self.speed_button_rect.left,
            self.speed_button_rect.top,
            48,
            self.speed_button_rect.height,
        )

        self.speed_up_button_rect = pygame.Rect(
            self.speed_button_rect.right - 48,
            self.speed_button_rect.top,
            48,
            self.speed_button_rect.height,
        )

        self.speed_label_rect = pygame.Rect(
            self.speed_down_button_rect.right,
            self.speed_button_rect.top,
            self.speed_button_rect.width - 96,
            self.speed_button_rect.height,
        )

        if speed_preset is not None:
            self.set_speed_preset(speed_preset)

        self._last_drawn_step: Optional[int] = None

        if asset_dir is None:
            asset_dir = Path(__file__).resolve().parent / "assets"

        self.asset_dir = Path(asset_dir)

        self.window: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None

        self._pygame_initialized = False
        self._fonts_initialized = False

        self.font_small: Optional[pygame.font.Font] = None
        self.font: Optional[pygame.font.Font] = None
        self.font_large: Optional[pygame.font.Font] = None
        self.font_title: Optional[pygame.font.Font] = None
        self.font_capacity: Optional[pygame.font.Font] = None

        self.assets: dict[str, Optional[pygame.Surface]] = {}

        self._scaled_package_cache: dict[tuple[int, int], pygame.Surface] = {}

        self._display_progress: dict[Any, float] = {}
        self._shipment_lanes: dict[Any, int] = {}
        self._free_lanes: list[int] = [0, 1, 2, 3]


        self._history = {
            "t": deque(maxlen=self.HISTORY_LENGTH),
            "inventory": deque(maxlen=self.HISTORY_LENGTH),
            "order": deque(maxlen=self.HISTORY_LENGTH),
            "demand": deque(maxlen=self.HISTORY_LENGTH),
            "arrived": deque(maxlen=self.HISTORY_LENGTH),
        }

        self._episode_demand_history: list[tuple[int, int]] = []

        self._last_history_step: Optional[int] = None

        self.show_reward_window = bool(show_reward_window)
        self.reward_window = None
        self.reward_renderer = None
        self._main_sdl2_window = None
        self._reward_window_warning_printed = False
        self._reward_window_closed_by_user = False

        self._reward_values: list[float] = []
        self._reward_moving_average_values: list[float] = []
        self._reward_moving_window: deque[float] = deque()
        self._reward_moving_sum = 0.0
        self._reward_moving_average_period = 1

        self._reward_min = 0.0
        self._reward_max = 0.0
        self._reward_global_step = 0
        self._last_reward_marker: Any = None

        self._ensure_pygame()
        self._load_assets()


    def _ensure_pygame(self) -> None:
        if not self._pygame_initialized:
            pygame.init()
            self._pygame_initialized = True

        if not self._fonts_initialized:
            pygame.font.init()

            self.font_small = pygame.font.Font(None, 22)
            self.font = pygame.font.Font(None, 28)
            self.font_large = pygame.font.Font(None, 34)
            self.font_title = pygame.font.Font(None, 42)
            self.font_capacity = pygame.font.Font(None, 38)

            self._fonts_initialized = True

    def _load_png(
        self,
        filename: str,
        size: tuple[int, int],
    ) -> Optional[pygame.Surface]:
        """
        Load and scale an image if it exists.

        If the PNG does not exist, return None.
        """
        path = self.asset_dir / filename

        if not path.exists():
            return None

        try:
            image = pygame.image.load(str(path))

            if pygame.display.get_surface() is not None:
                image = image.convert_alpha()

            return pygame.transform.smoothscale(image, size)

        except pygame.error as exc:
            print(f"Could not load {path}: {exc}")
            return None

    def _load_assets(self) -> None:
        self.assets = {
            "supplier": self._load_png(
                "supplier.png",
                (165, 145),
            ),
            "warehouse": self._load_png(
                "warehouse.png",
                (
                    self.WAREHOUSE_RECT.width,
                    self.WAREHOUSE_RECT.height,
                ),
            ),
            "truck": self._load_png(
                "truck.png",
                (85, 50),
            ),
            "package": self._load_png(
                "package.png",
                (42, 42),
            ),
            "customer": self._load_png(
                "customer.png",
                (90, 90),
            ),
        }


    def render(
        self,
        env: Any,
        animate: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Render the current environment state.

        The speed button is processed on EVERY environment step, even when
        drawing is currently skipped by `render_every_n_steps`.
        """
        self._record_history(env)
        self._record_reward(env)

        if self.render_mode == "rgb_array":
            canvas = self._draw_frame(env, interpolation=1.0)
            return self._canvas_to_rgb(canvas)

        if self._render_disabled_by_user:
            return None

        close_requested, speed_changed = self._handle_events()

        if close_requested:
            self._disable_human_rendering()
            return None

        step = int(getattr(env, "step_count", 0))

        episode_length = getattr(env, "episode_length", None)
        is_last_step = (
            episode_length is not None
            and step >= int(episode_length)
        )

        should_draw = (
            self._force_redraw
            or step == 0
            or step % self.render_every_n_steps == 0
            or is_last_step
        )

        if not should_draw:
            self._commit_display_progress(env)
            return None

        frames = self.animate_frames if animate else 1

        for frame_index in range(frames):
            interpolation = (frame_index + 1) / frames

            canvas = self._draw_frame(
                env,
                interpolation=interpolation,
            )
            self._show_canvas(canvas)

            close_requested, speed_changed = self._handle_events()

            if close_requested:
                self._disable_human_rendering()
                return None

            if speed_changed:
                break

        self._show_reward_window()

        self._commit_display_progress(env)
        self._last_drawn_step = step
        self._force_redraw = speed_changed
        return None

    def set_speed(
        self,
        *,
        render_every_n_steps: Optional[int] = None,
        animate_frames: Optional[int] = None,
        fps: Optional[int] = None,
        limit_fps: Optional[bool] = None,
    ) -> None:
        """Change the low-level rendering speed variables."""
        if render_every_n_steps is not None:
            self.render_every_n_steps = max(
                1,
                int(render_every_n_steps),
            )

        if animate_frames is not None:
            self.animate_frames = max(
                1,
                int(animate_frames),
            )

        if fps is not None:
            self.fps = max(1, int(fps))

        if limit_fps is not None:
            self.limit_fps = bool(limit_fps)

        self._speed_index = self._find_matching_speed_preset()
        self._force_redraw = True

    def _find_matching_speed_preset(self) -> int:
        """Return exact matching preset, otherwise nearest sensible preset."""
        current = (
            int(self.render_every_n_steps),
            int(self.animate_frames),
            int(self.fps),
            bool(self.limit_fps),
        )

        for index, preset in enumerate(self.SPEED_PRESETS):
            values = (
                int(preset["render_every_n_steps"]),
                int(preset["animate_frames"]),
                int(preset["fps"]),
                bool(preset["limit_fps"]),
            )
            if current == values:
                return index

        best_index = 0
        best_score = float("inf")

        for index, preset in enumerate(self.SPEED_PRESETS):
            score = (
                abs(
                    self.render_every_n_steps
                    - int(preset["render_every_n_steps"])
                ) * 8
                + abs(
                    self.animate_frames
                    - int(preset["animate_frames"])
                ) * 4
                + abs(self.fps - int(preset["fps"])) / 30
                + (
                    0
                    if self.limit_fps == bool(preset["limit_fps"])
                    else 10
                )
            )

            if score < best_score:
                best_score = score
                best_index = index

        return best_index

    def set_speed_preset(self, preset: str | int) -> None:
        """
        Apply one speed preset and update all frame-related variables.

        `preset` may be:
            "very_slow", "slow", "normal", "fast",
            "very_fast", "very_very_fast"
        or an integer preset index.
        """
        if isinstance(preset, int):
            index = max(
                0,
                min(
                    int(preset),
                    len(self.SPEED_PRESETS) - 1,
                ),
            )
        else:
            key = str(preset).strip().lower()

            aliases = {
                "sehr langsam": "very_slow",
                "langsam": "slow",
                "normal": "normal",
                "schnell": "fast",
                "sehr schnell": "very_fast",
                "sehr sehr schnell": "very_very_fast",
                "sehr sehr sehr schnell": "very_very_very_fast",
            }
            key = aliases.get(key, key)

            matches = [
                i
                for i, item in enumerate(self.SPEED_PRESETS)
                if item["key"] == key
            ]

            if not matches:
                valid = ", ".join(
                    item["key"] for item in self.SPEED_PRESETS
                )
                raise ValueError(
                    f"Unknown speed preset {preset!r}. Valid: {valid}"
                )

            index = matches[0]

        item = self.SPEED_PRESETS[index]

        self.render_every_n_steps = int(
            item["render_every_n_steps"]
        )
        self.animate_frames = int(
            item["animate_frames"]
        )
        self.fps = int(item["fps"])
        self.limit_fps = bool(item["limit_fps"])

        self._speed_index = index
        self._force_redraw = True

    def _speed_up(self) -> bool:

        if self._speed_index >= len(
            self.SPEED_PRESETS
        ) - 1:
            return False

        self.set_speed_preset(
            self._speed_index + 1
        )

        return True


    def _speed_down(self) -> bool:

        if self._speed_index <= 0:
            return False

        self.set_speed_preset(
            self._speed_index - 1
        )

        return True

    def _current_speed_preset(self) -> dict[str, Any]:
        return self.SPEED_PRESETS[self._speed_index]

    def _event_window_id(self, event: Any) -> Optional[int]:
        """Normalize pygame WINDOW* event.window to an integer id."""
        value = getattr(event, "window", None)

        if value is None:
            return None

        if isinstance(value, int):
            return value

        return getattr(value, "id", None)

    def _handle_events(self) -> tuple[bool, bool]:
        """
        Returns:
            (main_close_requested, speed_changed)

        Closing the reward window only disables the reward window.
        Closing the main supply-chain window disables human rendering.
        """
        close_requested = False
        speed_changed = False

        if self.window is None:
            return close_requested, speed_changed

        reward_id = (
            getattr(self.reward_window, "id", None)
            if self.reward_window is not None
            else None
        )
        main_id = (
            getattr(self._main_sdl2_window, "id", None)
            if self._main_sdl2_window is not None
            else None
        )

        for event in pygame.event.get():
            event_window_id = self._event_window_id(event)

            if event.type == getattr(pygame, "WINDOWCLOSE", -1):
                if (
                    reward_id is not None
                    and event_window_id == reward_id
                ):
                    self._close_reward_window(
                        closed_by_user=True
                    )
                    continue

                if (
                    main_id is None
                    or event_window_id is None
                    or event_window_id == main_id
                ):
                    close_requested = True
                    continue

            if event.type == pygame.QUIT:
                close_requested = True
                continue

            event_from_reward = (
                reward_id is not None
                and event_window_id == reward_id
            )

            if (
                event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and not event_from_reward
            ):

                if self.speed_down_button_rect.collidepoint(
                    event.pos
                ):

                    if self._speed_down():
                        speed_changed = True

                elif self.speed_up_button_rect.collidepoint(
                    event.pos
                ):

                    if self._speed_up():
                        speed_changed = True

            elif (
                event.type == pygame.KEYDOWN
                and not event_from_reward
            ):

                if event.key == pygame.K_RIGHT:

                    if self._speed_up():
                        speed_changed = True

                elif event.key == pygame.K_LEFT:

                    if self._speed_down():
                        speed_changed = True

        return close_requested, speed_changed

    def _disable_human_rendering(self) -> None:
        """Closing the main window stops visualization but lets RL training continue."""
        self._render_disabled_by_user = True

        self._close_reward_window(
            closed_by_user=False
        )

        if self.window is not None:
            pygame.display.quit()
            self.window = None

        self._main_sdl2_window = None
        self.clock = None

    def close(self) -> None:
        self._close_reward_window(
            closed_by_user=False
        )

        if self.window is not None:
            pygame.display.quit()
            self.window = None

        self._main_sdl2_window = None

        pygame.quit()

        self.clock = None
        self._render_disabled_by_user = True
        self._pygame_initialized = False
        self._fonts_initialized = False


    def _get_reward_ma_period(
        self,
        env: Any,
    ) -> int:

        return max(
            1,
            int(
                getattr(
                    env,
                    "episode_length",
                    1,
                )
            ),
        )

    def _extract_reward(self, env: Any) -> Optional[float]:
        """
        Preferred sources:
            env.last_reward
            env.last_info["reward"]

        As a fallback, reconstruct the current reward from the cost/revenue
        fields already present in this environment's info dictionary.
        """
        direct = getattr(env, "last_reward", None)

        if direct is not None:
            try:
                return float(direct)
            except (TypeError, ValueError):
                pass

        info = getattr(env, "last_info", {}) or {}

        if "reward" in info:
            try:
                return float(info["reward"])
            except (TypeError, ValueError):
                return None

        if not info or "sold" not in info:
            return None

        revenue = float(info.get("revenue", 0.0))
        ordering_cost = float(info.get("ordering_cost", 0.0))
        holding_cost = float(info.get("holding_cost", 0.0))
        overflow_cost = float(info.get("overflow_cost", 0.0))
        lost_sales_cost = float(info.get("lost_sales_cost", 0.0))

        return (
            revenue
            - ordering_cost
            - holding_cost
            - overflow_cost
            - lost_sales_cost
        )

    def _reward_marker(self, env: Any) -> Any:
        """
        Prevent recording the same RL step several times during animation.

        Best option:
            env.total_training_steps   (monotonic, never reset)

        Fallback:
            identity of last_info, because this env creates a fresh info dict
            for every completed step.
        """
        total_steps = getattr(
            env,
            "total_training_steps",
            None,
        )

        if total_steps is not None:
            return ("global", int(total_steps))

        info = getattr(env, "last_info", {}) or {}

        if not info:
            return None

        return ("info", id(info))

    def _record_reward(
        self,
        env: Any,
    ) -> None:

        reward = self._extract_reward(env)

        if reward is None or not np.isfinite(reward):
            return

        marker = self._reward_marker(env)

        if (
            marker is None
            or marker == self._last_reward_marker
        ):
            return

        self._last_reward_marker = marker
        self._reward_global_step += 1

        reward = float(reward)

        self._reward_values.append(
            reward
        )


        ma_period = self._get_reward_ma_period(
            env
        )

        self._reward_moving_average_period = (
            ma_period
        )

        self._reward_moving_window.append(
            reward
        )

        self._reward_moving_sum += reward

        while (
            len(self._reward_moving_window)
            > ma_period
        ):

            removed_reward = (
                self._reward_moving_window.popleft()
            )

            self._reward_moving_sum -= (
                removed_reward
            )

        moving_average = (
            self._reward_moving_sum
            / len(self._reward_moving_window)
        )

        self._reward_moving_average_values.append(
            float(moving_average)
        )

        if self._reward_global_step == 1:

            self._reward_min = reward
            self._reward_max = reward

        else:

            self._reward_min = min(
                self._reward_min,
                reward,
            )

            self._reward_max = max(
                self._reward_max,
                reward,
            )

    def _ensure_reward_window(self) -> bool:
        if (
            not self.show_reward_window
            or self.render_mode != "human"
            or self._reward_window_closed_by_user
        ):
            return False

        if not SDL2_MULTIWINDOW_AVAILABLE:
            if not self._reward_window_warning_printed:
                print(
                    "Reward window disabled: pygame._sdl2.video "
                    "is not available in this pygame installation."
                )
                self._reward_window_warning_printed = True
            return False

        if self.reward_window is not None:
            return True

        try:
            self.reward_window = SDL2Window(
                title="PPO Training Reward",
                size=(
                    self.REWARD_WINDOW_WIDTH,
                    self.REWARD_WINDOW_HEIGHT,
                ),
                position=(80, 80),
                resizable=True,
            )

            self.reward_renderer = SDL2Renderer(
                self.reward_window,
                accelerated=True,
                vsync=False,
            )
            return True

        except Exception as exc:
            self.reward_window = None
            self.reward_renderer = None

            if not self._reward_window_warning_printed:
                print(
                    "Could not create separate reward window: "
                    f"{exc}"
                )
                self._reward_window_warning_printed = True

            return False

    def _close_reward_window(
        self,
        *,
        closed_by_user: bool,
    ) -> None:
        self.reward_renderer = None

        if self.reward_window is not None:
            try:
                self.reward_window.destroy()
            except Exception:
                pass

        self.reward_window = None

        if closed_by_user:
            self._reward_window_closed_by_user = True

    def _show_reward_window(self) -> None:
        if not self._ensure_reward_window():
            return

        if self.reward_renderer is None:
            return

        try:
            canvas = self._draw_reward_canvas()

            texture = SDL2Texture.from_surface(
                self.reward_renderer,
                canvas,
            )

            self.reward_renderer.clear()
            texture.draw()
            self.reward_renderer.present()

        except Exception as exc:
            self._close_reward_window(
                closed_by_user=False
            )

            if not self._reward_window_warning_printed:
                print(
                    "Reward window rendering failed: "
                    f"{exc}"
                )
                self._reward_window_warning_printed = True

    def _draw_reward_canvas(self) -> pygame.Surface:
        width = self.REWARD_WINDOW_WIDTH
        height = self.REWARD_WINDOW_HEIGHT

        canvas = pygame.Surface((width, height))
        canvas.fill((242, 245, 247))

        title = self.font_large.render(
            "Training reward",
            True,
            (28, 34, 39),
        )
        canvas.blit(title, (26, 20))

        n = len(self._reward_values)

        if n == 0:
            empty = self.font.render(
                "Waiting for the first completed environment step ...",
                True,
                (90, 98, 104),
            )
            canvas.blit(empty, (26, 78))
            return canvas

        current = self._reward_values[-1]
        moving = self._reward_moving_average_values[-1]

        metrics = [
            f"Training step: {self._reward_global_step}",
            f"Reward: {current:.2f}",
            (
                f"Moving Episode Average: "
                f"{moving:.2f}"
            ),
        ]

        x = 28
        for text_value in metrics:
            text = self.font_small.render(
                text_value,
                True,
                (66, 74, 80),
            )
            canvas.blit(text, (x, 60))
            x += text.get_width() + 32

        plot = pygame.Rect(
            72,
            105,
            width - 105,
            height - 165,
        )

        pygame.draw.rect(
            canvas,
            (251, 252, 253),
            plot,
            border_radius=8,
        )
        pygame.draw.rect(
            canvas,
            (195, 202, 207),
            plot,
            width=1,
            border_radius=8,
        )


        Y_MIN_LIMIT = -1000.0

        y_min = max(
            Y_MIN_LIMIT,
            min(
                self._reward_min,
                0.0,
            ),
        )

        y_max = max(
            self._reward_max,
            0.0,
        )

        if abs(y_max - y_min) < 1e-9:

            margin = max(
                1.0,
                abs(y_max) * 0.1,
            )

            y_min = max(
                Y_MIN_LIMIT,
                y_min - margin,
            )

            y_max += margin

        else:

            margin = (
                0.08
                * (y_max - y_min)
            )

            y_min = max(
                Y_MIN_LIMIT,
                y_min - margin,
            )

            y_max += margin

        def y_to_screen(value: float) -> int:
            ratio = (value - y_min) / (y_max - y_min)
            return int(
                plot.bottom
                - np.clip(ratio, 0.0, 1.0) * plot.height
            )

        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            value = y_min + (y_max - y_min) * fraction
            y = y_to_screen(value)

            pygame.draw.line(
                canvas,
                (226, 230, 233),
                (plot.left, y),
                (plot.right, y),
                width=1,
            )

            label = self.font_small.render(
                f"{value:.1f}",
                True,
                (95, 102, 108),
            )
            canvas.blit(
                label,
                (
                    plot.left - label.get_width() - 8,
                    y - label.get_height() // 2,
                ),
            )

        if y_min <= 0.0 <= y_max:
            zero_y = y_to_screen(0.0)
            pygame.draw.line(
                canvas,
                (135, 142, 148),
                (plot.left, zero_y),
                (plot.right, zero_y),
                width=2,
            )

        max_draw_points = max(200, plot.width)
        stride = max(1, n // max_draw_points)

        indices = list(range(0, n, stride))
        if indices[-1] != n - 1:
            indices.append(n - 1)

        denominator = max(1, n - 1)

        reward_points = []
        average_points = []

        for index in indices:
            px = int(
                plot.left
                + (index / denominator) * plot.width
            )

            reward_points.append(
                (
                    px,
                    y_to_screen(
                        self._reward_values[index]
                    ),
                )
            )

            average_points.append(
                (
                    px,
                    y_to_screen(
                        self._reward_moving_average_values[index]
                    ),
                )
            )

        if len(reward_points) >= 2:
            pygame.draw.lines(
                canvas,
                (125, 139, 151),
                False,
                reward_points,
                width=1,
            )

        if len(average_points) >= 2:
            pygame.draw.lines(
                canvas,
                (48, 111, 176),
                False,
                average_points,
                width=3,
            )

        first_step = self.font_small.render(
            "1",
            True,
            (95, 102, 108),
        )
        last_step = self.font_small.render(
            str(self._reward_global_step),
            True,
            (95, 102, 108),
        )

        canvas.blit(
            first_step,
            (
                plot.left,
                plot.bottom + 8,
            ),
        )
        canvas.blit(
            last_step,
            (
                plot.right - last_step.get_width(),
                plot.bottom + 8,
            ),
        )

        axis = self.font_small.render(
            "training step",
            True,
            (95, 102, 108),
        )
        canvas.blit(
            axis,
            (
                plot.centerx - axis.get_width() // 2,
                plot.bottom + 8,
            ),
        )

        legend_y = height - 30

        pygame.draw.line(
            canvas,
            (125, 139, 151),
            (28, legend_y),
            (54, legend_y),
            width=2,
        )
        raw_label = self.font_small.render(
            "Step reward",
            True,
            (65, 72, 78),
        )
        canvas.blit(
            raw_label,
            (62, legend_y - 9),
        )

        second_x = 175
        pygame.draw.line(
            canvas,
            (48, 111, 176),
            (second_x, legend_y),
            (second_x + 26, legend_y),
            width=3,
        )
        ma_label = self.font_small.render(
            f"Moving episode average",
            True,
            (65, 72, 78),
        )
        canvas.blit(
            ma_label,
            (second_x + 34, legend_y - 9),
        )

        return canvas

    def _draw_frame(
        self,
        env: Any,
        interpolation: float,
    ) -> pygame.Surface:
        canvas = pygame.Surface(
            (self.WIDTH, self.HEIGHT)
        )
        canvas.fill((238, 242, 245))

        # Top title and rolling history graph.
        self._draw_header(canvas, env)
        self._draw_history_graph(canvas, env)


        scene = pygame.Surface(
            (self.WIDTH, self.SCENE_HEIGHT)
        )
        scene.fill((238, 242, 245))

        self._draw_supplier(scene)
        self._draw_transport_route(scene)
        self._draw_shipments(
            scene,
            env,
            interpolation=interpolation,
        )

        self._draw_warehouse(scene, env)

        self._draw_outbound_route(scene)
        self._draw_outgoing_goods(
            scene,
            env,
            interpolation=interpolation,
        )

        self._draw_customer_area(scene, env)
        self._draw_kpi_panel(scene, env)
        self._draw_episode_demand_graph(scene, env)

        canvas.blit(scene, (0, self.SCENE_OFFSET_Y))
        return canvas


    def _record_history(self, env: Any) -> None:
        """Record exactly one point per environment period."""
        step = int(getattr(env, "step_count", 0))

        if (
            self._last_history_step is not None
            and step < self._last_history_step
        ):
            for values in self._history.values():
                values.clear()
            self._episode_demand_history.clear()
            self._last_history_step = None

        if step == self._last_history_step:
            return

        info = getattr(env, "last_info", {}) or {}

        inventory = int(getattr(env, "inventory", 0))

        if step == 0 or not info:
            demand = int(getattr(env, "demand", 0))
            order = 0
            arrived = 0
        else:
            demand = int(
                info.get(
                    "current_demand",
                    getattr(env, "demand", 0),
                )
            )
            order = int(info.get("ordered_at_period_end", 0))
            arrived = int(info.get("arriving_before_order", 0))

        self._history["t"].append(step)
        self._history["inventory"].append(inventory)
        self._history["order"].append(order)
        self._history["demand"].append(demand)
        self._history["arrived"].append(arrived)

        self._episode_demand_history.append((step, demand))

        self._last_history_step = step

    def _draw_history_graph(
        self,
        canvas: pygame.Surface,
        env: Any,
    ) -> None:

        assert self.font is not None
        assert self.font_small is not None

        outer = pygame.Rect(
            45,
            78,
            self.WIDTH - 90,
            242,
        )

        pygame.draw.rect(
            canvas,
            (250, 251, 252),
            outer,
            border_radius=10,
        )

        pygame.draw.rect(
            canvas,
            (199, 206, 211),
            outer,
            width=1,
            border_radius=10,
        )

        title = self.font.render(
            f"Inventory & flows — last {self.HISTORY_LENGTH} periods",
            True,
            (35, 41, 46),
        )

        canvas.blit(
            title,
            (
                outer.left + 14,
                outer.top + 8,
            ),
        )

        legend_y = outer.top + 14
        legend_x = outer.right - 520

        legend = [
            (
                "Inventory",
                (52, 108, 176),
                "line",
            ),
            (
                "Order",
                (95, 154, 94),
                "bar",
            ),
            (
                "Demand",
                (205, 119, 73),
                "bar",
            ),
            (
                "Arrived",
                (120, 92, 166),
                "bar",
            ),
        ]

        for name, color, kind in legend:

            if kind == "line":

                pygame.draw.line(
                    canvas,
                    color,
                    (
                        legend_x,
                        legend_y + 7,
                    ),
                    (
                        legend_x + 22,
                        legend_y + 7,
                    ),
                    width=3,
                )

            else:

                pygame.draw.rect(
                    canvas,
                    color,
                    (
                        legend_x + 5,
                        legend_y + 1,
                        12,
                        13,
                    ),
                    border_radius=2,
                )

            txt = self.font_small.render(
                name,
                True,
                (65, 72, 78),
            )

            canvas.blit(
                txt,
                (
                    legend_x + 28,
                    legend_y - 2,
                ),
            )

            legend_x += (
                txt.get_width() + 62
            )

        plot_left = outer.left + 55
        plot_right = outer.right - 18

        plot_top = outer.top + 48
        plot_bottom = outer.bottom - 32

        plot_width = (
            plot_right - plot_left
        )

        plot_height = (
            plot_bottom - plot_top
        )

        t_values = list(
            self._history["t"]
        )

        inventory_values = list(
            self._history["inventory"]
        )

        order_values = list(
            self._history["order"]
        )

        demand_values = list(
            self._history["demand"]
        )

        arrived_values = list(
            self._history["arrived"]
        )

        if not t_values:
            return

        n = len(t_values)

        capacity = max(
            1,
            int(
                getattr(
                    env,
                    "capacity",
                    1,
                )
            ),
        )

        max_inventory = max(
            inventory_values,
            default=0,
        )

        max_order = max(
            order_values,
            default=0,
        )

        max_demand = max(
            demand_values,
            default=0,
        )

        max_arrived = max(
            arrived_values,
            default=0,
        )

        y_max = max(
            1,
            capacity,
            max_inventory,
            max_order,
            max_demand,
            max_arrived,
        )

        y_min = 0

        def value_to_y(
            value: float,
        ) -> int:

            ratio = (
                float(value) - y_min
            ) / (
                y_max - y_min
            )

            ratio = float(
                np.clip(
                    ratio,
                    0.0,
                    1.0,
                )
            )

            return int(
                plot_bottom
                - ratio * plot_height
            )

        if n == 1:

            x_positions = [
                plot_left
                + plot_width // 2
            ]

        else:

            x_positions = [
                plot_left
                + int(
                    i
                    * plot_width
                    / (n - 1)
                )
                for i in range(n)
            ]

        tick_fractions = (
            0.0,
            0.25,
            0.50,
            0.75,
            1.0,
        )

        for fraction in tick_fractions:

            value = (
                y_min
                + fraction
                * (y_max - y_min)
            )

            y = value_to_y(
                value
            )

            pygame.draw.line(
                canvas,
                (226, 231, 234),
                (
                    plot_left,
                    y,
                ),
                (
                    plot_right,
                    y,
                ),
                width=1,
            )

            tick = self.font_small.render(
                str(
                    int(
                        round(value)
                    )
                ),
                True,
                (95, 102, 108),
            )

            canvas.blit(
                tick,
                (
                    plot_left
                    - tick.get_width()
                    - 8,

                    y
                    - tick.get_height()
                    // 2,
                ),
            )

        pygame.draw.line(
            canvas,
            (160, 168, 174),
            (
                plot_left,
                plot_top,
            ),
            (
                plot_left,
                plot_bottom,
            ),
            width=1,
        )

        pygame.draw.line(
            canvas,
            (160, 168, 174),
            (
                plot_left,
                plot_bottom,
            ),
            (
                plot_right,
                plot_bottom,
            ),
            width=1,
        )

        point_spacing = (
            plot_width
            / max(
                n - 1,
                1,
            )
        )

        bar_width = max(
            2,
            min(
                6,
                int(
                    point_spacing / 5
                ),
            ),
        )

        offsets = (
            -bar_width - 1,
            0,
            bar_width + 1,
        )

        colors = (
            (95, 154, 94),   # order
            (205, 119, 73),  # demand
            (120, 92, 166),  # arrived
        )

        for i, x in enumerate(
            x_positions
        ):

            values = (
                order_values[i],
                demand_values[i],
                arrived_values[i],
            )

            for (
                value,
                offset,
                color,
            ) in zip(
                values,
                offsets,
                colors,
            ):

                if value <= 0:
                    continue

                top_y = value_to_y(
                    value
                )

                height = (
                    plot_bottom
                    - top_y
                )

                if height <= 0:
                    continue

                rect = pygame.Rect(
                    x
                    + offset
                    - bar_width // 2,

                    top_y,

                    bar_width,

                    height,
                )

                pygame.draw.rect(
                    canvas,
                    color,
                    rect,
                )

        inventory_points = []

        for x, inventory in zip(
            x_positions,
            inventory_values,
        ):

            y = value_to_y(
                inventory
            )

            inventory_points.append(
                (
                    x,
                    y,
                )
            )

        if len(
            inventory_points
        ) >= 2:

            pygame.draw.lines(
                canvas,
                (52, 108, 176),
                False,
                inventory_points,
                width=3,
            )

        elif inventory_points:

            pygame.draw.circle(
                canvas,
                (52, 108, 176),
                inventory_points[0],
                4,
            )

        if inventory_points:

            pygame.draw.circle(
                canvas,
                (52, 108, 176),
                inventory_points[-1],
                4,
            )

        tick_indices = sorted(
            set(
                [0, n - 1]
                + [
                    int(
                        round(
                            i
                            * (n - 1)
                            / 5
                        )
                    )
                    for i in range(
                        1,
                        5,
                    )
                ]
            )
        )

        for idx in tick_indices:

            x = x_positions[idx]

            pygame.draw.line(
                canvas,
                (185, 192, 198),
                (
                    x,
                    plot_bottom,
                ),
                (
                    x,
                    plot_bottom + 4,
                ),
                width=1,
            )

            label = (
                self.font_small.render(
                    str(
                        t_values[idx]
                    ),
                    True,
                    (90, 97, 103),
                )
            )

            canvas.blit(
                label,
                (
                    x
                    - label.get_width()
                    // 2,

                    plot_bottom + 5,
                ),
            )

        axis_label = (
            self.font_small.render(
                "time t",
                True,
                (75, 82, 88),
            )
        )

        canvas.blit(
            axis_label,
            (
                plot_right
                - axis_label.get_width(),

                outer.bottom - 18,
            ),
        )

    def _draw_header(
        self,
        canvas: pygame.Surface,
        env: Any,
    ) -> None:
        assert self.font_title is not None
        assert self.font is not None

        title = self.font_title.render(
            "Supply Chain Simulator",
            True,
            (25, 31, 36),
        )
        canvas.blit(title, (35, 22))

        period = self.font.render(
            f"Day {getattr(env, 'step_count', 0)}",
            True,
            (70, 78, 84),
        )
        canvas.blit(
            period,
            (
                self.speed_button_rect.left
                - period.get_width()
                - 22,
                31,
            ),
        )

        self._draw_speed_button(canvas)

    def _draw_speed_button(
        self,
        canvas: pygame.Surface,
    ) -> None:

        assert self.font is not None
        assert self.font_small is not None

        preset = self._current_speed_preset()

        pygame.draw.rect(
            canvas,
            (246, 248, 249),
            self.speed_button_rect,
            border_radius=9,
        )

        pygame.draw.rect(
            canvas,
            (151, 160, 168),
            self.speed_button_rect,
            width=1,
            border_radius=9,
        )


        can_slow_down = self._speed_index > 0

        slow_bg = (
            (238, 241, 243)
            if can_slow_down
            else (225, 228, 230)
        )

        pygame.draw.rect(
            canvas,
            slow_bg,
            self.speed_down_button_rect,
            border_radius=9,
        )

        pygame.draw.line(
            canvas,
            (180, 186, 191),
            (
                self.speed_down_button_rect.right,
                self.speed_down_button_rect.top + 5,
            ),
            (
                self.speed_down_button_rect.right,
                self.speed_down_button_rect.bottom - 5,
            ),
            width=1,
        )

        minus_color = (
            (35, 42, 47)
            if can_slow_down
            else (150, 155, 160)
        )

        minus = self.font.render(
            "−",
            True,
            minus_color,
        )

        canvas.blit(
            minus,
            (
                self.speed_down_button_rect.centerx
                - minus.get_width() // 2,
                self.speed_down_button_rect.centery
                - minus.get_height() // 2,
            ),
        )


        can_speed_up = (
            self._speed_index
            < len(self.SPEED_PRESETS) - 1
        )

        fast_bg = (
            (238, 241, 243)
            if can_speed_up
            else (225, 228, 230)
        )

        pygame.draw.rect(
            canvas,
            fast_bg,
            self.speed_up_button_rect,
            border_radius=9,
        )

        pygame.draw.line(
            canvas,
            (180, 186, 191),
            (
                self.speed_up_button_rect.left,
                self.speed_up_button_rect.top + 5,
            ),
            (
                self.speed_up_button_rect.left,
                self.speed_up_button_rect.bottom - 5,
            ),
            width=1,
        )

        plus_color = (
            (35, 42, 47)
            if can_speed_up
            else (150, 155, 160)
        )

        plus = self.font.render(
            "+",
            True,
            plus_color,
        )

        canvas.blit(
            plus,
            (
                self.speed_up_button_rect.centerx
                - plus.get_width() // 2,
                self.speed_up_button_rect.centery
                - plus.get_height() // 2,
            ),
        )


        label = self.font.render(
            f"Speed: {preset['label']}",
            True,
            (35, 42, 47),
        )

        canvas.blit(
            label,
            (
                self.speed_label_rect.centerx
                - label.width // 2,
                self.speed_label_rect.centery - label.height // 2,
            ),
        )



    def _draw_supplier(
        self,
        canvas: pygame.Surface,
    ) -> None:
        assert self.font_large is not None

        image = self.assets.get("supplier")

        rect = image.get_rect(
            center=self.SUPPLIER_CENTER
        )
        canvas.blit(image, rect)


    def _draw_transport_route(
        self,
        canvas: pygame.Surface,
    ) -> None:
        pygame.draw.line(
            canvas,
            (94, 101, 107),
            (self.ROAD_START_X, self.ROAD_Y),
            (self.ROAD_END_X, self.ROAD_Y),
            width=16,
        )

        dash_width = 30
        gap = 22
        x = self.ROAD_START_X + 15

        while x < self.ROAD_END_X - dash_width:
            pygame.draw.line(
                canvas,
                (225, 225, 220),
                (x, self.ROAD_Y),
                (x + dash_width, self.ROAD_Y),
                width=3,
            )
            x += dash_width + gap

        arrow_x = self.ROAD_END_X - 10

        pygame.draw.polygon(
            canvas,
            (94, 101, 107),
            [
                (arrow_x, self.ROAD_Y - 23),
                (arrow_x + 30, self.ROAD_Y),
                (arrow_x, self.ROAD_Y + 23),
            ],
        )

    def _shipment_id(
        self,
        shipment: dict[str, Any],
        index: int,
    ) -> Any:

        if "id" not in shipment:
            raise KeyError(
                "Shipment is missing a unique 'id'. "
                "Every shipment must get an id "
                "when it is created."
            )

        return shipment["id"]

    def _target_progress(
        self,
        env: Any,
        shipment: dict[str, Any],
    ) -> float:
        """
        0.0 = supplier
        1.0 = warehouse
        """
        remaining = float(
            max(0, shipment.get("remaining", 0))
        )

        initial = float(
            shipment.get(
                "initial_lead_time",
                getattr(env, "max_lead_time", 1),
            )
        )

        initial = max(initial, 1.0)

        progress = 1.0 - (remaining / initial)

        return float(
            np.clip(progress, 0.0, 1.0)
        )

    def _draw_shipments(
        self,
        canvas: pygame.Surface,
        env: Any,
        interpolation: float = 1.0,
    ) -> None:
        shipments = list(
            getattr(env, "_shipments", [])
        )

        shipment_ids = [
            self._shipment_id(shipment, index)
            for index, shipment in enumerate(shipments)
        ]

        if len(shipment_ids) != len(set(shipment_ids)):
            raise ValueError(
                f"Duplicate shipment IDs detected: {shipment_ids}"
            )

        active_ids = set(shipment_ids)

        self._release_unused_lanes(active_ids)

        ordered_shipments = sorted(
            zip(shipment_ids, shipments),
            key=lambda pair: pair[0],
        )

        lane_offsets = self._lane_offsets()

        for shipment_id, shipment in ordered_shipments:
            target_progress = self._target_progress(
                env,
                shipment,
            )

            previous_progress = self._display_progress.get(
                shipment_id,
                target_progress,
            )

            progress = (
                previous_progress
                + (target_progress - previous_progress)
                * interpolation
            )

            x = int(
                self.ROAD_START_X
                + (self.ROAD_END_X - self.ROAD_START_X)
                * progress
            )

            lane = self._get_lane_for_shipment(
                shipment_id
            )
            y = self.ROAD_Y + lane_offsets[lane]

            qty = int(shipment.get("qty", 0))

            self._draw_transport_unit(
                canvas=canvas,
                x=x,
                y=y,
                qty=qty,
                shipment_id=shipment_id,
            )

    def _draw_transport_unit(
        self,
        canvas: pygame.Surface,
        x: int,
        y: int,
        qty: int,
        shipment_id: Any,
    ) -> None:
        assert self.font_small is not None

        truck = self.assets.get("truck")

        if truck is not None:

            rect = truck.get_rect(
                center=(x, y)
            )

            canvas.blit(
                truck,
                rect,
            )

            label_y = rect.top - 26
        else:

            body = pygame.Rect(
                x - 38,
                y - 17,
                50,
                28,
            )

            pygame.draw.rect(
                canvas,
                (86, 102, 116),
                body,
                border_radius=4,
            )
            cab = pygame.Rect(
                x + 12,
                y - 11,
                24,
                22,
            )

            pygame.draw.rect(
                canvas,
                (105, 120, 133),
                cab,
                border_radius=4,
            )

            pygame.draw.circle(
                canvas,
                (40, 45, 50),
                (x - 24, y + 14),
                6,
            )

            pygame.draw.circle(
                canvas,
                (40, 45, 50),
                (x + 23, y + 14),
                6,
            )

            label_y = y - 48

        label = self.font_small.render(
            f"{qty}x",
            True,
            (35, 40, 45),
        )

        badge_rect = pygame.Rect(
            x - label.get_width() // 2 - 6,
            label_y - 2,
            label.get_width() + 12,
            label.get_height() + 4,
        )

        pygame.draw.rect(
            canvas,
            (250, 250, 248),
            badge_rect,
            border_radius=6,
        )

        pygame.draw.rect(
            canvas,
            (205, 211, 215),
            badge_rect,
            width=1,
            border_radius=6,
        )

        canvas.blit(
            label,
            (
                x - label.get_width() // 2,
                label_y,
            ),
        )

    def _lane_offsets(self) -> list[int]:
        return [-100, -55, 45, 90]


    def _get_lane_for_shipment(
        self,
        shipment_id: Any,
    ) -> int:
        if shipment_id in self._shipment_lanes:
            return self._shipment_lanes[shipment_id]

        if self._free_lanes:
            lane = self._free_lanes.pop(0)
        else:
            lane = hash(shipment_id) % 4

        self._shipment_lanes[shipment_id] = lane
        return lane


    def _release_unused_lanes(
        self,
        active_shipment_ids: set[Any],
    ) -> None:
        to_remove = []

        for shipment_id, lane in self._shipment_lanes.items():
            if shipment_id not in active_shipment_ids:
                to_remove.append((shipment_id, lane))

        for shipment_id, lane in to_remove:
            del self._shipment_lanes[shipment_id]

            if lane not in self._free_lanes:
                self._free_lanes.append(lane)

        self._free_lanes.sort()    

    def _commit_display_progress(
        self,
        env: Any,
    ) -> None:
        shipments = list(
            getattr(env, "_shipments", [])
        )

        new_progress: dict[Any, float] = {}

        for index, shipment in enumerate(shipments):
            shipment_id = self._shipment_id(
                shipment,
                index,
            )

            new_progress[shipment_id] = (
                self._target_progress(
                    env,
                    shipment,
                )
            )

        self._display_progress = new_progress

    def _draw_warehouse(
        self,
        canvas: pygame.Surface,
        env: Any,
    ) -> None:
        assert self.font_large is not None
        assert self.font_small is not None
        assert self.font_capacity is not None

        rect = self.WAREHOUSE_RECT
        inventory = int(getattr(env, "inventory", 0))
        capacity = int(getattr(env, "capacity", 0))

        capacity_text = self.font_capacity.render(
            f"{inventory} / {capacity}",
            True,
            (30, 37, 42),
        )
        canvas.blit(
            capacity_text,
            (
                rect.centerx - capacity_text.get_width() // 2,
                rect.top - 107,
            ),
        )

        utilization = inventory / capacity if capacity > 0 else 0.0
        util_text = self.font_small.render(
            f"Utilization: {utilization:.0%}",
            True,
            (75, 82, 88),
        )
        canvas.blit(
            util_text,
            (
                rect.centerx - util_text.get_width() // 2,
                rect.top - 75,
            ),
        )

        warehouse_image = self.assets.get("warehouse")

        if warehouse_image is not None:
            canvas.blit(warehouse_image, rect)
            storage_rect = rect.inflate(-56, -82)

            overlay = pygame.Surface(
                (storage_rect.width, storage_rect.height),
                pygame.SRCALPHA,
            )
            overlay.fill((245, 247, 248, 190))
            canvas.blit(overlay, storage_rect.topleft)
        else:
            pygame.draw.rect(
                canvas,
                (84, 94, 104),
                rect,
                border_radius=8,
            )
            pygame.draw.polygon(
                canvas,
                (68, 78, 86),
                [
                    (rect.left - 8, rect.top + 5),
                    (rect.centerx, rect.top - 50),
                    (rect.right + 8, rect.top + 5),
                ],
            )
            storage_rect = rect.inflate(-56, -72)
            pygame.draw.rect(
                canvas,
                (239, 241, 242),
                storage_rect,
            )

        self._draw_warehouse_boxes(
            canvas,
            storage_rect,
            inventory,
            capacity,
        )



    def _draw_warehouse_boxes(
        self,
        canvas: pygame.Surface,
        storage_rect: pygame.Rect,
        inventory: int,
        capacity: int,
    ) -> None:
        if capacity <= 0:
            return

        cols = 6
        rows = 6
        slots = cols * rows

        ratio = float(np.clip(inventory / capacity, 0.0, 1.0))
        filled_slots = int(round(slots * ratio))

        gap = 5
        box_w = (storage_rect.width - gap * (cols + 1)) / cols
        box_h = (storage_rect.height - gap * (rows + 1)) / rows

        package_img = self.assets.get("package")

        for slot in range(filled_slots):
            col = slot % cols
            row_from_bottom = slot // cols

            x = int(storage_rect.left + gap + col * (box_w + gap))
            y = int(
                storage_rect.bottom
                - gap
                - (row_from_bottom + 1) * box_h
                - row_from_bottom * gap
            )

            box_rect = pygame.Rect(
                x,
                y,
                int(box_w),
                int(box_h),
            )

            if package_img is not None:
                size = (box_rect.width, box_rect.height)

                box_img = self._scaled_package_cache.get(size)
                if box_img is None:
                    box_img = pygame.transform.smoothscale(
                        package_img,
                        size,
                    )
                    self._scaled_package_cache[size] = box_img

                canvas.blit(box_img, box_rect)
            else:
                pygame.draw.rect(
                    canvas,
                    (181, 128, 75),
                    box_rect,
                    border_radius=3,
                )
                pygame.draw.rect(
                    canvas,
                    (118, 81, 49),
                    box_rect,
                    width=2,
                    border_radius=3,
                )
                pygame.draw.line(
                    canvas,
                    (224, 188, 130),
                    (box_rect.centerx, box_rect.top + 2),
                    (box_rect.centerx, box_rect.bottom - 2),
                    width=max(1, box_rect.width // 12),
                )


    def _draw_outbound_route(
        self,
        canvas: pygame.Surface,
    ) -> None:
        start_x = self.WAREHOUSE_RECT.right + 18
        end_x = 1140
        y = self.ROAD_Y

        pygame.draw.line(
            canvas,
            (94, 101, 107),
            (start_x, y),
            (end_x, y),
            width=14,
        )

        x = start_x + 10
        while x < end_x - 25:
            pygame.draw.line(
                canvas,
                (225, 225, 220),
                (x, y),
                (x + 22, y),
                width=3,
            )
            x += 40

        pygame.draw.polygon(
            canvas,
            (94, 101, 107),
            [
                (end_x - 5, y - 16),
                (end_x + 19, y),
                (end_x - 5, y + 16),
            ],
        )

    def _draw_outgoing_goods(
        self,
        canvas: pygame.Surface,
        env: Any,
        interpolation: float,
    ) -> None:
        assert self.font is not None
        assert self.font_small is not None

        info = getattr(env, "last_info", {})
        sold = int(info.get("sold", 0))
        lost_sales = int(info.get("lost_sales", 0))

        start_x = self.WAREHOUSE_RECT.right + 20
        end_x = 1150
        y = self.ROAD_Y - 48


        if sold > 0:
            progress = float(np.clip(interpolation, 0.0, 1.0))
            x = int(start_x + (end_x - start_x) * progress)

            package = self.assets.get("package")
            truck = self.assets.get("truck")

            if truck is not None:
                rect = truck.get_rect(center=(x, y))
                canvas.blit(truck, rect)
                label_y = rect.top - 24
            elif package is not None:
                rect = package.get_rect(center=(x, y))
                canvas.blit(package, rect)
                label_y = rect.top - 24
            else:
                rect = pygame.Rect(x - 25, y - 19, 50, 38)
                pygame.draw.rect(
                    canvas,
                    (181, 128, 75),
                    rect,
                    border_radius=3,
                )
                pygame.draw.rect(
                    canvas,
                    (118, 81, 49),
                    rect,
                    width=2,
                    border_radius=3,
                )
                label_y = rect.top - 24

            qty = self.font_small.render(
                f"x{sold}",
                True,
                (35, 40, 45),
            )
            canvas.blit(
                qty,
                (x - qty.get_width() // 2, label_y),
            )


        if lost_sales > 0:
            lost = self.font_small.render(
                f"Lost sales: {lost_sales}",
                True,
                (160, 58, 58),
            )
            canvas.blit(
                lost,
                (start_x + 55, self.ROAD_Y + 45),
            )

    def _draw_customer_area(
        self,
        canvas: pygame.Surface,
        env: Any,
    ) -> None:
        assert self.font is not None
        assert self.font_small is not None

        customer = self.assets.get(
            "customer"
        )

        if customer is not None:
            rect = customer.get_rect(
                center=self.CUSTOMER_CENTER
            )
            canvas.blit(customer, rect)

        demand = int(
            getattr(env, "demand", 0)
        )

        demand_text = self.font.render(
            f"Demand: {demand}",
            True,
            (35, 40, 45),
        )

        canvas.blit(
            demand_text,
            (
                self.CUSTOMER_CENTER[0]
                - demand_text.get_width() // 2,
                390,
            ),
        )


    def _draw_kpi_panel(
        self,
        canvas: pygame.Surface,
        env: Any,
    ) -> None:
        assert self.font is not None
        assert self.font_small is not None

        panel = pygame.Rect(
            30,
            495,
            535,
            175,
        )

        pygame.draw.rect(
            canvas,
            (249, 250, 251),
            panel,
            border_radius=10,
        )

        pygame.draw.rect(
            canvas,
            (203, 210, 215),
            panel,
            width=1,
            border_radius=10,
        )

        info = getattr(env, "last_info", {})

        in_transit = self._get_in_transit(
            env
        )

        stats = [
            (
                "In transit",
                in_transit,
            ),
            (
                "Last order",
                info.get(
                    "ordered_at_period_end",
                    0,
                ),
            ),
            (
                "Arrived",
                info.get(
                    "arriving_before_order",
                    0,
                ),
            ),
            (
                "Sold",
                info.get("sold", 0),
            ),
            (
                "Revenue",
                info.get("revenue", 0.0),
            ),
            (
                "Ordering cost",
                info.get(
                    "ordering_cost",
                    0.0,
                ),
            ),
            (
                "Holding cost",
                info.get(
                    "holding_cost",
                    0.0,
                ),
            ),
            (
                "Overflow cost",
                info.get(
                    "overflow_cost",
                    0.0,
                ),
            ),
        ]

        cols = 2
        cell_w = 250
        row_h = 36

        for index, (name, value) in enumerate(stats):
            col = index % cols
            row = index // cols

            x = panel.left + 18 + col * cell_w
            y = panel.top + 16 + row * row_h

            if isinstance(value, float):
                value_str = f"{value:.2f}"
            else:
                value_str = str(value)

            text = self.font_small.render(
                f"{name}: {value_str}",
                True,
                (46, 52, 57),
            )

            canvas.blit(
                text,
                (x, y),
            )


    def _draw_episode_demand_graph(
        self,
        canvas: pygame.Surface,
        env: Any,
    ) -> None:
        """Draw demand over the complete current episode in the lower right."""
        assert self.font is not None
        assert self.font_small is not None

        panel = pygame.Rect(
            590,
            495,
            self.WIDTH - 620,
            175,
        )

        pygame.draw.rect(
            canvas,
            (249, 250, 251),
            panel,
            border_radius=10,
        )
        pygame.draw.rect(
            canvas,
            (203, 210, 215),
            panel,
            width=1,
            border_radius=10,
        )

        title = self.font_small.render(
            "Demand — current episode",
            True,
            (46, 52, 57),
        )
        canvas.blit(title, (panel.left + 14, panel.top + 9))

        history = self._episode_demand_history
        if not history:
            return

        steps = [item[0] for item in history]
        demands = [item[1] for item in history]

        plot_left = panel.left + 48
        plot_right = panel.right - 14
        plot_top = panel.top + 35
        plot_bottom = panel.bottom - 28
        plot_width = max(1, plot_right - plot_left)
        plot_height = max(1, plot_bottom - plot_top)

        episode_length = max(
            1,
            int(
                getattr(
                    env,
                    "episode_length",
                    max(steps[-1], 1),
                )
            ),
        )

        y_max = max(1, max(demands, default=1))

        def x_to_screen(step_value: int) -> int:
            ratio = float(np.clip(step_value / episode_length, 0.0, 1.0))
            return int(plot_left + ratio * plot_width)

        def y_to_screen(value: float) -> int:
            ratio = float(np.clip(value / y_max, 0.0, 1.0))
            return int(plot_bottom - ratio * plot_height)

        for fraction in (0.0, 0.5, 1.0):
            value = y_max * fraction
            y = y_to_screen(value)
            pygame.draw.line(
                canvas,
                (228, 232, 235),
                (plot_left, y),
                (plot_right, y),
                width=1,
            )
            label = self.font_small.render(
                str(int(round(value))),
                True,
                (95, 102, 108),
            )
            canvas.blit(
                label,
                (
                    plot_left - label.get_width() - 7,
                    y - label.get_height() // 2,
                ),
            )

        pygame.draw.line(
            canvas,
            (160, 168, 174),
            (plot_left, plot_top),
            (plot_left, plot_bottom),
            width=1,
        )
        pygame.draw.line(
            canvas,
            (160, 168, 174),
            (plot_left, plot_bottom),
            (plot_right, plot_bottom),
            width=1,
        )

        points = [
            (x_to_screen(step), y_to_screen(demand))
            for step, demand in history
        ]

        demand_color = (205, 119, 73)

        if len(points) >= 2:
            pygame.draw.lines(
                canvas,
                demand_color,
                False,
                points,
                width=2,
            )
        elif points:
            pygame.draw.circle(
                canvas,
                demand_color,
                points[0],
                3,
            )

        if points:
            pygame.draw.circle(
                canvas,
                demand_color,
                points[-1],
                4,
            )

        tick_steps = sorted(
            set(
                int(round(episode_length * fraction))
                for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)
            )
        )

        for tick_step in tick_steps:
            x = x_to_screen(tick_step)
            pygame.draw.line(
                canvas,
                (185, 192, 198),
                (x, plot_bottom),
                (x, plot_bottom + 4),
                width=1,
            )
            label = self.font_small.render(
                str(tick_step),
                True,
                (90, 97, 103),
            )
            canvas.blit(
                label,
                (
                    x - label.get_width() // 2,
                    plot_bottom + 4,
                ),
            )

        current_label = self.font_small.render(
            f"Demand={demands[-1]}",
            True,
            (75, 82, 88),
        )
        canvas.blit(
            current_label,
            (
                panel.right - current_label.get_width() - 14,
                panel.top + 9,
            ),
        )




    def _get_in_transit(
        self,
        env: Any,
    ) -> int:
        fn = getattr(
            env,
            "_get_in_transit",
            None,
        )

        if callable(fn):
            return int(fn())

        shipments = getattr(
            env,
            "_shipments",
            [],
        )

        return int(
            sum(
                int(s.get("qty", 0))
                for s in shipments
            )
        )

    @staticmethod
    def _safe_ratio(
        numerator: float,
        denominator: float,
    ) -> float:
        if denominator <= 0:
            return 0.0

        return float(
            np.clip(
                numerator / denominator,
                0.0,
                1.0,
            )
        )

    def _show_canvas(
        self,
        canvas: pygame.Surface,
    ) -> None:
        if self.window is None:
            pygame.display.init()

            self.window = pygame.display.set_mode(
                (self.WIDTH, self.HEIGHT)
            )

            pygame.display.set_caption(
                "Supply Chain Environment"
            )

            if SDL2_MULTIWINDOW_AVAILABLE:
                try:
                    self._main_sdl2_window = (
                        SDL2Window.from_display_module()
                    )
                except Exception:
                    self._main_sdl2_window = None

        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.window.blit(
            canvas,
            (0, 0),
        )

        pygame.display.flip()

        if self.limit_fps:
            self.clock.tick(self.fps)
        else:
            self.clock.tick(0)

    @staticmethod
    def _canvas_to_rgb(
        canvas: pygame.Surface,
    ) -> np.ndarray:
        """
        pygame uses (width, height, channels).
        Gymnasium expects (height, width, channels).
        """
        return np.transpose(
            np.asarray(
                pygame.surfarray.pixels3d(
                    canvas
                )
            ),
            axes=(1, 0, 2),
        ).copy()

    def _window_wants_to_close(self) -> bool:
        close_requested, _ = self._handle_events()
        return close_requested