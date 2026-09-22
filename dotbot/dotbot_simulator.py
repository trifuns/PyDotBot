# SPDX-FileCopyrightText: 2023-present Inria
# SPDX-FileCopyrightText: 2023-present Filip Maksimovic <filip.maksimovic@inria.fr>
# SPDX-FileCopyrightText: 2024-present Alexandre Abadie <alexandre.abadie@inria.fr>
#
# SPDX-License-Identifier: BSD-3-Clause

"""Dotbot simulator for the DotBot project."""

import ctypes
import heapq
import queue
import random
import threading
import time
from dataclasses import dataclass
from enum import Enum
from math import atan2, cos, hypot, pi, sin, sqrt
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import toml
from dotbot_utils.protocol import Frame, Header, Packet
from pydantic import BaseModel, Field, model_validator

from dotbot import (
    GATEWAY_ADDRESS_DEFAULT,
    SIMULATOR_INIT_STATE_DEFAULT,
    addr_to_hex,
)
from dotbot.logger import LOGGER
from dotbot.mari_schedules import MariSchedule, load_schedules, select_schedule
from dotbot.protocol import ControlModeType, PayloadDotBotAdvertisement, PayloadType

Kv = 700  # motor speed constant in RPM
R = 50  # motor reduction ratio
D = 44  # wheel diameter in mm
L = 78  # distance between the two wheels in mm

# Encoder model: counts per mm of wheel travel (must match C-side DB_MM_PER_COUNT)
# mm_per_count = pi * D / (CPR * R)
ENCODER_CPR = 12  # counts per motor shaft revolution
MM_PER_COUNT = (pi * D) / (ENCODER_CPR * R)  # ~0.2618 mm/count

# Control parameters for the automatic mode
MOTOR_SPEED = 60
ANGULAR_SPEED_GAIN = 1.5
REDUCE_SPEED_FACTOR = 0.8
REDUCE_SPEED_ANGLE = 25

SIMULATOR_STEP_DELTA_T = 0.01  # 10 ms

# Battery model parameters
INITIAL_BATTERY_VOLTAGE = 3000  # mV
MAX_BATTERY_DURATION = 60 * 60 * 3  # 3 hours in seconds

ADVERTISEMENT_INTERVAL_S = 0.5
SIMULATOR_UPDATE_INTERVAL_S = 0.05

# Round-trip latency an empirical mari hardware campaign measured (p50, ms) at
# each schedule's full node capacity. MariNetworkSimulator derives a small
# additive per-direction overhead from these against the per-cell wait its own
# schedule model computes (never a negative one — the overhead only fills a
# gap the theoretical wait underestimates). A starting calibration; a future
# hardware campaign may revise it.
MARI_MEASURED_RTT_P50_MS = {"tiny": 40.0, "medium": 96.0, "big": 146.0, "huge": 233.0}

# Join/association backoff constants, from mari/firmware/mari/association.c —
# a node contending for a shared uplink (S) cell doubles its backoff window
# (as 2^n - 1 slots) after each collision, up to n=MARI_BACKOFF_N_MAX, and
# gives up (rescans) after MARI_BACKOFF_MAX_STREAK consecutive collisions at
# that max window.
MARI_BACKOFF_N_MIN = 4
MARI_BACKOFF_N_MAX = 6
MARI_BACKOFF_MAX_STREAK = 3
# association.c's MARI_JOINING_STATE_TIMEOUT is ~1.5 slot durations.
MARI_JOINING_STATE_TIMEOUT_SLOTS = 1.5


class MariJoinState(str, Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    SYNCED = "synced"
    JOINING = "joining"
    JOINED = "joined"


@dataclass
class _AssocNodeState:
    state: MariJoinState = MariJoinState.IDLE
    backoff_n: int = -1
    backoff_remaining: int = 0
    consecutive_max_backoff: int = 0

# Feature order must match utils/sim_to_real/train_gru.py FEATURE_COLS
GRU_FEATURE_COLS = [
    "pwm_left",
    "pwm_right",
    "encoder_left",
    "encoder_right",
    "direction",
    "pos_x",
    "pos_y",
]
GRU_SEQ_LEN_DEFAULT = 20  # must match --seq-len used during training


def battery_discharge_model(time_elapsed_s: float) -> int:
    """Linear discharge over MAX_BATTERY_DURATION (supercapacitor idle model)."""
    t = min(time_elapsed_s / MAX_BATTERY_DURATION, 1.0)
    return max(0, int(INITIAL_BATTERY_VOLTAGE * (1 - t)))


def wheel_speed_from_pwm(pwm: float) -> float:
    """Convert a PWM value to a wheel speed in mm/s."""
    if pwm > 100:
        pwm = 100
    if pwm < -100:
        pwm = -100
    return pwm * D * Kv / (R * 127)


@dataclass
class Waypoint:
    """Waypoint class for the dotbot simulator."""

    x: int
    y: int


class SimulatedNetworkMode(str, Enum):
    DEFAULT = "default"
    MARI = "mari"


class SimulatedNetworkSettings(BaseModel):
    pdr: int = 100
    uplink_pdr: Optional[int] = None
    downlink_pdr: Optional[int] = None
    # None = derive from the mari schedule the fleet size selects (firmware
    # mac.h, computed rather than hand-copied — see dotbot.mari_schedules).
    slot_duration_ms: Optional[float] = None
    mqtt_latency_ms: float = 0.0
    # None = auto-select the smallest mari schedule that fits the fleet size
    # ("tiny"/"medium"/"big"/"huge"); set to force one explicitly.
    schedule: Optional[str] = None
    # Explicit mari firmware checkout to parse schedules/timing from; None
    # resolves via $MARI_FIRMWARE_DIR or a sibling `mari/` directory, falling
    # back to the bundled generated snapshot if neither is found.
    mari_dir: Optional[str] = None
    # Gateway position in mm, arena coordinates — the reference point distance
    # is measured from for pdr_by_distance_m.
    gateway_pos_x: int = 1000
    gateway_pos_y: int = 1000
    # [[distance_m, pdr_percent], ...] anchors, linearly interpolated (clamped
    # outside the range) to derive per-bot PDR from its distance to the
    # gateway. None falls back to the flat pdr/uplink_pdr/downlink_pdr above.
    # Deliberately never a function of node count or schedule size — an
    # empirical mari hardware campaign found PDR tracks distance only; an
    # apparent PDR-vs-schedule-size correlation in raw aggregates turned out
    # to be a selection-bias artifact (smaller schedules fill with
    # better-connected bots first), not a real swarm-size effect.
    pdr_by_distance_m: Optional[List[Tuple[float, float]]] = None

    @model_validator(mode="after")
    def _fill_mari_pdrs(self):
        if self.uplink_pdr is None:
            self.uplink_pdr = self.pdr
        if self.downlink_pdr is None:
            self.downlink_pdr = self.pdr
        return self


def _random_address() -> str:
    return f"{random.getrandbits(64):016X}"


class SimulatedDotBotSettings(BaseModel):
    address: str = Field(default_factory=_random_address)
    pos_x: int
    pos_y: int
    direction: int = -1000
    calibrated: int = 0xFF
    motor_left_error: float = 0
    motor_right_error: float = 0
    custom_control_loop_library: Path = None
    gru_model_path: Path = None
    battery_model_path: Path = None
    network_mode: SimulatedNetworkMode = SimulatedNetworkMode.DEFAULT


class ControlLoopWaypoint(ctypes.Structure):
    """Mirrors coordinate_t from control_loop.h — used when calling control_loop_set_waypoints."""

    _fields_ = [
        ("x", ctypes.c_uint32),
        ("y", ctypes.c_uint32),
    ]


class RobotControl(ctypes.Structure):
    """Mirrors robot_control_t from control_loop.h.

    Only the stable external I/O boundary is represented here.  All internal
    algorithm state lives in the opaque context managed by the C library.
    Layout must stay in sync with the C struct (no internal padding gaps).
    """

    _fields_ = [
        # Inputs — robot state (4-byte fields first, no padding gaps)
        ("pos_x", ctypes.c_uint32),
        ("pos_y", ctypes.c_uint32),
        ("encoder_left", ctypes.c_int32),  # signed delta counts since last call
        ("encoder_right", ctypes.c_int32),  # signed delta counts since last call
        # Outputs — current target waypoint coordinates (written by C, for telemetry)
        ("waypoint_x", ctypes.c_uint32),
        ("waypoint_y", ctypes.c_uint32),
        # Input — robot heading (2-byte, followed by 1-byte fields — no internal padding)
        ("direction", ctypes.c_int16),
        # Outputs — actuation (written by C)
        ("pwm_left", ctypes.c_int8),
        ("pwm_right", ctypes.c_int8),
        # Outputs — status flags (written by C)
        ("waypoint_reached", ctypes.c_uint8),
        ("all_done", ctypes.c_uint8),
        ("waypoint_idx", ctypes.c_uint8),
    ]


class InitStateToml(BaseModel):
    dotbots: List[SimulatedDotBotSettings]
    network: SimulatedNetworkSettings = SimulatedNetworkSettings()


class DotBotSimulator:
    """Simulator class for the dotbot."""

    def __init__(self, settings: SimulatedDotBotSettings, tx_queue: queue.Queue):
        self.address = settings.address.upper()
        self.pos_x = settings.pos_x
        self.pos_y = settings.pos_y
        self.theta = settings.direction * -1 if settings.direction != -1000 else 0
        self.motor_left_error = settings.motor_left_error
        self.motor_right_error = settings.motor_right_error
        self.custom_control_loop_library = settings.custom_control_loop_library
        self._control_loop_func = self._init_control_loop()
        self.time_elapsed_s = 0

        self.pwm_left = 0
        self.pwm_right = 0
        self.direction = settings.direction

        # Accumulated encoder deltas between control-loop calls (control runs at
        # SIMULATOR_UPDATE_INTERVAL_S, physics at SIMULATOR_STEP_DELTA_T — multiple
        # physics steps per control call)
        self.encoder_left_acc = 0.0
        self.encoder_right_acc = 0.0
        # Last encoder delta actually passed to update_control — advertised to match
        # real-robot telemetry semantics (the value from the most recent control call)
        self._last_encoder_left = 0
        self._last_encoder_right = 0

        self.calibrated = settings.calibrated
        self.waypoint_threshold = 0
        self.waypoints = []
        self.waypoint_index = 0
        self.waypoint_x = 0
        self.waypoint_y = 0

        self.logger = LOGGER.bind(context=__name__, address=self.address)
        self._gru_model = None
        self._gru_buffer: list[list[float]] = (
            []
        )  # rolling window of raw feature vectors
        if settings.gru_model_path is not None:
            self._gru_model = self._load_gru_model(settings.gru_model_path)

        self._battery_model = None
        self.battery_voltage: float = float(INITIAL_BATTERY_VOLTAGE)
        if settings.battery_model_path is not None:
            self._battery_model = self._load_battery_model(settings.battery_model_path)

        self._lock = threading.Lock()
        self.tx_queue = tx_queue
        self.queue = queue.Queue()
        self.advertise_thread = threading.Thread(target=self.advertise, daemon=True)
        self.control_thread = threading.Thread(target=self.control_thread, daemon=True)
        self.rx_thread = threading.Thread(target=self.rx_frame, daemon=True)
        self.main_thread = threading.Thread(target=self.update_state, daemon=True)
        self.controller_mode: ControlModeType = ControlModeType.MANUAL
        self._stop_event = threading.Event()
        self.logger.info(
            "DotBot simulator initialized",
            pos_x=self.pos_x,
            pos_y=self.pos_y,
            direction=self.direction,
            theta=self.theta,
        )

    def _load_gru_model(self, path: Path):
        """Load a TorchScript GRU residual model from *path*."""
        try:
            import torch  # imported lazily — not required when model is unused

            model = torch.jit.load(str(path), map_location="cpu")
            model.eval()
            self.logger.info("GRU residual model loaded", path=str(path))
            return model
        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "Failed to load GRU model", path=str(path), error=str(exc)
            )
            return None

    def _load_battery_model(self, path: Path):
        """Load a TorchScript battery discharge model from *path*."""
        try:
            import torch  # imported lazily — not required when model is unused

            model = torch.jit.load(str(path), map_location="cpu")
            model.eval()
            self.logger.info("Battery discharge model loaded", path=str(path))
            return model
        except Exception as exc:  # noqa: BLE001
            self.logger.error(
                "Failed to load battery model", path=str(path), error=str(exc)
            )
            return None

    def _gru_residual(self) -> tuple[float, float, float, float]:
        """Return (dx, dy, d_enc_left, d_enc_right) predicted by the GRU, or zeros."""
        if self._gru_model is None or len(self._gru_buffer) < GRU_SEQ_LEN_DEFAULT:
            return 0.0, 0.0, 0.0, 0.0
        try:
            import torch

            seq = self._gru_buffer[-GRU_SEQ_LEN_DEFAULT:]
            x = torch.tensor([seq], dtype=torch.float32)  # (1, seq_len, n_features)
            with torch.no_grad():
                pred = self._gru_model(x)  # (1, 4)
            return (
                float(pred[0, 0]),
                float(pred[0, 1]),
                float(pred[0, 2]),
                float(pred[0, 3]),
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("GRU inference failed", error=str(exc))
            return 0.0, 0.0, 0.0, 0.0

    def start(self):
        self.rx_thread.start()
        self.advertise_thread.start()
        self.control_thread.start()
        self.main_thread.start()
        self.logger.info("DotBot simulator started")

    @property
    def header(self):
        return Header(
            destination=int(GATEWAY_ADDRESS_DEFAULT, 16),
            source=int(self.address, 16),
        )

    def diff_drive_model_update(self, dt=SIMULATOR_STEP_DELTA_T):
        """State space model update."""
        pos_x_old = self.pos_x
        pos_y_old = self.pos_y
        theta_old = self.theta

        # Compute each wheel's real speed considering the motor error and the minimum PWM to move
        v_left_real = wheel_speed_from_pwm(self.pwm_left) * (1 - self.motor_left_error)
        v_right_real = wheel_speed_from_pwm(self.pwm_right) * (
            1 - self.motor_right_error
        )

        V = (v_right_real + v_left_real) / 2
        w = (v_right_real - v_left_real) / L
        x_dot = V * cos(theta_old * pi / 180 - pi / 2)
        y_dot = V * sin(theta_old * pi / 180 + pi / 2)
        dx = x_dot * dt
        dy = y_dot * dt

        self.pos_x = pos_x_old + dx
        self.pos_y = pos_y_old + dy
        self.theta = (theta_old + w * dt * 180 / pi) % 360

        if sqrt(dx**2 + dy**2):
            self.direction = int(-1 * atan2(dx, dy) * 180 / pi) % 360
            if self.direction > 180:
                self.direction -= 360
            elif self.direction < -180:
                self.direction += 360

        # Accumulate encoder counts for this physics step
        if self.controller_mode == ControlModeType.AUTO:
            self.encoder_left_acc += v_left_real * SIMULATOR_STEP_DELTA_T / MM_PER_COUNT
            self.encoder_right_acc += (
                v_right_real * SIMULATOR_STEP_DELTA_T / MM_PER_COUNT
            )

        # Update GRU feature buffer with the post-step state
        if self._gru_model is not None:
            self._gru_buffer.append(
                [
                    float(self.pwm_left),
                    float(self.pwm_right),
                    float(self.encoder_left_acc),
                    float(self.encoder_right_acc),
                    float(self.direction),
                    float(self.pos_x),
                    float(self.pos_y),
                ]
            )
            # Keep only as many steps as needed to avoid unbounded growth
            if len(self._gru_buffer) > GRU_SEQ_LEN_DEFAULT:
                self._gru_buffer.pop(0)
            res_x, res_y, res_enc_l, res_enc_r = self._gru_residual()
            self.pos_x += res_x
            self.pos_y += res_y
            if self.controller_mode == ControlModeType.AUTO:
                self.encoder_left_acc += res_enc_l
                self.encoder_right_acc += res_enc_r

        self.time_elapsed_s += dt
        if self._battery_model is not None:
            try:
                import torch

                # Encoders are only reported by the real hardware in AUTO mode;
                # mirror that here so the battery model sees consistent inputs.
                in_auto = self.controller_mode == ControlModeType.AUTO
                enc_left = float(self.encoder_left_acc) if in_auto else 0.0
                enc_right = float(self.encoder_right_acc) if in_auto else 0.0
                features = torch.tensor(
                    [
                        [
                            float(self.pwm_left),
                            float(self.pwm_right),
                            enc_left,
                            enc_right,
                            float(int(self.controller_mode)),
                        ]
                    ],
                    dtype=torch.float32,
                )
                with torch.no_grad():
                    rate = float(self._battery_model(features)[0, 0])  # mV/s
                self.battery_voltage = max(0.0, self.battery_voltage + rate * dt)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Battery model inference failed", error=str(exc))
        else:
            self.battery_voltage = battery_discharge_model(self.time_elapsed_s)

        self.logger.debug(
            "State updated",
            pos_x=int(self.pos_x),
            pos_y=int(self.pos_y),
            theta=int(self.theta),
            direction=int(self.direction),
            pwm_left=int(self.pwm_left),
            pwm_right=int(self.pwm_right),
        )

    def update_state(self):
        """Update the state of the dotbot simulator."""
        while True:
            with self._lock:
                self.diff_drive_model_update()
            is_stopped = self._stop_event.wait(SIMULATOR_STEP_DELTA_T)
            if is_stopped:
                break

    def _init_control_loop(self) -> callable:
        """Initialize the control loop, potentially loading a custom control loop library."""
        if self.custom_control_loop_library is not None:
            lib = ctypes.CDLL(self.custom_control_loop_library)
            self.custom_control_loop_library = lib

            lib.control_loop_alloc.argtypes = []
            lib.control_loop_alloc.restype = ctypes.c_void_p

            lib.control_loop_free.argtypes = [ctypes.c_void_p]
            lib.control_loop_free.restype = None

            lib.control_loop_set_waypoints.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ControlLoopWaypoint),
                ctypes.c_uint8,
                ctypes.c_uint32,
            ]
            lib.control_loop_set_waypoints.restype = None

            lib.update_control.argtypes = [
                ctypes.POINTER(RobotControl),
                ctypes.c_void_p,
            ]
            lib.update_control.restype = None

            self._control_ctx = lib.control_loop_alloc()
            self.custom_robot_control = RobotControl()
            return self._control_loop_custom
        else:
            return self._control_loop_default

    def _control_loop_custom(self):
        """Control loop using a custom control loop library."""
        self.custom_robot_control.pos_x = int(self.pos_x)
        self.custom_robot_control.pos_y = int(self.pos_y)
        self.custom_robot_control.direction = self.direction
        self._last_encoder_left = int(self.encoder_left_acc)
        self._last_encoder_right = int(self.encoder_right_acc)
        self.custom_robot_control.encoder_left = self._last_encoder_left
        self.custom_robot_control.encoder_right = self._last_encoder_right
        self.encoder_left_acc = 0
        self.encoder_right_acc = 0

        self.custom_control_loop_library.update_control(
            ctypes.byref(self.custom_robot_control),
            self._control_ctx,
        )

        self.pwm_left = self.custom_robot_control.pwm_left
        self.pwm_right = self.custom_robot_control.pwm_right
        self.waypoint_index = self.custom_robot_control.waypoint_idx
        self.waypoint_x = self.custom_robot_control.waypoint_x
        self.waypoint_y = self.custom_robot_control.waypoint_y

        self.logger.info(
            "Custom loop",
            pwm_left=self.pwm_left,
            pwm_right=self.pwm_right,
            direction=self.direction,
            encoder_left=int(self.custom_robot_control.encoder_left),
            encoder_right=int(self.custom_robot_control.encoder_right),
            waypoint_index=self.custom_robot_control.waypoint_idx,
            waypoint_x=self.custom_robot_control.waypoint_x,
            waypoint_y=self.custom_robot_control.waypoint_y,
            waypoint_reached=self.custom_robot_control.waypoint_reached,
            all_done=self.custom_robot_control.all_done,
        )

        if self.custom_robot_control.all_done:
            self.logger.info("All waypoints completed")
            self.waypoint_index = 0
            self.waypoint_x = 0
            self.waypoint_y = 0
            self.controller_mode = ControlModeType.MANUAL
            self.encoder_right_acc = 0
            self.encoder_left_acc = 0

    def _control_loop_default(self):
        self._last_encoder_left = int(self.encoder_left_acc)
        self._last_encoder_right = int(self.encoder_right_acc)
        self.encoder_left_acc = 0.0
        self.encoder_right_acc = 0.0

        delta_x = self.waypoints[self.waypoint_index].pos_x - self.pos_x
        delta_y = self.waypoints[self.waypoint_index].pos_y - self.pos_y
        distance_to_target = sqrt(delta_x**2 + delta_y**2)

        # check if we are close enough to the "next" waypoint
        if distance_to_target < self.waypoint_threshold:
            self.logger.info("Waypoint reached", waypoint_index=self.waypoint_index)
            self.waypoint_index += 1
            # check if there are no more waypoints:
            if self.waypoint_index >= len(self.waypoints):
                self.logger.info(
                    "Last waypoint reached", waypoint_index=self.waypoint_index
                )
                self.pwm_left = 0
                self.pwm_right = 0
                self.waypoint_index = 0
                self.waypoint_x = 0
                self.waypoint_y = 0
                self.controller_mode = ControlModeType.MANUAL
                self.encoder_right_acc = 0
                self.encoder_left_acc = 0
                return

        self.waypoint_x = int(self.waypoints[self.waypoint_index].pos_x)
        self.waypoint_y = int(self.waypoints[self.waypoint_index].pos_y)

        angle_to_target = -1 * atan2(delta_x, delta_y) * 180 / pi
        robot_angle = self.direction
        if robot_angle >= 180:
            robot_angle -= 360
        elif robot_angle < -180:
            robot_angle += 360

        error_angle = angle_to_target - robot_angle
        if error_angle >= 180:
            error_angle -= 360
        elif error_angle < -180:
            error_angle += 360

        speed_reduction_factor: float = 1.0
        if distance_to_target < self.waypoint_threshold * 2:
            speed_reduction_factor = REDUCE_SPEED_FACTOR
        if error_angle > REDUCE_SPEED_ANGLE or error_angle < -REDUCE_SPEED_ANGLE:
            speed_reduction_factor = REDUCE_SPEED_FACTOR

        angular_speed = (error_angle / 180) * MOTOR_SPEED * ANGULAR_SPEED_GAIN
        self.pwm_left = MOTOR_SPEED * speed_reduction_factor + angular_speed
        self.pwm_right = MOTOR_SPEED * speed_reduction_factor - angular_speed

        self.logger.info(
            "Loop update",
            robot_angle=int(robot_angle),
            angle_to_target=int(angle_to_target),
            error_angle=int(error_angle),
            angular_speed=int(angular_speed),
            pwm_left=int(self.pwm_left),
            pwm_right=int(self.pwm_right),
            theta=int(self.theta),
            waypoint=f"{self.waypoint_index}/{len(self.waypoints)}",
        )

    def control_thread(self):
        """Control thread to update the state of the dotbot simulator."""
        while self._stop_event.is_set() is False:
            if self.controller_mode == ControlModeType.AUTO:
                with self._lock:
                    self._control_loop_func()
            is_stopped = self._stop_event.wait(SIMULATOR_UPDATE_INTERVAL_S)
            if is_stopped:
                break

    def advertise(self):
        """Send an advertisement message to the gateway."""
        while self._stop_event.is_set() is False:
            payload = Frame(
                header=self.header,
                packet=Packet.from_payload(
                    PayloadDotBotAdvertisement(
                        calibrated=self.calibrated,
                        direction=self.direction,
                        pos_x=int(self.pos_x) if self.pos_x >= 0 else 0,
                        pos_y=int(self.pos_y) if self.pos_y >= 0 else 0,
                        battery=int(self.battery_voltage),
                        pwm_left=int(self.pwm_left),
                        pwm_right=int(self.pwm_right),
                        mode=int(self.controller_mode),
                        encoder_left=self._last_encoder_left,
                        encoder_right=self._last_encoder_right,
                        waypoint_x=int(self.waypoint_x),
                        waypoint_y=int(self.waypoint_y),
                        waypoint_idx=int(self.waypoint_index),
                    )
                ),
            )
            self.tx_queue.put_nowait(payload)
            is_stopped = self._stop_event.wait(ADVERTISEMENT_INTERVAL_S)
            if is_stopped:
                break

    def rx_frame(self):
        """Decode the serial input received from the gateway."""

        while self._stop_event.is_set() is False:
            frame = self.queue.get()
            if frame is None:
                break
            with self._lock:
                if self.address == addr_to_hex(int(frame.header.destination)):
                    if frame.payload_type == PayloadType.CMD_MOVE_RAW:
                        self.controller_mode = ControlModeType.MANUAL
                        self.waypoint_index = 0
                        self.waypoint_x = 0
                        self.waypoint_y = 0
                        self.pwm_left = frame.packet.payload.left_y
                        self.pwm_right = frame.packet.payload.right_y
                        if self.pwm_left > 127:
                            self.pwm_left = self.pwm_left - 256
                        if self.pwm_right > 127:
                            self.pwm_right = self.pwm_right - 256
                        self.logger.info(
                            "RAW command received",
                            pwm_left=self.pwm_left,
                            pwm_right=self.pwm_right,
                        )
                    elif frame.payload_type == PayloadType.LH2_WAYPOINTS:
                        self.waypoint_threshold = frame.packet.payload.threshold
                        self.waypoints = frame.packet.payload.waypoints
                        self.waypoint_index = 0
                        self.encoder_left_acc = 0.0
                        self.encoder_right_acc = 0.0
                        if hasattr(self, "_control_ctx"):
                            n = len(self.waypoints)
                            WaypointArray = ControlLoopWaypoint * n
                            waypoint_arr = WaypointArray(
                                *[
                                    ControlLoopWaypoint(x=int(w.pos_x), y=int(w.pos_y))
                                    for w in self.waypoints
                                ]
                            )
                            self.custom_control_loop_library.control_loop_set_waypoints(
                                self._control_ctx,
                                waypoint_arr,
                                n,
                                int(self.waypoint_threshold),
                            )
                        self.logger.info(
                            "Waypoints received",
                            threshold=self.waypoint_threshold,
                            waypoints=self.waypoints,
                        )
                        if self.waypoints:
                            self.controller_mode = ControlModeType.AUTO
                        else:
                            self.pwm_left = 0
                            self.pwm_right = 0
                            self.controller_mode = ControlModeType.MANUAL

    def stop(self):
        self.logger.info(f"Stopping DotBot {self.address} simulator...")
        self._stop_event.set()
        self.queue.put_nowait(None)  # unblock the rx_thread if waiting on the queue
        self.advertise_thread.join()
        self.control_thread.join()
        self.rx_thread.join()
        self.main_thread.join()
        if hasattr(self, "_control_ctx"):
            self.custom_control_loop_library.control_loop_free(self._control_ctx)
            self._control_ctx = None


def _load_named_schedule(name: str, mari_dir: Optional[str]) -> MariSchedule:
    schedules = load_schedules(mari_dir)
    try:
        return schedules[name]
    except KeyError:
        raise ValueError(f"unknown mari schedule {name!r}; have {sorted(schedules)}") from None


def _next_downlink_cell(after: int, downlink_cells: Tuple[int, ...]) -> int:
    """The nearest downlink cell strictly after `after`.

    Wraps to the first downlink cell of the next slotframe if none remain in
    this one, mirroring the cyclic schedule.
    """
    for cell in downlink_cells:
        if cell > after:
            return cell
    return downlink_cells[0]


def _interpolate_pdr(distance_m: float, anchors: List[Tuple[float, float]]) -> int:
    """Linear interpolation over (distance_m, pdr_percent) anchors, clamped at the ends."""
    points = sorted(anchors, key=lambda p: p[0])
    if distance_m <= points[0][0]:
        return round(points[0][1])
    if distance_m >= points[-1][0]:
        return round(points[-1][1])
    for (d0, p0), (d1, p1) in zip(points, points[1:]):
        if d0 <= distance_m <= d1:
            t = (distance_m - d0) / (d1 - d0)
            return round(p0 + t * (p1 - p0))
    return round(points[-1][1])  # unreachable — points is sorted and covers [d0, d_last]


class MariNetworkSimulator:
    """Discrete-event model of mari's TSCH link layer for the DotBot simulator.

    Delivers frames at the delay to a bot's assigned cell — a real ``U`` cell
    for uplink, the nearest ``D`` cell after it for downlink — on a schedule
    picked from mari's real fixed schedules. PDR is drawn per direction, per
    delivery attempt (mari has no link-layer ACK/retransmission, so every
    attempt is single-shot), from each bot's live distance to the gateway —
    never from the fleet size or the schedule.

    A bot has no assigned cell until :meth:`assign_cell` is called for it —
    driven by :class:`MariAssociationSimulator` reporting a completed join,
    via the :class:`MariSimulator` facade that owns both. Frames for a
    not-yet-assigned bot are silently dropped, matching a real unjoined node
    having no cell to transmit in.
    """

    def __init__(
        self,
        schedule: MariSchedule,
        settings: SimulatedNetworkSettings,
        dotbots: List["DotBotSimulator"],
        enqueue: Callable[[float, Callable], None],
        on_frame_received: Callable,
    ):
        self._schedule = schedule
        self._settings = settings
        self._dotbots = dotbots
        self._enqueue = enqueue
        self._on_frame_received = on_frame_received
        self._slot_duration_ms = settings.slot_duration_ms or schedule.slot_duration_ms

        self._uplink_cells = schedule.uplink_cell_indices()
        self._downlink_cells = schedule.downlink_cell_indices()
        self._next_uplink_i = 0
        # Guards the three attributes above/below against the facade thread's
        # assign_cell() racing an external caller thread's schedule_uplink/
        # schedule_downlink (unlike phase 1, cell assignment now happens after
        # start() rather than once, eagerly, before any thread runs).
        self._lock = threading.Lock()
        self._uplink_cell: dict[int, int] = {}
        self._downlink_cell: dict[int, int] = {}

        overhead_s = self._calibrated_overhead_ms() / 1000
        self._uplink_overhead_s = overhead_s
        self._downlink_overhead_s = overhead_s

    def assign_cell(self, dotbot_index: int) -> None:
        """Give a just-joined bot the next available uplink cell, in schedule
        order — mirrors scheduler.c's "first available U cell" semantics."""
        with self._lock:
            if self._next_uplink_i >= len(self._uplink_cells):
                raise ValueError(
                    f"mari schedule {self._schedule.name!r} has only "
                    f"{len(self._uplink_cells)} uplink cells, all already assigned"
                )
            uplink_cell = self._uplink_cells[self._next_uplink_i]
            self._next_uplink_i += 1
            self._uplink_cell[dotbot_index] = uplink_cell
            self._downlink_cell[dotbot_index] = _next_downlink_cell(
                uplink_cell, self._downlink_cells
            )

    def _calibrated_overhead_ms(self) -> float:
        """Extra per-direction delay an empirical mari campaign's measured RTT
        implies beyond this schedule's own per-cell wait — never negative, so
        calibration only ever fills a gap the theoretical wait underestimates,
        never shortens it."""
        measured_rtt_p50 = MARI_MEASURED_RTT_P50_MS.get(self._schedule.name)
        if measured_rtt_p50 is None:
            return 0.0
        # Expected one-way wait for a uniformly-phased request to hit its
        # assigned cell, averaged over a slotframe.
        theoretical_one_way_p50 = self._schedule.slotframe_ms / 2
        return max(0.0, measured_rtt_p50 / 2 - theoretical_one_way_p50)

    def _cell_delay_s(self, cell_index: int) -> float:
        slotframe_s = self._schedule.slotframe_ms / 1000
        cell_offset_s = cell_index * self._slot_duration_ms / 1000
        phase = time.monotonic() % slotframe_s
        return (cell_offset_s - phase) % slotframe_s

    def _pdr_percent(self, dotbot_index: int, flat_default: int) -> int:
        anchors = self._settings.pdr_by_distance_m
        if not anchors:
            return flat_default
        dotbot = self._dotbots[dotbot_index]
        distance_m = (
            hypot(
                dotbot.pos_x - self._settings.gateway_pos_x,
                dotbot.pos_y - self._settings.gateway_pos_y,
            )
            / 1000  # pos_x/pos_y are in mm
        )
        return _interpolate_pdr(distance_m, anchors)

    def schedule_uplink(self, frame, dotbot_index: int):
        with self._lock:
            uplink_cell = self._uplink_cell.get(dotbot_index)
        if uplink_cell is None:
            return  # not joined yet — no assigned cell to send in
        if random.randint(0, 100) > self._pdr_percent(dotbot_index, self._settings.uplink_pdr):
            return
        delay = (
            self._cell_delay_s(uplink_cell)
            + self._uplink_overhead_s
            + self._settings.mqtt_latency_ms / 1000
        )
        self._enqueue(delay, lambda: self._on_frame_received(frame))

    def schedule_downlink(
        self, bytes_: bytes, dotbot: "DotBotSimulator", dotbot_index: int
    ):
        with self._lock:
            downlink_cell = self._downlink_cell.get(dotbot_index)
        if downlink_cell is None:
            return  # not joined yet — no assigned cell to receive in
        if random.randint(0, 100) > self._pdr_percent(dotbot_index, self._settings.downlink_pdr):
            return
        frame = Frame.from_bytes(bytes_)
        delay = (
            self._cell_delay_s(downlink_cell)
            + self._downlink_overhead_s
            + self._settings.mqtt_latency_ms / 1000
        )
        self._enqueue(delay, lambda: dotbot.queue.put_nowait(frame))


class MariAssociationSimulator:
    """Discrete-event model of mari's join/association state machine
    (``IDLE -> SCANNING -> SYNCED -> JOINING -> JOINED``,
    ``repos/mari/firmware/mari/association.c``).

    Models the slotted-ALOHA contention on shared uplink (``S``) cells that
    dominates real join/formation time under a "join storm" (many nodes
    joining at once): every ``SYNCED`` node whose backoff has elapsed
    attempts a join request in the same ``S``-cell tick; more than one
    attempt in the same tick collides, doubling (up to
    ``MARI_BACKOFF_N_MAX``) the backoff window for every collider. A node
    that collides ``MARI_BACKOFF_MAX_STREAK`` times in a row at the max
    window rescans instead of retrying forever.

    Two simplifications relative to the firmware, made for a formation-time
    *distribution* model rather than a byte-exact protocol replay: scanning
    is a single random delay up to one slotframe (time to the first beacon a
    node happens to see) rather than a full rolling beacon-channel scan, and
    a collision is resolved immediately at the colliding ``S``-cell tick
    rather than after the firmware's own ~1.5-slot join-response timeout —
    both negligible next to the backoff windows and collision counts a join
    storm actually produces at the reference swarm sizes. The 5 s
    ``MARI_JOIN_TIMEOUT_SINCE_SYNCED`` wall-clock guard is not modeled either:
    it is a rare-case backstop against a lost gateway in real firmware, not a
    driver of the typical formation-time shape this class targets.
    """

    def __init__(
        self,
        schedule: MariSchedule,
        mari_indices: List[int],
        enqueue: Callable[[float, Callable], None],
        on_joined: Callable[[int], None],
    ):
        self._enqueue = enqueue
        self._on_joined = on_joined
        self._slot_duration_s = schedule.slot_duration_ms / 1000
        self._slotframe_s = schedule.slotframe_ms / 1000
        self._shared_uplink_offsets_s = tuple(
            cell * self._slot_duration_s for cell in schedule.shared_uplink_cell_indices()
        )
        if not self._shared_uplink_offsets_s:
            raise ValueError(f"mari schedule {schedule.name!r} has no shared-uplink (S) cells")
        self._nodes = {index: _AssocNodeState() for index in mari_indices}
        self._tick_running = False

    def start(self):
        for index in self._nodes:
            self._begin_scanning(index)

    def _begin_scanning(self, index: int):
        self._nodes[index].state = MariJoinState.SCANNING
        delay_s = random.uniform(0, self._slotframe_s)
        self._enqueue(delay_s, lambda: self._handle_synced(index))

    def _handle_synced(self, index: int):
        node = self._nodes[index]
        node.state = MariJoinState.SYNCED
        self._init_backoff(node)
        self._ensure_tick_scheduled()

    def _init_backoff(self, node: _AssocNodeState):
        node.backoff_n = MARI_BACKOFF_N_MIN
        node.consecutive_max_backoff = 0
        node.backoff_remaining = random.randint(0, (1 << node.backoff_n) - 1)

    def _delay_to_next_shared_uplink(self) -> float:
        phase = time.monotonic() % self._slotframe_s
        return min((offset - phase) % self._slotframe_s for offset in self._shared_uplink_offsets_s)

    def _ensure_tick_scheduled(self):
        if self._tick_running:
            return
        self._tick_running = True
        self._enqueue(self._delay_to_next_shared_uplink(), self._tick)

    def _tick(self):
        ready = []
        for index, node in self._nodes.items():
            if node.state != MariJoinState.SYNCED:
                continue
            if node.backoff_remaining > 0:
                node.backoff_remaining -= 1
            else:
                ready.append(index)

        if len(ready) == 1:
            index = ready[0]
            self._nodes[index].state = MariJoinState.JOINING
            timeout_s = MARI_JOINING_STATE_TIMEOUT_SLOTS * self._slot_duration_s
            self._enqueue(timeout_s, lambda: self._handle_joined(index))
        elif len(ready) > 1:
            for index in ready:
                self._handle_collision(index)

        if any(node.state != MariJoinState.JOINED for node in self._nodes.values()):
            self._enqueue(self._delay_to_next_shared_uplink(), self._tick)
        else:
            self._tick_running = False

    def _handle_collision(self, index: int):
        node = self._nodes[index]
        node.backoff_n = min(node.backoff_n + 1, MARI_BACKOFF_N_MAX)
        if node.backoff_n >= MARI_BACKOFF_N_MAX:
            node.consecutive_max_backoff += 1
        else:
            node.consecutive_max_backoff = 0

        if node.consecutive_max_backoff >= MARI_BACKOFF_MAX_STREAK:
            self._begin_scanning(index)  # stuck: give up and rescan
            return
        node.backoff_remaining = random.randint(0, (1 << node.backoff_n) - 1)

    def _handle_joined(self, index: int):
        self._nodes[index].state = MariJoinState.JOINED
        self._on_joined(index)


class MariSimulator:
    """Facade owning mari's steady-state network model
    (:class:`MariNetworkSimulator`) and its join/association model
    (:class:`MariAssociationSimulator`) behind one thread and one event heap,
    so :class:`DotBotSimulatorCommunicationInterface` sees exactly one mari
    simulator object rather than two independently-wired ones.

    This is also where the two sub-models are coupled: a bot's frames only
    reach :class:`MariNetworkSimulator`'s scheduling once
    :class:`MariAssociationSimulator` reports that bot joined
    (``on_joined=self._network.assign_cell``).
    """

    def __init__(
        self,
        settings: SimulatedNetworkSettings,
        dotbots: List["DotBotSimulator"],
        mari_indices: List[int],
        on_frame_received: Callable,
    ):
        schedule: MariSchedule = (
            select_schedule(len(mari_indices), mari_dir=settings.mari_dir)
            if settings.schedule is None
            else _load_named_schedule(settings.schedule, settings.mari_dir)
        )
        if len(mari_indices) > schedule.max_nodes:
            raise ValueError(
                f"mari schedule {schedule.name!r} supports at most "
                f"{schedule.max_nodes} nodes, but {len(mari_indices)} "
                "dotbots are in mari mode"
            )

        self._heap: list = []
        self._seq = 0
        self._cond = threading.Condition()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self._network = MariNetworkSimulator(
            schedule=schedule,
            settings=settings,
            dotbots=dotbots,
            enqueue=self._enqueue,
            on_frame_received=on_frame_received,
        )
        self._association = MariAssociationSimulator(
            schedule=schedule,
            mari_indices=mari_indices,
            enqueue=self._enqueue,
            on_joined=self._network.assign_cell,
        )

    def start(self):
        self._thread.start()
        self._association.start()

    def stop(self):
        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join()

    def schedule_uplink(self, frame, dotbot_index: int):
        self._network.schedule_uplink(frame, dotbot_index)

    def schedule_downlink(self, bytes_: bytes, dotbot: "DotBotSimulator", dotbot_index: int):
        self._network.schedule_downlink(bytes_, dotbot, dotbot_index)

    def _enqueue(self, delay_s: float, fn: Callable):
        delivery = time.monotonic() + delay_s
        with self._cond:
            heapq.heappush(self._heap, (delivery, self._seq, fn))
            self._seq += 1
            self._cond.notify()

    def _run(self):
        with self._cond:
            while not self._stop_event.is_set():
                now = time.monotonic()
                if self._heap:
                    deadline, _, fn = self._heap[0]
                    if deadline <= now:
                        heapq.heappop(self._heap)
                        self._cond.release()
                        try:
                            fn()
                        finally:
                            self._cond.acquire()
                        continue
                    wait = deadline - now
                else:
                    wait = None
                self._cond.wait(timeout=wait)


def packaged_init_state_path() -> Path:
    """Absolute path to the default simulator world shipped in the package."""
    return Path(__file__).with_name(SIMULATOR_INIT_STATE_DEFAULT)


def resolve_init_state_path(path: str) -> str:
    """Resolve the simulator init-state .toml to load.

    An existing file — an explicit ``--simulator-init-state`` path, or a
    ``simulator_init_state.toml`` in the working directory — is used as
    given. When the default is requested and no such file is present,
    fall back to the world shipped inside the package, so the no-hardware
    path (``dotbot run simulator`` / ``--conn simulator``) works from any directory
    and from a pip-installed wheel. An explicit path that does not exist
    is returned unchanged so the caller gets a clear FileNotFoundError.
    """
    if Path(path).is_file():
        return path
    if path == SIMULATOR_INIT_STATE_DEFAULT:
        return str(packaged_init_state_path())
    return path


class DotBotSimulatorCommunicationInterface:
    """Bidirectional serial interface to control simulated robots"""

    def __init__(self, on_frame_received: Callable, simulator_init_state: str):
        self.queue = queue.Queue()
        self.on_frame_received = on_frame_received
        self._stp_event = threading.Event()
        self.main_thread = threading.Thread(target=self.run, daemon=True)
        init_state = InitStateToml(
            **toml.load(resolve_init_state_path(simulator_init_state))
        )
        self._network = init_state.network
        self.dotbots = [
            DotBotSimulator(
                settings=dotbot_settings,
                tx_queue=self.queue,
            )
            for dotbot_settings in init_state.dotbots
        ]
        self._dotbot_modes = [s.network_mode for s in init_state.dotbots]
        self._address_to_index = {d.address: i for i, d in enumerate(self.dotbots)}
        self._mari = None
        mari_indices = [
            i for i, m in enumerate(self._dotbot_modes) if m == SimulatedNetworkMode.MARI
        ]
        if mari_indices:
            self._mari = MariSimulator(
                settings=self._network,
                dotbots=self.dotbots,
                mari_indices=mari_indices,
                on_frame_received=self.on_frame_received,
            )

        self.logger = LOGGER.bind(context=__name__)

    def start(self):
        for dotbot in self.dotbots:
            dotbot.start()
        if self._mari is not None:
            self._mari.start()
        self.main_thread.start()
        self.logger.info("DotBot Simulation Started")

    def run(self):
        """Listen continuously at each byte received on the fake serial interface."""
        while self._stp_event.is_set() is False:
            frame = self.queue.get()
            if frame is None:
                break
            self.handle_dotbot_frame(frame)

    def stop(self):
        self.logger.info("Stopping DotBot Simulation...")
        self._stp_event.set()
        self.queue.put_nowait(None)  # unblock the run thread if waiting on the queue
        for dotbot in self.dotbots:
            dotbot.stop()
        if self._mari is not None:
            self._mari.stop()
        self.main_thread.join()

    def flush(self):
        """Flush fake serial output."""
        pass

    def _packet_delivered(self, pdr: int) -> bool:
        return random.randint(0, 100) <= pdr

    def handle_dotbot_frame(self, frame):
        """Send bytes to the fake serial, similar to the real gateway."""
        addr = addr_to_hex(int(frame.header.source))
        index = self._address_to_index.get(addr, 0)
        if self._dotbot_modes[index] == SimulatedNetworkMode.MARI:
            self._mari.schedule_uplink(frame, index)
            return
        if not self._packet_delivered(self._network.pdr):
            self.logger.info(
                f"Packet from DotBot {addr_to_hex(int(frame.header.source))} lost in simulation"
            )
            return
        self.on_frame_received(frame)

    def write(self, bytes_):
        """Write bytes on the fake serial."""
        for index, dotbot in enumerate(self.dotbots):
            if self._dotbot_modes[index] == SimulatedNetworkMode.MARI:
                self._mari.schedule_downlink(bytes_, dotbot, index)
                continue
            if not self._packet_delivered(self._network.pdr):
                self.logger.info(
                    f"Packet to DotBot {dotbot.address} lost in simulation"
                )
                continue
            dotbot.queue.put_nowait(Frame.from_bytes(bytes_))
