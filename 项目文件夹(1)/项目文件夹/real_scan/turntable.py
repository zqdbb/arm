"""Y200RA60 电动转台控制模块 (1SC控制器).

协议: RS232/CH340, 115200 8N1
转台参数: 1000脉冲/度, 最大速度 50°/s (50000Hz)
"""

import serial
import struct
import time


# 转台机械参数 (可覆盖: from config import *)
PULSES_PER_DEGREE = 400        # ★ 每度脉冲数, 调这个校准角度
                                #   400 → 8细分(最可能), 实际值需微调
MAX_SPEED = 50000              # 最大速度 Hz
MIN_SPEED = 20                 # 最小速度 Hz
DEFAULT_SPEED = 20000          # 默认速度 Hz

# 尝试从 config 覆盖
try:
    from config import TURNTABLE_PULSES_PER_DEG as _ppd
    PULSES_PER_DEGREE = _ppd
except ImportError:
    pass


class TurntableController:
    """Y200RA60 电动转台控制器."""

    def __init__(self, port='/dev/ttyUSB0', baud=115200, timeout=2.0):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self._ser = None

    # ── 连接管理 ──────────────────────

    def open(self):
        self._ser = serial.Serial(self.port, self.baud, timeout=self.timeout)

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    # ── 底层通信 ──────────────────────

    def _send(self, cmd_body: bytes):
        """发送指令, 自动加帧头帧尾."""
        full = bytes([0x55, 0xAA]) + cmd_body + bytes([0xC3])
        self._ser.reset_input_buffer()
        self._ser.write(full)

    def _read(self, n: int) -> bytes:
        return self._ser.read(n)

    # ── 状态查询 ──────────────────────

    def get_status(self) -> dict:
        """读取控制器状态, 返回字典."""
        self._send(bytes([0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]))
        time.sleep(0.05)
        data = self._read(20)
        if len(data) < 8:
            # 重试一次
            time.sleep(0.1)
            self._send(bytes([0x0C, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]))
            time.sleep(0.05)
            data = self._read(20)
        if len(data) < 8:
            return {}

        pos = int.from_bytes(data[0:4], 'big', signed=True)
        speed = int.from_bytes(data[4:6], 'big')
        input_byte = data[6]
        state_byte = data[7]

        states = {0x00: 'poweron', 0x01: 'moving', 0x04: 'manual', 0x08: 'homing', 0x20: 'stopped'}

        return {
            'position': pos,          # 脉冲
            'degrees': pos / PULSES_PER_DEGREE,
            'speed': speed,           # Hz
            'input': {
                'pos_limit':  bool((input_byte >> 2) & 1),  # 0=有效
                'zero':       bool((input_byte >> 3) & 1),
                'neg_limit':  bool((input_byte >> 4) & 1),
                'input4':     bool((input_byte >> 5) & 1),
                'raw':        input_byte,
            },
            'state': state_byte,
            'state_name': states.get(state_byte, f'0x{state_byte:02X}'),
        }

    # ── 运动控制 ──────────────────────

    def _speed_bytes(self, speed: int) -> bytes:
        speed = max(MIN_SPEED, min(MAX_SPEED, speed))
        return speed.to_bytes(2, 'little')

    def _steps_bytes(self, steps: int) -> bytes:
        return steps.to_bytes(4, 'little', signed=True)

    def move_absolute(self, degrees: float, speed: int = DEFAULT_SPEED):
        """绝对运动到指定角度(度)."""
        steps = int(degrees * PULSES_PER_DEGREE)
        self._send(
            bytes([0x07]) +
            self._speed_bytes(speed) +
            self._steps_bytes(steps)
        )

    def move_relative(self, degrees: float, speed: int = DEFAULT_SPEED):
        """相对运动指定角度(度), 正=顺时针."""
        steps = int(degrees * PULSES_PER_DEGREE)
        self._send(
            bytes([0x08]) +
            self._speed_bytes(speed) +
            self._steps_bytes(steps)
        )

    def set_coordinate(self, degrees: float):
        """设置当前坐标为零位(电机不动作)."""
        steps = int(degrees * PULSES_PER_DEGREE)
        self._send(
            bytes([0x09]) +
            self._speed_bytes(DEFAULT_SPEED) +
            self._steps_bytes(steps)
        )

    def rotate_forward(self, speed: int = DEFAULT_SPEED):
        """正向持续旋转(不指定终点), 用 stop() 停止."""
        self._send(
            bytes([0x06, 0x09]) +
            self._speed_bytes(speed) +
            bytes([0x00, 0x00, 0x00])
        )

    def rotate_reverse(self, speed: int = DEFAULT_SPEED):
        """反向持续旋转, 用 stop() 停止."""
        self._send(
            bytes([0x06, 0x0A]) +
            self._speed_bytes(speed) +
            bytes([0x00, 0x00, 0x00])
        )

    def stop(self):
        """立即停止."""
        self._send(bytes([0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00]))

    # ── 回零 ──────────────────────

    def home(self, speed: int = 10000, direction: str = 'positive'):
        """回机械零位. direction: 'positive' 或 'negative'."""
        dir_byte = 0x09 if direction == 'positive' else 0x0A
        self._send(
            bytes([0x0B, dir_byte]) +
            self._speed_bytes(speed) +
            bytes([0x00, 0x00, 0x00])
        )

    # ── I/O 控制 ──────────────────────

    def set_output(self, port: int, on: bool):
        """设置输出口. port=1或2, on=True=打开."""
        self._send(
            bytes([0x0D, port, 0x01 if on else 0x00]) +
            bytes([0x00, 0x00, 0x00, 0x00])
        )

    def set_ramp(self, rise: int, down: int):
        """动态设置升降速(不保存)."""
        self._send(
            bytes([0x0A]) +
            rise.to_bytes(2, 'little') +
            down.to_bytes(2, 'little') +
            bytes([0x00, 0x00])
        )

    # ── 高级操作 ──────────────────────

    def wait_stop(self, poll_interval: float = 0.1, timeout: float = 120.0):
        """轮询等待直到电机停止."""
        time.sleep(0.15)  # 给电机启动时间
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self.get_status()
            if status.get('state_name') in ('stopped', 'poweron'):
                return status
            time.sleep(poll_interval)
        raise TimeoutError("电机未在超时时间内停止")

    def move_and_wait(self, degrees: float, speed: int = DEFAULT_SPEED, timeout: float = 120.0):
        """相对运动并等待完成."""
        self.move_relative(degrees, speed)
        return self.wait_stop(timeout=timeout)

    def scan_360(self, num_frames: int, speed: int = DEFAULT_SPEED):
        """360度等角度旋转, 返回各角度位置的生成器."""
        angle_per_frame = 360.0 / num_frames
        for i in range(num_frames):
            yield i * angle_per_frame
            if i < num_frames - 1:
                self.move_and_wait(angle_per_frame, speed)

    def zero(self):
        """将当前位置设为坐标零点."""
        self.set_coordinate(0.0)
