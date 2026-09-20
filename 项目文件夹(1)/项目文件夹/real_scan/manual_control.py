"""
══════════════════════════════════════════════════════════════
  Y200RA60 电动转台控制程序
  用法: 直接点 VSCode 右上角 ▶ 运行, 或在终端 python3 manual_control.py
══════════════════════════════════════════════════════════════

【角度校准】
  如果指令角度 ≠ 实际角度, 改 config.py 里的 TURNTABLE_PULSES_PER_DEG:
     实际转了 90°, 但指令是 30° → 把值改小 (当前值的 30/90)
     实际转了 30°, 但指令是 90° → 把值改大 (当前值的 90/30)
  当前值: 400 (8细分)

【方向】
  如果正数转反了, 改下面的 DIRECTION = -1
"""

# ╔══════════════════════════════════════════════════════════╗
# ║                                                          ║
# ║    ★★★ 参数修改区 ★★★                                    ║
# ║    (改这里的数字, 然后点 ▶ 运行, 效果立即可见)              ║
# ║                                                          ║
# ╚══════════════════════════════════════════════════════════╝

# ── 基础参数 ──────────────────────────────

SERIAL_PORT = '/dev/ttyUSB0'   # 串口路径, 一般不用改
SPEED = 14400                  # 转动速度 (Hz), 范围 20~50000
                               #   14400 ≈ 36°/s = 10秒/圈 (基于400脉冲/度)

DIRECTION = -1                  # 方向: 1=正向, -1=反转
                               #   如果发现正数转反了, 改这个就行

# ── 脉冲校准 ────────────────────────────
# ★ 如果指令角度和实际角度对不上, 改这个值 ★
# 不需要在这改, 去 config.py 改 TURNTABLE_PULSES_PER_DEG 就行
#   值变大 → 同样指令转得少
#   值变小 → 同样指令转得多

# ── 运行模式选择 ────────────────────────

# 运行模式, 可选:
#MODE = 'menu'                   #   'menu'       — 交互菜单, 手动输入角度
#MODE = 'sequence'               #   'sequence'   — 按角度列表自动旋转
#MODE = 'scan'                   #   'scan'       — 360° 等分扫描 (配合相机拍照)
#MODE = 'single'                 #   'single'     — 只做一次旋转演示
MODE = 'continuous'              #   'continuous' — 持续匀速旋转, Ctrl+C 停止

# ── 启动标定 ────────────────────────────
STARTUP_CALIBRATE = False         # True=启动时先转一圈归零标定
STARTUP_CAL_SPEED = 14400         # 标定时转动速度 Hz (慢一点, 好看清)

# ── 退出行为 ────────────────────────────
RETURN_TO_ZERO = False             # True=退出时自动归零, False=直接停止

# ── 模式参数 ────────────────────────────

# 当 MODE='single' 时:
SINGLE_ANGLE = 30              # 单次旋转的角度 (度)
SINGLE_WAIT = 2                # 旋转后等待秒数

# 当 MODE='sequence' 时:
SEQUENCE_ANGLES = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]  # 要转到的绝对角度列表
SEQ_PAUSE = 2                # 每步之间暂停秒数 (0 表示不停)

# 当 MODE='scan' 时:
SCAN_FRAMES = 12               # 360° 拍多少张照片 (12=每30°一张)
SCAN_PAUSE = 1.0               # 每个角度停留秒数 (给相机拍照时间)
SCAN_SPEED = 20000             # 扫描时转动速度 (建议快一些)

# 当 MODE='continuous' 时:
CONTINUOUS_DIRECTION = 1        # 1=正向旋转, -1=反向旋转
# 10秒一圈 = 36°/s → 36 * 400 = 14400 Hz, 根据实际校准值微调

# ╔══════════════════════════════════════════════════════════╗
# ║    以下为代码, 一般不用改                                  ║
# ╚══════════════════════════════════════════════════════════╝

import sys, time, signal, os
sys.path.insert(0, '/home/xie/项目文件夹/real_scan')
from turntable import TurntableController, PULSES_PER_DEGREE


_exit_requested = False
_signal_received = None  # 记录收到的信号类型


def _deg(s):
    """显示角度: DIRECTION=-1 时翻转符号, 保证物理方向对且显示也正确."""
    d = s.get('degrees', 0.0) if isinstance(s, dict) else 0.0
    return -d if DIRECTION == -1 else d


def _sdeg(s):
    """返回格式化的角度字符串."""
    return f"{_deg(s):.2f}°"


def _on_exit_signal(signum, frame):
    """捕获退出信号, 标志退出."""
    global _exit_requested, _signal_received
    _signal_received = signum
    sig_name = dict(signal.Signals).get(signum, str(signum))
    print(f'\n收到信号 {sig_name}, 正在处理...')
    _exit_requested = True


def _run_with_cleanup(func, tt):
    """执行主逻辑, 退出时根据 RETURN_TO_ZERO 决定是否归零."""
    global _signal_received
    try:
        func(tt)
    finally:
        if RETURN_TO_ZERO:
            print('\n返回零点...', end=' ')
            tt.stop()
            time.sleep(0.2)
            tt.move_absolute(0, speed=SPEED)
            tt.wait_stop(timeout=30)
            print('完成.')
        else:
            tt.stop()
            print('\n已停止.')
        # Ctrl+Z: 归零后重新挂起进程
        if _signal_received == signal.SIGTSTP:
            signal.signal(signal.SIGTSTP, signal.SIG_DFL)
            print('进程已挂起, 用 fg 恢复.')
            os.kill(os.getpid(), signal.SIGTSTP)


def mode_single(tt):
    """单次旋转演示."""
    tt.zero()
    angle = SINGLE_ANGLE
    print(f'\n>>> 旋转 {angle}° (速度={SPEED}Hz)...')
    tt.move_and_wait(angle * DIRECTION, speed=SPEED)
    s = tt.get_status()
    print(f'>>> 到达 {_sdeg(s)}, 等待 {SINGLE_WAIT} 秒')
    time.sleep(SINGLE_WAIT)
    print(f'>>> 反转回原点...')
    tt.move_and_wait(-angle * DIRECTION, speed=SPEED)
    s = tt.get_status()
    print(f'>>> 回到 {_sdeg(s)}, 完成!')


def mode_sequence(tt):
    """按角度列表自动旋转."""
    tt.zero()
    print(f'\n序列旋转: {SEQUENCE_ANGLES}')
    for deg in SEQUENCE_ANGLES:
        print(f'  → {deg}°...', end=' ')
        tt.move_absolute(deg * DIRECTION, speed=SPEED)
        s = tt.wait_stop()
        print(f'到达 {_sdeg(s)}')
        if SEQ_PAUSE > 0:
            time.sleep(SEQ_PAUSE)
    print('序列完成!')


def mode_scan(tt):
    """360° 等分扫描."""
    tt.zero()
    angle_per_frame = 360.0 / SCAN_FRAMES
    print(f'\n扫描模式: {SCAN_FRAMES} 帧, 每帧 {angle_per_frame:.1f}°')
    for i in range(SCAN_FRAMES):
        current_angle = i * angle_per_frame
        print(f'\n[{i+1}/{SCAN_FRAMES}] 角度={current_angle:.0f}°')
        if SCAN_PAUSE > 0:
            print(f'    停留 {SCAN_PAUSE}s 等待拍照...')
            time.sleep(SCAN_PAUSE)
        if i < SCAN_FRAMES - 1:
            tt.move_and_wait(angle_per_frame * DIRECTION, speed=SCAN_SPEED)
    print('\n扫描完成!')


def mode_continuous(tt):
    """持续匀速旋转, Ctrl+C 停止.

    一次发 N_REVS 圈，提前续发，实现无缝匀速旋转。
    """
    global _exit_requested
    direction = CONTINUOUS_DIRECTION * DIRECTION
    dir_name = '正向' if direction > 0 else '反向'
    period = 360 * PULSES_PER_DEGREE / SPEED  # 秒/圈

    # 每次发 10 圈（3600°），停顿间隔从 10s 拉长到 100s
    N_REVS = 10
    chunk_deg = 360 * N_REVS
    chunk_period = period * N_REVS

    print(f'\n持续旋转: {dir_name} | 速度={SPEED}Hz | 约 {period:.1f} 秒/圈')
    print('按 Ctrl+C 停止...\n')

    # 多提前一点发，给控制器充足的缓冲时间
    tt.move_relative(chunk_deg * direction, speed=SPEED)
    next_cmd_time = time.time() + chunk_period * 0.5

    try:
        while not _exit_requested:
            now = time.time()
            if now >= next_cmd_time:
                tt.move_relative(chunk_deg * direction, speed=SPEED)
                next_cmd_time = now + chunk_period * 0.5

            s = tt.get_status()
            if s:
                print(f'\r  当前位置: {_sdeg(s)}  ', end='', flush=True)
            time.sleep(0.2)
    finally:
        print('\n停止旋转...')
        tt.stop()


def mode_menu(tt):
    """交互菜单."""
    global _exit_requested
    tt.zero()
    print(f'\n当前位置 0.00° | 速度={SPEED}Hz | 输入角度旋转, 命令:')
    print('  <数字>°  — 增转动指定角度')
    print('  a<数字>° — 绝对运动到指定角度')
    print('  s        — 查看状态')
    print('  z        — 当前位置归零')
    print('  q        — 退出')

    while not _exit_requested:
        try:
            cmd = input('\n> ').strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue
        if cmd.lower() == 'q':
            break
        if cmd.lower() == 'z':
            tt.zero()
            time.sleep(0.1)
            s = tt.get_status()
            print(f'  已归零 → {_sdeg(s)}')
            continue
        if cmd.lower() == 's':
            s = tt.get_status()
            if s:
                print(f'  位置={_sdeg(s)} 速度={s["speed"]}Hz 状态={s["state_name"]}')
            else:
                print('  状态读取失败')
            continue

        is_absolute = cmd.lower().startswith('a')
        if is_absolute:
            cmd = cmd[1:]

        try:
            deg = float(cmd.rstrip('°'))
        except ValueError:
            print('  输入错误: 例如 90 或 a45 或 q')
            continue

        if is_absolute:
            tt.move_absolute(deg * DIRECTION, speed=SPEED)
            tt.wait_stop()
        else:
            tt.move_and_wait(deg * DIRECTION, speed=SPEED)
        s = tt.get_status()
        print(f'  位置={_sdeg(s)}')


def main():
    # 注册信号处理: Ctrl+C, Ctrl+\；Ctrl+Z 会挂起进程, 归零后恢复默认行为
    signal.signal(signal.SIGINT, _on_exit_signal)
    signal.signal(signal.SIGTERM, _on_exit_signal)
    # Ctrl+Z (SIGTSTP): 归零后再挂起
    signal.signal(signal.SIGTSTP, _on_exit_signal)

    with TurntableController(port=SERIAL_PORT) as tt:
        s = tt.get_status()
        print(f'转台已连接: {_sdeg(s)}')

        if STARTUP_CALIBRATE:
            print(f'启动标定: 转一圈并归零 (速度={STARTUP_CAL_SPEED}Hz)...')
            tt.zero()
            time.sleep(0.2)
            print('  转动 360°...', end=' ')
            tt.move_and_wait(360 * DIRECTION, speed=STARTUP_CAL_SPEED)
            print('归零...', end=' ')
            tt.move_absolute(0, speed=STARTUP_CAL_SPEED)
            tt.wait_stop(timeout=60)
            print('完成.')

        if MODE == 'single':
            _run_with_cleanup(mode_single, tt)
        elif MODE == 'sequence':
            _run_with_cleanup(mode_sequence, tt)
        elif MODE == 'scan':
            _run_with_cleanup(mode_scan, tt)
        elif MODE == 'continuous':
            _run_with_cleanup(mode_continuous, tt)
        else:
            _run_with_cleanup(mode_menu, tt)

    print('程序结束.')


if __name__ == '__main__':
    main()
