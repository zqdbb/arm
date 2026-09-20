#!/usr/bin/env python3
"""
雅格美天 喷漆机器人 — 自动管线入口

用法:
  python auto_run.py --file chair_scan.ply                    # 自动分类+分割+可视化
  python auto_run.py --file points.npy                        # numpy 格式
  python auto_run.py --file scan.xyz                          # 纯文本 xyz
  python auto_run.py --file chair.ply --correction fix.yaml   # 带纠正

输入格式:
  .ply    — 标准点云 (ASCII/binary), 扫描仪/深度相机直接导出
  .npy    — NumPy 数组 (N,3), float32
  .xyz    — 纯文本, 每行 "x y z", 空格分隔

点云要求: ≥500 点, 物体居中, 任意单位
"""

import os, sys, argparse
import numpy as np


def load_points(filepath):
    """自动识别格式加载点云: .ply / .npy / .xyz"""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.ply':
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(filepath)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if len(pts) == 0:
            raise ValueError(f"PLY 文件无点云数据: {filepath}")

    elif ext == '.npy':
        pts = np.load(filepath).astype(np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"npy 需要 (N,3) 形状, 实际: {pts.shape}")

    elif ext == '.xyz':
        pts = np.loadtxt(filepath, dtype=np.float32)
        if pts.ndim == 1:
            pts = pts.reshape(-1, 3)
        if pts.shape[1] < 3:
            raise ValueError(f"xyz 需要至少 3 列 (x y z), 实际: {pts.shape}")

    else:
        raise ValueError(f"不支持的格式: {ext} (支持 .ply .npy .xyz)")

    # 归一化
    center = pts.mean(0)
    pts = pts - center
    scale = np.abs(pts).max()
    if scale > 1e-10:
        pts = pts / scale

    print(f"[加载] {filepath} → {len(pts)} 点")
    return pts


def main():
    parser = argparse.ArgumentParser(
        description='雅格美天 喷漆机器人 — 自动管线',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python auto_run.py --file chair.ply
  python auto_run.py --file table.npy
  python auto_run.py --file cabinet.xyz
  python auto_run.py --file chair.ply --correction correction.yaml
        """)
    parser.add_argument('--file', type=str, required=True,
                        help='输入点云 (.ply/.npy/.xyz)')
    parser.add_argument('--correction', type=str, default=None,
                        help='纠正规则 YAML 文件')
    parser.add_argument('--html', action='store_true',
                        help='同时生成交互式 HTML (默认仅 PNG)')
    args = parser.parse_args()

    # 导入管线
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from spray_pipeline import auto_pipeline

    # 加载点云
    try:
        points = load_points(args.file)
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        sys.exit(1)

    if len(points) < 100:
        print(f"✗ 点数太少 ({len(points)}), 需要 ≥100")
        sys.exit(1)

    # 运行
    try:
        result = auto_pipeline(
            points,
            correction=args.correction,
            visualize=True,
            save_html=args.html,
        )
    except Exception as e:
        print(f"\n✗ 管线异常: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 总结
    print(f"\n{'='*60}")
    print(f"  完成")
    print(f"  类别: {result.get('_auto_category', '?')}")
    print(f"  分割方式: {'模型+几何' if result.get('_model_based') else '几何'}")
    print(f"  部件数: {result.get('n_regions', 0)}")
    print(f"  自检: {result.get('_validation_status', '?')}")
    print(f"  路径: {len(result.get('waypoints', []))}点, "
          f"{len(result.get('segments', []))}段")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
