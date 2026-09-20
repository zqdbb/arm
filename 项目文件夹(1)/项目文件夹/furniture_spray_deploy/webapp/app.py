#!/usr/bin/env python3
"""家具喷漆 Web 服务 — Flask REST API + 前端页面."""

import io, json, os, sys, uuid, time, threading, base64
import numpy as np
import torch
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'ultralytics'))
from pipeline_worker import (
    load_and_normalize, run_classify_with_regions, run_segment_from_cache,
    run_segment_ai, generate_path_for_part,
    _MESH_EXTS
)

# ── 图片分类器 (YOLOv11n, 模块级加载) ──
from ultralytics import YOLO
_MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
_WEIGHTS_PATH = os.path.join(_MODEL_DIR, 'best.pt')

_image_model = YOLO(_WEIGHTS_PATH)
_yolo_names = _image_model.names
IMAGE_CLASSES = [_yolo_names[i] for i in sorted(_yolo_names.keys())]

print(f'[ImageClassifier] YOLOv11 已加载: {len(IMAGE_CLASSES)} 类 {IMAGE_CLASSES}')

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# ---- 内存 job store ----
_jobs = {}        # job_id → {points, classify_result, created_at}
_lock = threading.Lock()
_JOB_TTL = 1800   # 30 分钟过期

ALLOWED_EXTENSIONS = {'.ply', '.npy', '.xyz', '.pcd', '.pts',
                        '.obj', '.stl', '.off', '.glb', '.gltf'}


def _cleanup_expired():
    """清理过期 job."""
    now = time.time()
    with _lock:
        expired = [jid for jid, j in _jobs.items()
                   if now - j['created_at'] > _JOB_TTL]
        for jid in expired:
            del _jobs[jid]


# ---- 错误处理 ----
@app.errorhandler(400)
def bad_request(e):
    return jsonify({'error': 'bad_request', 'detail': str(e)}), 400


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'file_too_large', 'max_mb': 50}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'not_found'}), 404


# ---- API ----
@app.route('/api/health')
def health():
    _cleanup_expired()
    return jsonify({'status': 'ok', 'jobs_active': len(_jobs)})


@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'no_file', 'detail': '请选择文件上传'}), 400

    f = request.files['file']
    if not f.filename:
        return jsonify({'error': 'no_filename'}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            'error': 'unsupported_format',
            'allowed': list(ALLOWED_EXTENSIONS),
            'detail': f'不支持 {ext} 格式. 点云: .ply .npy .xyz .pcd .pts | 网格: .obj .stl .off .glb .gltf'
        }), 400

    # 保存临时文件
    tmp_path = os.path.join('/tmp', f'webapp_{uuid.uuid4().hex}{ext}')
    f.save(tmp_path)

    try:
        points, n_orig = load_and_normalize(tmp_path)
    except Exception as e:
        os.remove(tmp_path)
        return jsonify({'error': 'parse_error', 'detail': str(e)}), 400
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    if len(points) < 100:
        return jsonify({
            'error': 'insufficient_points',
            'n_points': int(len(points)),
            'min_required': 100
        }), 400

    job_id = uuid.uuid4().hex
    with _lock:
        _jobs[job_id] = {
            'points': points,
            'classify_result': None,
            'created_at': time.time(),
        }

    # 构建预览点云 (最多5000点, base64)
    preview_pts = points
    if len(points) > 5000:
        idx = np.random.choice(len(points), 5000, replace=False)
        preview_pts = points[idx]
    preview_b64 = base64.b64encode(preview_pts.astype(np.float32).tobytes()).decode('ascii')

    return jsonify({
        'job_id': job_id,
        'status': 'loaded',
        'n_points': int(len(points)),
        'n_original': int(n_orig),
        'downsampled': n_orig != len(points),
        'file_ext': ext,
        'sampled_from_mesh': ext in _MESH_EXTS,
        'preview_b64': preview_b64,
    })


@app.route('/api/classify', methods=['POST'])
def classify():
    data = request.get_json(silent=True) or {}
    job_id = data.get('job_id')
    retry = data.get('retry', False)

    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'job_not_found', 'job_id': job_id}), 404

    # retry 模式: 传入上次分类结果, 强制换逻辑重新分类
    retry_category = None
    if retry and job.get('classify_result'):
        retry_category = job['classify_result']['category']

    try:
        result = run_classify_with_regions(job['points'], retry_category=retry_category)
    except Exception as e:
        return jsonify({'error': 'classification_failed', 'detail': str(e)}), 500

    regions_cache = result.pop('_regions_cache')
    with _lock:
        job['classify_result'] = result
        job['_regions_cache'] = regions_cache

    return jsonify(result)


@app.route('/api/segment', methods=['POST'])
def segment():
    data = request.get_json(silent=True) or {}
    job_id = data.get('job_id')

    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'job_not_found', 'job_id': job_id}), 404
    if not job.get('_regions_cache'):
        return jsonify({'error': 'not_classified',
                        'detail': '请先执行分类 (/api/classify)'}), 400

    # 允许前端传 category 覆盖自动分类结果
    override = data.get('category', '').strip()
    category = override if override else job['classify_result']['category']
    try:
        result = run_segment_from_cache(job['_regions_cache'], category)
    except Exception as e:
        return jsonify({'error': 'segmentation_failed', 'detail': str(e)}), 500

    # 附加分类信息
    result['category'] = category
    result['confidence'] = 1.0 if override else job['classify_result']['confidence']
    result['category_overridden'] = bool(override)
    result['geometry_class'] = job['classify_result']['geometry_class']
    result['geometry_class_info'] = job['classify_result']['geometry_class_info']

    return jsonify(result)


# ---- 图片分类 ----
@app.route('/api/classify-image', methods=['POST'])
def classify_image():
    """上传家具照片 → YOLOv11 识别 (cabinet/chair/table)."""
    if 'image' not in request.files:
        return jsonify({'error': 'no_image', 'detail': '请上传图片'}), 400

    file = request.files['image']
    if not file.filename:
        return jsonify({'error': 'empty_filename'}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception:
        return jsonify({'error': 'invalid_image'}), 400

    results = _image_model(img, verbose=False)[0]

    # 取置信度最高的检测结果
    if results.boxes is not None and len(results.boxes) > 0:
        top_idx = int(results.boxes.cls[0])
        top_conf = float(results.boxes.conf[0])
    elif results.probs is not None:
        top_idx = int(results.probs.top1)
        top_conf = float(results.probs.top1conf)
    else:
        top_idx = 0
        top_conf = 0.0

    pred_class = _yolo_names.get(top_idx, 'unknown')

    # 构建各类别置信度
    conf_by_class = {}
    if results.boxes is not None:
        for cls_id, conf in zip(results.boxes.cls.tolist(), results.boxes.conf.tolist()):
            cid = int(cls_id)
            conf_by_class[cid] = max(conf_by_class.get(cid, 0), conf)
    all_results = []
    for cid in sorted(_yolo_names.keys()):
        all_results.append({
            'class': _yolo_names[cid],
            'prob': round(float(conf_by_class.get(cid, 0)) * 100, 1),
        })
    all_results.sort(key=lambda x: x['prob'], reverse=True)

    return jsonify({
        'prediction': pred_class,
        'confidence': round(float(top_conf) * 100, 1),
        'all': all_results,
    })


# ---- AI 分割 ----
@app.route('/api/segment-ai', methods=['POST'])
def segment_ai():
    """使用 AI 模型 (PointNeXt/PointCNN) 进行部件分割."""
    data = request.get_json(silent=True) or {}
    job_id = data.get('job_id')
    category = data.get('category', '').strip()

    if not category:
        return jsonify({'error': 'no_category',
                        'detail': '请先通过图片分类确定家具类别'}), 400

    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'job_not_found', 'job_id': job_id}), 404

    try:
        result = run_segment_ai(job['points'], category)
    except ValueError as e:
        return jsonify({'error': 'unsupported_category', 'detail': str(e)}), 400
    except Exception as e:
        return jsonify({'error': 'ai_segmentation_failed', 'detail': str(e)}), 500

    # 缓存原始 regions 供后续路径生成
    regions = result.pop('_regions', None)
    if regions is not None:
        with _lock:
            job['_ai_regions'] = regions
            job['category'] = category

    result['category'] = category
    result['confidence'] = 1.0
    return jsonify(result)


# ---- 路径生成 ----
@app.route('/api/generate-path', methods=['POST'])
def generate_path():
    """为指定部件生成喷涂路径."""
    data = request.get_json(silent=True) or {}
    job_id = data.get('job_id')
    part_index = data.get('part_index', 0)
    spacing = data.get('spacing', 0.06)
    spray_distance = data.get('spray_distance', 0.15)

    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({'error': 'job_not_found', 'job_id': job_id}), 404

    regions = job.get('_ai_regions')
    if not regions:
        return jsonify({'error': 'no_regions',
                        'detail': '请先执行 AI 分割 (/api/segment-ai)'}), 400

    if part_index < 0 or part_index >= len(regions):
        return jsonify({'error': 'invalid_part_index',
                        'detail': f'部件索引 {part_index} 超出范围 [0, {len(regions)-1}]'}), 400

    region = regions[part_index]
    try:
        wp, norms, segs = generate_path_for_part(region, spacing, spray_distance)
    except Exception as e:
        return jsonify({'error': 'path_generation_failed', 'detail': str(e)}), 500

    if wp is None or len(wp) == 0:
        return jsonify({
            'part_index': part_index,
            'part_name': region.get('part_name', '?'),
            'waypoints_b64': '',
            'segments': [],
            'n_waypoints': 0,
            'path_type': region['type'],
            'empty': True,
            'detail': '该部件无法生成有效路径',
        })

    wp_b64 = base64.b64encode(wp.astype(np.float32).tobytes()).decode('ascii')
    norms_b64 = base64.b64encode(norms.astype(np.float32).tobytes()).decode('ascii')

    return jsonify({
        'part_index': part_index,
        'part_name': region.get('part_name', '?'),
        'waypoints_b64': wp_b64,
        'normals_b64': norms_b64,
        'segments': segs.tolist() if len(segs) > 0 else [],
        'n_waypoints': int(len(wp)),
        'path_type': region['type'],
        'spray_strategy': region.get('spray', '?'),
    })


# ---- 前端 ----
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/pathplan')
def pathplan():
    return render_template('pathplan.html')


@app.route('/segview')
def segview():
    return render_template('segview.html')


if __name__ == '__main__':
    # 启动后台清理线程
    def _bg_cleanup():
        while True:
            time.sleep(300)  # 每 5 分钟清理一次
            _cleanup_expired()

    t = threading.Thread(target=_bg_cleanup, daemon=True)
    t.start()

    print("\n" + "=" * 60)
    print("  雅格美天 喷漆机器人 — Web 可视化平台")
    print(f"  地址: http://localhost:5000")
    print(f"  按 Ctrl+C 退出")
    print("=" * 60 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)
