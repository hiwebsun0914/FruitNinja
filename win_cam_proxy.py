# win_cam_proxy.py
import os
import threading
import time

import cv2
from flask import Flask, Response

app = Flask(__name__)

CAM_INDEX = int(os.environ.get("WCAM_INDEX", 1))
JPEG_QUALITY = int(os.environ.get("WCAM_JPEG_QUALITY", 80))
REOPEN_FAILS = int(os.environ.get("WCAM_REOPEN_FAILS", 30))
REOPEN_DELAY = float(os.environ.get("WCAM_REOPEN_DELAY", 0.5))
CAM_WIDTH = int(os.environ.get("WCAM_WIDTH", 0))
CAM_HEIGHT = int(os.environ.get("WCAM_HEIGHT", 0))
CAM_FPS = float(os.environ.get("WCAM_FPS", 0))
CAM_BUFFERSIZE = int(os.environ.get("WCAM_BUFFERSIZE", 1))
CAM_FOURCC = os.environ.get("WCAM_FOURCC", "").strip()
STREAM_FPS = float(os.environ.get("WCAM_STREAM_FPS", 0))

_cap_lock = threading.Lock()
_cap = None
_frame_lock = threading.Lock()
_latest_frame = None
_capture_started = False
_capture_lock = threading.Lock()


def apply_camera_settings(cap):
    if CAM_FOURCC and len(CAM_FOURCC) == 4:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*CAM_FOURCC))
    if CAM_WIDTH > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    if CAM_HEIGHT > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if CAM_FPS > 0:
        cap.set(cv2.CAP_PROP_FPS, CAM_FPS)
    if CAM_BUFFERSIZE > 0:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, CAM_BUFFERSIZE)


def open_camera():
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    if cap.isOpened():
        apply_camera_settings(cap)
    else:
        cap.release()
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_MSMF)
    if cap.isOpened():
        apply_camera_settings(cap)
    else:
        cap.release()
        return None
    return cap


def ensure_camera(force: bool = False):
    global _cap
    with _cap_lock:
        if force or _cap is None or not _cap.isOpened():
            if _cap is not None:
                try:
                    _cap.release()
                except Exception:
                    pass
            _cap = open_camera()
    return _cap


def read_frame():
    with _cap_lock:
        if _cap is None or not _cap.isOpened():
            return False, None
        return _cap.read()


def capture_loop():
    global _latest_frame
    fail_count = 0
    while True:
        if ensure_camera() is None:
            time.sleep(REOPEN_DELAY)
            continue
        ok, frame = read_frame()
        if not ok or frame is None:
            fail_count += 1
            if fail_count >= REOPEN_FAILS:
                ensure_camera(force=True)
                fail_count = 0
                time.sleep(REOPEN_DELAY)
            else:
                time.sleep(0.01)
            continue
        fail_count = 0
        with _frame_lock:
            _latest_frame = frame


def start_capture_thread():
    global _capture_started
    with _capture_lock:
        if _capture_started:
            return
        _capture_started = True
        thread = threading.Thread(target=capture_loop, daemon=True)
        thread.start()


def mjpeg_gen():
    start_capture_thread()
    delay = 1.0 / STREAM_FPS if STREAM_FPS > 0 else 0.0
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame is None:
            time.sleep(0.02)
            continue
        loop_start = time.time()
        ret, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ret:
            continue
        payload = jpg.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n'
            b'Content-Length: ' + str(len(payload)).encode('ascii') + b'\r\n\r\n' +
            payload + b'\r\n'
        )
        if delay > 0:
            elapsed = time.time() - loop_start
            if elapsed < delay:
                time.sleep(delay - elapsed)

@app.route('/video')
def video():
    return Response(mjpeg_gen(), content_type="multipart/x-mixed-replace;boundary=frame")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
# # win_cam_proxy.py
# import cv2, threading, time
# from flask import Flask, Response, request, jsonify, make_response
#
# app = Flask(__name__)
#
# # ---------- 全局相机 ----------
# _cap_lock = threading.Lock()
# _cap = None
# _cam_index = 0
# _width = None
# _height = None
# _fps_limit = None  # e.g., 15
#
# def _open_camera(index=0, width=None, height=None):
#     """在 Windows 上优先 DSHOW，失败再用 MSMF。"""
#     cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
#     if not cap.isOpened():
#         cap.release()
#         cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
#
#     if not cap.isOpened():
#         raise RuntimeError(f"无法打开摄像头索引 {index}（DSHOW/MSMF 都失败）")
#
#     # 可选分辨率设置
#     if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(width))
#     if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
#     return cap
#
# def _ensure_cap(index=None, width=None, height=None):
#     global _cap, _cam_index, _width, _height
#     with _cap_lock:
#         need_reopen = (
#             _cap is None or
#             (index is not None and index != _cam_index) or
#             (width is not None and width != _width) or
#             (height is not None and height != _height) or
#             (not _cap.isOpened())
#         )
#         if need_reopen:
#             # 关闭旧的
#             if _cap is not None:
#                 try: _cap.release()
#                 except: pass
#             _cap = _open_camera(index if index is not None else _cam_index,
#                                 width if width is not None else _width,
#                                 height if height is not None else _height)
#             _cam_index = _cam_index if index is None else index
#             _width    = _width if width is None else width
#             _height   = _height if height is None else height
#     return _cap
#
# def _read_frame():
#     with _cap_lock:
#         if _cap is None or not _cap.isOpened():
#             return False, None
#         return _cap.read()
#
# # ---------- MJPEG 生成器 ----------
# def mjpeg_gen(boundary=b"frame", fps_limit=None):
#     """生成 multipart/x-mixed-replace 流。"""
#     min_interval = 0.0
#     if fps_limit and fps_limit > 0:
#         min_interval = 1.0 / float(fps_limit)
#
#     while True:
#         t0 = time.time()
#         ok, frame = _read_frame()
#         if not ok or frame is None:
#             # 给客户端一个短暂的空转，避免死循环占满 CPU
#             time.sleep(0.02)
#             continue
#         ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#         if not ok:
#             continue
#
#         # 注意：HTTP 头里写了 boundary=frame，则每个 part 以 --frame 开头
#         yield (
#             b"--" + boundary + b"\r\n"
#             b"Content-Type: image/jpeg\r\n"
#             b"Cache-Control: no-cache\r\n"
#             b"Pragma: no-cache\r\n"
#             b"\r\n" + jpg.tobytes() + b"\r\n"
#         )
#
#         if min_interval > 0:
#             el = time.time() - t0
#             if el < min_interval:
#                 time.sleep(min_interval - el)
#
# # ---------- 路由 ----------
# @app.route("/video")
# @app.route("/stream.mjpg")
# def video():
#     """MJPEG 流。可选 query:
#       cam=0/1/2  选择摄像头索引
#       w=1280     目标宽
#       h=720      目标高
#       fps=15     限制 MJPEG 推送帧率
#     """
#     try:
#         cam = int(request.args.get("cam",  _cam_index))
#         w   = request.args.get("w",   None)
#         h   = request.args.get("h",   None)
#         fps = request.args.get("fps", None)
#         w = int(w) if w else None
#         h = int(h) if h else None
#         fps = int(fps) if fps else None
#
#         _ensure_cap(cam, w, h)
#         boundary = b"frame"
#         resp = Response(
#             mjpeg_gen(boundary=boundary, fps_limit=fps),
#             mimetype="multipart/x-mixed-replace; boundary=frame"
#         )
#         # 关闭缓存，提升实时性
#         resp.headers["Cache-Control"] = "no-cache, private"
#         resp.headers["Pragma"] = "no-cache"
#         return resp
#     except Exception as e:
#         return make_response(f"open camera failed: {e}", 500)
#
# @app.route("/shot.jpg")
# def shot():
#     """单帧 JPEG 快照（给不支持 MJPEG 的客户端）"""
#     ok, frame = _read_frame()
#     if not ok or frame is None:
#         return make_response("no frame", 503)
#     ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
#     if not ok:
#         return make_response("encode fail", 500)
#     resp = Response(jpg.tobytes(), mimetype="image/jpeg")
#     resp.headers["Cache-Control"] = "no-cache, private"
#     resp.headers["Pragma"] = "no-cache"
#     return resp
#
# @app.route("/info")
# def info():
#     """简单诊断信息"""
#     with _cap_lock:
#         opened = (_cap is not None and _cap.isOpened())
#         w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  if opened else 0
#         h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
#         fps = _cap.get(cv2.CAP_PROP_FPS) if opened else 0
#     return jsonify({
#         "opened": opened,
#         "cam_index": _cam_index,
#         "width": w, "height": h, "fps": fps
#     })
#
# @app.after_request
# def add_cors(resp):
#     # 简单 CORS，方便跨端调试
#     resp.headers["Access-Control-Allow-Origin"] = "*"
#     return resp
#
# # ---------- 主入口 ----------
# if __name__ == "__main__":
#     # 首次打开默认相机（你可以改默认索引）
#     try:
#         _ensure_cap(index=1)
#     except Exception as e:
#         print(f"[warn] 启动时未能打开摄像头：{e}")
#     # 在 0.0.0.0 上监听，给 WSL/局域网访问
#     # 生产部署请改用 gunicorn/waitress
#     app.run(host="0.0.0.0", port=8080, threaded=True)
