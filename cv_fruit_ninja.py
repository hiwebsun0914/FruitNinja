"""
Web Fruit Ninja (MediaPipe) - browser UI with live camera stream.

Dependencies: pip install opencv-python mediapipe numpy flask
Run: python cv_fruit_ninja.py
Input: uses FN_CAMERA_STREAM_URL (default local camera).
Then open http://localhost:FN_PORT (default 8888)
"""

import math
import os
import random
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from google.protobuf import message_factory as protobuf_message_factory
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify, request, send_file


if not hasattr(protobuf_message_factory, "GetMessageClass"):
    def _compat_get_message_class(descriptor):
        factory = protobuf_message_factory.MessageFactory()
        return factory.GetPrototype(descriptor)

    protobuf_message_factory.GetMessageClass = _compat_get_message_class

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "source" / "images"
LOGO_PATH = BASE_DIR / "logo.png"

WIDTH = int(os.environ.get("FN_WIDTH", 640))
HEIGHT = int(os.environ.get("FN_HEIGHT", 480))
SPAWN_INTERVAL = float(os.environ.get("FN_SPAWN_INTERVAL", 1.1))
GRAVITY = float(os.environ.get("FN_GRAVITY", 1300))
BLADE_SPEED_THRESHOLD = float(os.environ.get("FN_BLADE_SPEED_THRESHOLD", 900))
DISPLAY_SCALE = float(os.environ.get("FN_DISPLAY_SCALE", 1.0))
MIN_PERSON_RATIO = float(os.environ.get("FN_MIN_PERSON_RATIO", 0.02))
POSE_DETECTION_CONF = float(os.environ.get("FN_DETECTION_CONF", 0.5))
POSE_TRACKING_CONF = float(os.environ.get("FN_TRACKING_CONF", 0.5))
POSE_MODEL_COMPLEXITY = int(os.environ.get("FN_MODEL_COMPLEXITY", 1))
POSE_ENABLE_SEGMENTATION = os.environ.get("FN_ENABLE_SEGMENTATION", "1").strip().lower() not in {"0", "false", "no", "off"}
JPEG_QUALITY = int(os.environ.get("FN_JPEG_QUALITY", 80))
STREAM_FPS = float(os.environ.get("FN_STREAM_FPS", 30))
HOST = os.environ.get("FN_HOST", "0.0.0.0")
PORT = int(os.environ.get("FN_PORT", 8888))
CAMERA_STREAM_URL = os.environ.get("FN_CAMERA_STREAM_URL", "local").strip()
CAMERA_SCAN_LIMIT = max(2, int(os.environ.get("FN_CAMERA_SCAN_LIMIT", 6)))

FRUIT_FILES = {
    "peach": ("fruit/peach.png", "#e6c731"),
    "sandia": ("fruit/sandia.png", "#c00000"),
    "apple": ("fruit/apple.png", "#c8e925"),
    "banana": ("fruit/banana.png", None),
    "basaha": ("fruit/basaha.png", "#c00000"),
    "boom": ("fruit/boom.png", None),
}

INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="icon" href="/logo.png" type="image/png" />
  <title>Fruit Ninja Pose Arena</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Noto+Sans+SC:wght@400;600;700&display=swap');
    :root {
      --bg: #f8f3ea;
      --ink: #1d1a17;
      --muted: #6f655d;
      --card: rgba(255, 255, 255, 0.78);
      --line: rgba(0, 0, 0, 0.08);
      --accent: #e16c2f;
      --accent-soft: rgba(225, 108, 47, 0.16);
      --accent-cool: #3aa7b8;
      --shadow: 0 24px 60px rgba(0, 0, 0, 0.12);
    }
    * {
      box-sizing: border-box;
    }
    body {
      margin: 0;
      font-family: 'Space Grotesk', 'Noto Sans SC', sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 15% 10%, rgba(255, 212, 160, 0.55), transparent 45%),
        radial-gradient(circle at 85% 0%, rgba(122, 208, 214, 0.35), transparent 40%),
        linear-gradient(135deg, #f8f3ea 0%, #f2f6f7 55%, #fbefe1 100%);
      min-height: 100vh;
      position: relative;
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background: repeating-linear-gradient(-45deg, rgba(0, 0, 0, 0.035) 0, rgba(0, 0, 0, 0.035) 1px, transparent 1px, transparent 8px);
      pointer-events: none;
      mix-blend-mode: multiply;
      opacity: 0.45;
      z-index: 0;
    }
    .page {
      position: relative;
      z-index: 1;
      max-width: 1200px;
      margin: 0 auto;
      padding: 28px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }
    .header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 24px;
      flex-wrap: wrap;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 16px;
    }
    .logo {
      width: auto;
      height: auto;
      max-width: 30vw;
      border-radius: 18px;
      background: white;
      box-shadow: 0 16px 30px rgba(0, 0, 0, 0.15);
      object-fit: contain;
    }
    .brand h1 {
      margin: 0;
      font-size: clamp(1.7rem, 2.6vw, 2.8rem);
      letter-spacing: -0.02em;
    }
    .brand p {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 0.98rem;
    }
    .status-pill {
      padding: 10px 18px;
      border-radius: 999px;
      background: var(--accent);
      color: white;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-size: 0.75rem;
      box-shadow: 0 12px 24px rgba(225, 108, 47, 0.35);
    }
    .status-pill.status-bad {
      background: #c44536;
      box-shadow: 0 12px 24px rgba(196, 69, 54, 0.3);
    }
    .stage {
      background: var(--card);
      border-radius: 26px;
      padding: 18px;
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .video-shell {
      position: relative;
      border-radius: 22px;
      overflow: hidden;
      background: #0f1418;
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.06);
    }
    .video-shell img {
      width: 100%;
      display: block;
      height: auto;
      filter: saturate(1.05);
    }
    .video-glow {
      position: absolute;
      inset: 0;
      background: linear-gradient(120deg, rgba(225, 108, 47, 0.18), rgba(58, 167, 184, 0.15));
      mix-blend-mode: screen;
      pointer-events: none;
    }
    .scanlines {
      position: absolute;
      inset: 0;
      background: repeating-linear-gradient(
        to bottom,
        rgba(255, 255, 255, 0.05) 0,
        rgba(255, 255, 255, 0.05) 1px,
        transparent 1px,
        transparent 5px
      );
      opacity: 0.15;
      pointer-events: none;
    }
    .stage-caption {
      color: var(--muted);
      font-size: 0.95rem;
    }
    .panel-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 16px;
    }
    .panel {
      background: var(--card);
      border-radius: 20px;
      padding: 18px;
      border: 1px solid var(--line);
      box-shadow: 0 18px 40px rgba(0, 0, 0, 0.1);
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .panel h3 {
      margin: 0;
      font-size: 0.85rem;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--accent-cool);
    }
    .camera-toolbar {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      align-items: center;
    }
    .camera-select {
      flex: 1 1 240px;
      min-width: 220px;
      padding: 12px 14px;
      border-radius: 14px;
      border: 1px solid rgba(0, 0, 0, 0.1);
      background: rgba(255, 255, 255, 0.9);
      color: var(--ink);
      font: inherit;
    }
    .camera-button {
      border: 0;
      border-radius: 14px;
      padding: 12px 18px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
      box-shadow: 0 12px 24px rgba(225, 108, 47, 0.25);
    }
    .camera-button:disabled {
      opacity: 0.6;
      cursor: wait;
    }
    .camera-hint {
      color: var(--muted);
      font-size: 0.92rem;
      min-height: 1.4em;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
      gap: 12px;
    }
    .metric {
      padding: 10px 12px;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.65);
      border: 1px solid rgba(0, 0, 0, 0.05);
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-height: 64px;
    }
    .metric .label {
      font-size: 1.5rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--muted);
    }
    .metric .value {
      font-size: 1.7rem;
      font-weight: 600;
    }
    .reveal {
      opacity: 0;
      transform: translateY(14px);
      animation: reveal 0.8s ease forwards;
      animation-delay: var(--delay, 0s);
    }
    @keyframes reveal {
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }
    @media (max-width: 780px) {
      .page {
        padding: 20px;
      }
      .brand {
        flex-direction: column;
        align-items: flex-start;
      }
      .logo {
        width: auto;
        height: auto;
        max-width: 30vw;
      }
    }
    @media (prefers-reduced-motion: reduce) {
      .reveal {
        animation: none;
        opacity: 1;
        transform: none;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <header class="header reveal" style="--delay: 0.05s">
      <div class="brand">
        <img class="logo" src="/logo.png" alt="Academy logo" />
        <div>
          <h1>基于骨骼提取的水果忍者游戏</h1>
          <p>未来人机交互新范式</p>
        </div>
      </div>
      <div class="status-pill" id="status">Starting</div>
    </header>

    <section class="stage reveal" style="--delay: 0.15s">
      <div class="video-shell">
        <img id="stream" src="/stream" alt="Live camera stream" />
        <div class="video-glow"></div>
        <div class="scanlines"></div>
      </div>
      <div class="stage-caption">Pose keypoints animate with each frame.</div>
    </section>

    <section class="panel-grid">
      <div class="panel reveal" style="--delay: 0.2s">
        <h3>Camera Source</h3>
        <div class="camera-toolbar">
          <select id="camera-select" class="camera-select">
            <option value="">正在扫描摄像头...</option>
          </select>
          <button id="camera-apply" class="camera-button" type="button">切换到该摄像头</button>
        </div>
        <div class="camera-hint" id="camera-hint">优先选择你的外接摄像头，例如 Logi C270 HD WebCam。</div>
        <div class="metrics">
          <div class="metric"><span class="label">当前设备</span><span class="value" id="active-camera">--</span></div>
          <div class="metric"><span class="label">输入源</span><span class="value" id="stream-source">--</span></div>
        </div>
      </div>

      <div class="panel reveal" style="--delay: 0.25s">
        <h3>Scoreboard</h3>
        <div class="metrics">
          <div class="metric"><span class="label">得分</span><span class="value" id="score">--</span></div>
          <div class="metric"><span class="label">生命值</span><span class="value" id="lives">--</span></div>
          <div class="metric"><span class="label">水果数</span><span class="value" id="fruits">--</span></div>
          <div class="metric"><span class="label">腕部速度</span><span class="value" id="wrist-speed">--</span></div>
          <div class="metric"><span class="label">帧率</span><span class="value" id="fps">--</span></div>
          <div class="metric"><span class="label">姿势</span><span class="value" id="pose-visible">--</span></div>
          <div class="metric"><span class="label">最后事件</span><span class="value" id="last-event">--</span></div>
        </div>
      </div>

      <div class="panel reveal" style="--delay: 0.3s">
        <h3>Output Parameters</h3>
        <div class="metrics">
          <div class="metric"><span class="label">帧大小</span><span class="value" id="frame-size">--</span></div>
          <div class="metric"><span class="label">生成间隔</span><span class="value" id="spawn-interval">--</span></div>
          <div class="metric"><span class="label">重力</span><span class="value" id="gravity">--</span></div>
          <div class="metric"><span class="label">刀片阈值</span><span class="value" id="blade-threshold">--</span></div>
          <div class="metric"><span class="label">人物比例</span><span class="value" id="min-person-ratio">--</span></div>
          <div class="metric"><span class="label">检测置信度</span><span class="value" id="detection-conf">--</span></div>
          <div class="metric"><span class="label">跟踪置信度</span><span class="value" id="tracking-conf">--</span></div>
        </div>
      </div>
    </section>
  </div>

  <script>
    const fields = {
      score: document.getElementById("score"),
      lives: document.getElementById("lives"),
      fruits: document.getElementById("fruits"),
      wristSpeed: document.getElementById("wrist-speed"),
      fps: document.getElementById("fps"),
      poseVisible: document.getElementById("pose-visible"),
      status: document.getElementById("status"),
      frameSize: document.getElementById("frame-size"),
      spawnInterval: document.getElementById("spawn-interval"),
      gravity: document.getElementById("gravity"),
      bladeThreshold: document.getElementById("blade-threshold"),
      minPersonRatio: document.getElementById("min-person-ratio"),
      detectionConf: document.getElementById("detection-conf"),
      trackingConf: document.getElementById("tracking-conf"),
      lastEvent: document.getElementById("last-event"),
      activeCamera: document.getElementById("active-camera"),
      streamSource: document.getElementById("stream-source")
    };
    const streamEl = document.getElementById("stream");
    const cameraSelect = document.getElementById("camera-select");
    const cameraApply = document.getElementById("camera-apply");
    const cameraHint = document.getElementById("camera-hint");

    const formatNumber = (value, digits = 1) => {
      if (value === null || value === undefined || Number.isNaN(value)) {
        return "--";
      }
      const num = Number(value);
      if (Number.isNaN(num)) {
        return "--";
      }
      return num.toFixed(digits);
    };

    const setText = (el, value) => {
      if (!el) {
        return;
      }
      el.textContent = value;
    };

    const resetStream = () => {
      if (!streamEl) {
        return;
      }
      streamEl.src = `/stream?t=${Date.now()}`;
    };

    const setCameraHint = (message) => {
      if (cameraHint) {
        cameraHint.textContent = message;
      }
    };

    const renderCameraOptions = (payload) => {
      if (!cameraSelect) {
        return;
      }
      const cameras = Array.isArray(payload?.cameras) ? payload.cameras : [];
      const activeIndex = payload?.active_camera_index;
      cameraSelect.innerHTML = "";
      if (!cameras.length) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "未找到可用摄像头";
        cameraSelect.appendChild(option);
        cameraSelect.disabled = true;
        if (cameraApply) {
          cameraApply.disabled = true;
        }
        return;
      }
      cameraSelect.disabled = false;
      if (cameraApply) {
        cameraApply.disabled = false;
      }
      cameras.forEach((camera) => {
        const option = document.createElement("option");
        option.value = String(camera.index);
        option.textContent = `${camera.name} · 索引 ${camera.index} · ${camera.backend} · ${camera.frame_size}`;
        if (camera.index === activeIndex) {
          option.selected = true;
        }
        cameraSelect.appendChild(option);
      });
    };

    const loadCameras = async () => {
      try {
        const response = await fetch("/cameras", { cache: "no-store" });
        if (!response.ok) {
          throw new Error("camera fetch failed");
        }
        const payload = await response.json();
        renderCameraOptions(payload);
      } catch (error) {
        setCameraHint("摄像头列表读取失败，请确认服务仍在运行。");
      }
    };

    const applyCameraSelection = async () => {
      if (!cameraSelect || !cameraApply) {
        return;
      }
      const selected = cameraSelect.value;
      cameraApply.disabled = true;
      setCameraHint("正在切换摄像头，请稍候...");
      try {
        const response = await fetch("/camera/select", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ index: selected })
        });
        const payload = await response.json();
        if (!response.ok || !payload.ok) {
          throw new Error(payload.message || "camera switch failed");
        }
        setCameraHint(payload.message || "摄像头切换成功。");
        renderCameraOptions(payload);
        resetStream();
        await refreshStats();
      } catch (error) {
        setCameraHint(`切换失败：${error.message}`);
      } finally {
        cameraApply.disabled = false;
      }
    };

    const updateStatus = (status) => {
      if (!fields.status) {
        return;
      }
      const isRunning = status === "running";
      fields.status.textContent = isRunning ? "Live" : "Camera Error";
      fields.status.classList.toggle("status-bad", !isRunning);
    };

    const refreshStats = async () => {
      try {
        const response = await fetch("/stats", { cache: "no-store" });
        if (!response.ok) {
          throw new Error("stats fetch failed");
        }
        const data = await response.json();
        setText(fields.score, data.score ?? "--");
        setText(fields.lives, data.lives ?? "--");
        setText(fields.fruits, data.fruits ?? "--");
        setText(fields.wristSpeed, `${formatNumber(data.wrist_speed, 0)} px/s`);
        setText(fields.fps, formatNumber(data.fps, 1));
        setText(fields.poseVisible, data.pose_visible ? "检出" : "未检出");
        setText(fields.lastEvent, data.last_event ?? "--");
        setText(fields.frameSize, data.frame_size ?? "--");
        setText(fields.spawnInterval, formatNumber(data.spawn_interval, 2));
        setText(fields.gravity, formatNumber(data.gravity, 0));
        setText(fields.bladeThreshold, formatNumber(data.blade_threshold, 0));
        setText(fields.minPersonRatio, formatNumber(data.min_person_ratio, 3));
        setText(fields.detectionConf, formatNumber(data.detection_conf, 2));
        setText(fields.trackingConf, formatNumber(data.tracking_conf, 2));
        const cameraLabel = data.active_camera_backend
          ? `${data.active_camera_name ?? "--"} · ${data.active_camera_backend}`
          : (data.active_camera_name ?? "--");
        setText(fields.activeCamera, cameraLabel);
        setText(fields.streamSource, data.stream_source ?? "--");
        updateStatus(data.status ?? "running");
      } catch (error) {
        updateStatus("camera_error");
      }
    };

    if (cameraApply) {
      cameraApply.addEventListener("click", applyCameraSelection);
    }

    loadCameras();
    refreshStats();
    setInterval(refreshStats, 250);
    setInterval(loadCameras, 5000);
  </script>
</body>
</html>
"""


@dataclass
class StreamState:
    frame: Optional[bytes] = None
    stats: Dict[str, object] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)
    started: bool = False
    active_camera_index: Optional[int] = None
    active_camera_name: str = "Auto"
    active_camera_backend: str = "Auto"
    stream_source: str = CAMERA_STREAM_URL if CAMERA_STREAM_URL else "local"
    available_cameras: List[Dict[str, object]] = field(default_factory=list)


STATE = StreamState()
STOP_EVENT = threading.Event()
CAMERA_SWITCH_EVENT = threading.Event()


def debug(msg: str) -> None:
    print(f"[DEBUG {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def list_windows_camera_names() -> List[str]:
    command = (
        "Get-PnpDevice -Class Camera | "
        "Where-Object { $_.Status -eq 'OK' } | "
        "Select-Object -ExpandProperty FriendlyName"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=8,
            check=False,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return []
    names = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return names


def detect_available_cameras(scan_limit: int = CAMERA_SCAN_LIMIT) -> List[Dict[str, object]]:
    names = list_windows_camera_names()
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    cameras: List[Dict[str, object]] = []
    for idx in range(scan_limit):
        detected_backend = None
        frame_size = None
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            ret, frame = cap.read()
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            if not ret or frame is None:
                continue
            if width <= 0 or height <= 0:
                height, width = frame.shape[:2]
            detected_backend = backend_label(backend)
            frame_size = f"{width}x{height}"
            break
        if detected_backend is None:
            continue
        cameras.append(
            {
                "index": idx,
                "name": names[idx] if idx < len(names) else f"Camera {idx}",
                "backend": detected_backend,
                "frame_size": frame_size or "unknown",
            }
        )
    return cameras


def refresh_available_cameras() -> List[Dict[str, object]]:
    cameras = detect_available_cameras()
    with STATE.lock:
        STATE.available_cameras = cameras
    return cameras


def backend_code_from_label(label: Optional[str]) -> Optional[int]:
    name_map = {
        "CAP_ANY": cv2.CAP_ANY,
        "CAP_DSHOW": cv2.CAP_DSHOW,
        "CAP_MSMF": cv2.CAP_MSMF,
    }
    return name_map.get(label or "")


def set_active_camera_state(index: Optional[int], name: str, source: str, backend: str = "Auto") -> None:
    with STATE.lock:
        STATE.active_camera_index = index
        STATE.active_camera_name = name
        STATE.active_camera_backend = backend
        STATE.stream_source = source


def get_camera_name(index: Optional[int]) -> str:
    if index is None:
        return "Auto"
    with STATE.lock:
        cameras = list(STATE.available_cameras)
    for camera in cameras:
        if camera.get("index") == index:
            return str(camera.get("name"))
    return f"Camera {index}"


def request_camera_switch(index: Optional[int]) -> Dict[str, object]:
    with STATE.lock:
        cameras = list(STATE.available_cameras)
    if not cameras:
        cameras = refresh_available_cameras()
    available_indices = {camera["index"] for camera in cameras}
    if index is not None and index not in available_indices:
        return {
            "ok": False,
            "message": f"Camera index {index} is not available.",
            "available_cameras": cameras,
        }
    camera_name = "Auto"
    camera_backend = "Auto"
    if index is not None:
        for camera in cameras:
            if camera.get("index") == index:
                camera_name = str(camera.get("name"))
                camera_backend = str(camera.get("backend", "Auto"))
                break
        else:
            camera_name = f"Camera {index}"
    set_active_camera_state(index, camera_name, "local", camera_backend)
    CAMERA_SWITCH_EVENT.set()
    return {
        "ok": True,
        "message": f"Switching to {camera_name}.",
        "active_camera_index": index,
        "available_cameras": cameras,
    }


def backend_label(backend: int) -> str:
    names = {
        cv2.CAP_ANY: "CAP_ANY",
        cv2.CAP_DSHOW: "CAP_DSHOW",
        cv2.CAP_MSMF: "CAP_MSMF",
    }
    return names.get(backend, str(backend))


def hex_to_bgr(value: str):
    value = value.lstrip("#")
    r, g, b = int(value[:2], 16), int(value[2:4], 16), int(value[4:], 16)
    return b, g, r


def read_image_rgba(path: Path):
    """Load PNG with Unicode-safe path handling."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def segment_circle_intersect(p1, p2, center, radius):
    (x1, y1), (x2, y2) = p1, p2
    cx, cy = center
    dx = x2 - x1
    dy = y2 - y1
    fx = x1 - cx
    fy = y1 - cy
    a = dx * dx + dy * dy
    if a == 0:
        return fx * fx + fy * fy <= radius * radius
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    disc = math.sqrt(disc)
    t1 = (-b - disc) / (2 * a)
    t2 = (-b + disc) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)


def overlay_rgba(frame, img, x, y):
    """Alpha-blend img (BGRA) onto frame (BGR) at top-left x,y."""
    h, w = img.shape[:2]
    if x >= frame.shape[1] or y >= frame.shape[0] or x + w <= 0 or y + h <= 0:
        return
    x1, y1 = max(0, x), max(0, y)
    x2 = min(x + w, frame.shape[1])
    y2 = min(y + h, frame.shape[0])
    if x1 >= x2 or y1 >= y2:
        return
    img_x1 = x1 - x
    img_y1 = y1 - y
    img_x2 = img_x1 + (x2 - x1)
    img_y2 = img_y1 + (y2 - y1)
    roi = frame[y1:y2, x1:x2]
    rgba = img[img_y1:img_y2, img_x1:img_x2]
    alpha = rgba[:, :, 3:4] / 255.0
    roi[:] = alpha * rgba[:, :, :3] + (1 - alpha) * roi


class Fruit:
    def __init__(self, kind, pos, velocity, image, juice_color=None):
        self.kind = kind
        self.x, self.y = pos
        self.vx, self.vy = velocity
        self.image = image
        self.juice_color = juice_color
        self.radius = max(image.shape[0], image.shape[1]) // 2
        self.alive = True

    def update(self, dt):
        if not self.alive:
            return "dead"
        self.vy += GRAVITY * dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.y - self.radius > HEIGHT + 120:
            self.alive = False
            return "miss"
        return None

    def draw(self, frame):
        if not self.alive:
            return
        x = int(self.x - self.image.shape[1] / 2)
        y = int(self.y - self.image.shape[0] / 2)
        overlay_rgba(frame, self.image, x, y)

    def slice(self):
        self.alive = False
        return "bomb" if self.kind == "boom" else "fruit"


class JuiceBurst:
    def __init__(self, pos, color_bgr):
        self.x, self.y = pos
        self.color = color_bgr
        self.life = 0.45
        self.radius = 30

    def update(self, dt):
        self.life -= dt
        self.radius += 320 * dt
        return self.life > 0

    def draw(self, frame):
        if self.life <= 0:
            return
        alpha = max(0.0, self.life / 0.45)
        overlay = frame.copy()
        cv2.circle(overlay, (int(self.x), int(self.y)), int(self.radius), self.color, -1)
        cv2.addWeighted(overlay, 0.25 * alpha, frame, 1 - 0.25 * alpha, 0, frame)


class BladeTrail:
    def __init__(self, point, time_stamp):
        self.point = point
        self.time = time_stamp


class Game:
    def __init__(self):
        self.fruit_images, self.juice_colors = self.load_images()
        self.pose = mp.solutions.pose.Pose(
            enable_segmentation=POSE_ENABLE_SEGMENTATION,
            model_complexity=POSE_MODEL_COMPLEXITY,
            min_detection_confidence=POSE_DETECTION_CONF,
            min_tracking_confidence=POSE_TRACKING_CONF,
        )
        self.reset(reason="ready")

    def load_images(self):
        fruit_images = {}
        juice_colors = {}
        for kind, (file, juice) in FRUIT_FILES.items():
            path = ASSETS_DIR / file
            img = read_image_rgba(path)
            if img is None:
                raise FileNotFoundError(f"Missing asset {path}")
            fruit_images[kind] = img
            if juice:
                juice_colors[kind] = hex_to_bgr(juice)
        return fruit_images, juice_colors

    def reset(self, reason: str = "reset"):
        self.score = 0
        self.lives = 3
        self.fruits = []
        self.effects = []
        self.last_spawn = 0.0
        self.last_time = time.time()
        self.trails = {"left": deque(maxlen=15), "right": deque(maxlen=15)}
        self.last_speed = 0.0
        self.last_event = reason
        self.last_event_time = time.time()

    def spawn_wave(self):
        count = random.randint(1, 3)
        for _ in range(count):
            start_x = random.uniform(80, WIDTH - 80)
            end_x = random.uniform(80, WIDTH - 80)
            start_y = HEIGHT + 60
            flight_time = 1.0
            vx = (end_x - start_x) / flight_time
            vy = random.uniform(-1300, -1000)
            kind = self.pick_kind()
            fruit = Fruit(kind, (start_x, start_y), (vx, vy), self.fruit_images[kind], self.juice_colors.get(kind))
            self.fruits.append(fruit)

    def pick_kind(self):
        if random.randint(0, 8) == 4:
            return "boom"
        choices = [k for k in FRUIT_FILES.keys() if k != "boom"]
        return random.choice(choices)

    def update(self, frame, wrist_positions, now):
        dt = now - self.last_time
        self.last_time = now

        if now - self.last_spawn > SPAWN_INTERVAL:
            self.spawn_wave()
            self.last_spawn = now

        for fruit in list(self.fruits):
            state = fruit.update(dt)
            if state == "miss" and fruit.kind != "boom":
                self.lives -= 1
                self.last_event = "miss"
                self.last_event_time = now
                if self.lives <= 0:
                    self.reset(reason="game_over")
                    return
            if not fruit.alive and fruit in self.fruits:
                self.fruits.remove(fruit)

        self.effects = [fx for fx in self.effects if fx.update(dt)]

        self.update_trails(wrist_positions, now)

        max_speed = 0.0
        for trail in self.trails.values():
            for trail_idx in range(len(trail) - 1):
                p1 = trail[trail_idx].point
                p2 = trail[trail_idx + 1].point
                speed = self.segment_speed(trail[trail_idx], trail[trail_idx + 1])
                if speed > max_speed:
                    max_speed = speed
                if speed < BLADE_SPEED_THRESHOLD:
                    continue
                for fruit in list(self.fruits):
                    if not fruit.alive:
                        continue
                    if segment_circle_intersect(p1, p2, (fruit.x, fruit.y), fruit.radius):
                        result = fruit.slice()
                        self.fruits.remove(fruit)
                        if result == "bomb":
                            self.reset(reason="bomb")
                            return
                        self.score += 1
                        self.last_event = "slice"
                        self.last_event_time = now
                        if fruit.juice_color:
                            self.effects.append(JuiceBurst((fruit.x, fruit.y), fruit.juice_color))

        self.last_speed = max_speed
        self.draw(frame)

    def update_trails(self, wrist_positions, now):
        max_age = 0.35
        for hand in ["left", "right"]:
            trail = self.trails[hand]
            pos = wrist_positions.get(hand)
            if pos is None:
                trail.clear()
                continue
            point = (int(pos[0]), int(pos[1]))
            trail.append(BladeTrail(point, now))
            while trail and now - trail[0].time > max_age:
                trail.popleft()

    @staticmethod
    def segment_speed(a: BladeTrail, b: BladeTrail):
        dist = math.hypot(b.point[0] - a.point[0], b.point[1] - a.point[1])
        dt = max(b.time - a.time, 1e-3)
        return dist / dt

    def draw(self, frame):
        for fruit in self.fruits:
            fruit.draw(frame)

        for trail in self.trails.values():
            for i in range(len(trail) - 1):
                p1 = trail[i].point
                p2 = trail[i + 1].point
                age = time.time() - trail[i].time
                alpha = max(0.0, 1 - age / 0.35)
                thickness = int(12 * alpha) + 2
                color = (255, int(255 * alpha), 80)
                cv2.line(frame, p1, p2, color, thickness, lineType=cv2.LINE_AA)

        for fx in self.effects:
            fx.draw(frame)


def draw_pose_overlay(frame, results, now):
    if not results.pose_landmarks:
        return
    h, w = frame.shape[:2]
    landmarks = results.pose_landmarks.landmark
    pulse = 0.5 + 0.5 * math.sin(now * 6.0)
    point_radius = int(3 + 2 * pulse)
    line_color = (64, 220, 255)
    point_color = (255, 196, 64)
    glow_color = (255, 255, 255)
    wrist_ids = {
        mp.solutions.pose.PoseLandmark.LEFT_WRIST.value,
        mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value,
    }

    for connection in mp.solutions.pose.POSE_CONNECTIONS:
        a, b = connection
        if landmarks[a].visibility < 0.55 or landmarks[b].visibility < 0.55:
            continue
        x1, y1 = int(landmarks[a].x * w), int(landmarks[a].y * h)
        x2, y2 = int(landmarks[b].x * w), int(landmarks[b].y * h)
        cv2.line(frame, (x1, y1), (x2, y2), line_color, 2, cv2.LINE_AA)

    for idx, lm in enumerate(landmarks):
        if lm.visibility < 0.55:
            continue
        x, y = int(lm.x * w), int(lm.y * h)
        if idx in wrist_ids:
            radius = point_radius + 3
            color = (0, 255, 255)
        else:
            radius = point_radius
            color = point_color
        cv2.circle(frame, (x, y), radius + 2, glow_color, 1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)


def get_wrist_positions(results, frame_shape):
    """Return both wrists (left/right) if visible enough; ignore small/secondary persons."""
    if not results.pose_landmarks:
        return {}
    h, w = frame_shape[:2]

    if results.segmentation_mask is not None:
        mask = results.segmentation_mask
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h))
        _, binary = cv2.threshold(mask, 0.3, 1, cv2.THRESH_BINARY)
        binary = binary.astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest = max((cv2.contourArea(c) for c in contours), default=0)
        if largest < MIN_PERSON_RATIO * h * w:
            return {}

    def landmark_ok(point, elbow):
        return point.visibility >= 0.6 and elbow.visibility >= 0.5

    l_wrist = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    r_wrist = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    l_elbow = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
    r_elbow = results.pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]

    wrists = {}
    if landmark_ok(l_wrist, l_elbow):
        wrists["left"] = (int(l_wrist.x * w), int(l_wrist.y * h))
    if landmark_ok(r_wrist, r_elbow):
        wrists["right"] = (int(r_wrist.x * w), int(r_wrist.y * h))
    return wrists


def open_camera(preferred_index: Optional[int] = None, preferred_backend: Optional[int] = None) -> Optional[cv2.VideoCapture]:
    """Try multiple backends/indices and emit debug info."""
    stream_url = CAMERA_STREAM_URL
    if stream_url and stream_url.lower() not in {"0", "false", "none", "local"}:
        cap = cv2.VideoCapture(stream_url)
        if cap.isOpened():
            ret, test = cap.read()
            if ret and test is not None:
                h, w = test.shape[:2]
                debug(f"Using camera stream {stream_url}; first frame {w}x{h}")
                return cap
            cap.release()
            debug(f"Camera stream {stream_url} opened but no frame.")
        else:
            debug(f"Could not open camera stream {stream_url}.")
        return None

    attempts = []
    env_backends = os.environ.get("FN_CAMERA_BACKENDS")
    if env_backends:
        name_map = {"ANY": cv2.CAP_ANY, "DSHOW": cv2.CAP_DSHOW, "MSMF": cv2.CAP_MSMF}
        backends = [name_map[b.strip().upper()] for b in env_backends.split(",") if b.strip().upper() in name_map]
    else:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    if preferred_backend is not None:
        backends = [preferred_backend] + [backend for backend in backends if backend != preferred_backend]
        debug(f"Trying preferred backend {backend_label(preferred_backend)} first.")
    indices = list(range(CAMERA_SCAN_LIMIT))
    if preferred_index is not None:
        indices = [preferred_index] + [i for i in indices if i != preferred_index]
        debug(f"Trying preferred camera index {preferred_index} first.")
    for backend in backends:
        for idx in indices:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                attempts.append(f"index {idx} backend {backend_label(backend)} (not opened)")
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
            ret, test = cap.read()
            if not ret or test is None:
                cap.release()
                attempts.append(f"index {idx} backend {backend_label(backend)} (no frame)")
                continue
            h, w = test.shape[:2]
            debug(f"Using camera index {idx} backend {backend_label(backend)}; first frame {w}x{h}")
            return cap
    debug(f"Could not open any camera. Tried: {attempts}")
    return None


def encode_frame(frame) -> Optional[bytes]:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        return None
    return buffer.tobytes()


def build_error_frame(message: str) -> np.ndarray:
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    cv2.putText(frame, "Camera error", (20, HEIGHT // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, message, (20, HEIGHT // 2 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


def build_stats(game: Game, fps: float, frame_shape, pose_visible: bool, now: float, status: str):
    h, w = frame_shape[:2]
    event = game.last_event
    if now - game.last_event_time > 2.5:
        event = "steady"
    with STATE.lock:
        active_camera_index = STATE.active_camera_index
        active_camera_name = STATE.active_camera_name
        active_camera_backend = STATE.active_camera_backend
        stream_source = STATE.stream_source
    return {
        "status": status,
        "score": game.score,
        "lives": game.lives,
        "fruits": len(game.fruits),
        "fps": round(fps, 1),
        "wrist_speed": round(game.last_speed, 0),
        "pose_visible": bool(pose_visible),
        "frame_size": f"{w}x{h}",
        "spawn_interval": SPAWN_INTERVAL,
        "gravity": GRAVITY,
        "blade_threshold": BLADE_SPEED_THRESHOLD,
        "min_person_ratio": MIN_PERSON_RATIO,
        "detection_conf": POSE_DETECTION_CONF,
        "tracking_conf": POSE_TRACKING_CONF,
        "last_event": event,
        "active_camera_index": active_camera_index,
        "active_camera_name": active_camera_name,
        "active_camera_backend": active_camera_backend,
        "stream_source": stream_source,
    }


def capture_loop() -> None:
    game = Game()
    preferred_idx = os.environ.get("FN_CAMERA_INDEX")
    preferred_idx_val = None
    preferred_backend_val = None
    preferred_backend_name = "Auto"
    if preferred_idx:
        try:
            preferred_idx_val = int(preferred_idx)
        except ValueError:
            debug(f"Ignoring FN_CAMERA_INDEX='{preferred_idx}' (not an int).")
    cameras = refresh_available_cameras()
    if preferred_idx_val is None and CAMERA_STREAM_URL.lower() in {"0", "false", "none", "local"} and cameras:
        preferred_idx_val = int(cameras[0]["index"])
    selected_camera = next((camera for camera in cameras if camera.get("index") == preferred_idx_val), None)
    if selected_camera is not None:
        preferred_backend_name = str(selected_camera.get("backend", "Auto"))
        preferred_backend_val = backend_code_from_label(preferred_backend_name)
    set_active_camera_state(
        preferred_idx_val,
        get_camera_name(preferred_idx_val),
        "local" if CAMERA_STREAM_URL.lower() in {"0", "false", "none", "local"} else CAMERA_STREAM_URL,
        preferred_backend_name,
    )
    cap = open_camera(preferred_idx_val, preferred_backend_val)
    if cap is None:
        error_frame = build_error_frame("No camera available.")
        encoded = encode_frame(error_frame)
        if encoded:
            with STATE.lock:
                STATE.frame = encoded
                STATE.stats = {
                    "status": "camera_error",
                    "score": 0,
                    "lives": 0,
                    "fruits": 0,
                    "fps": 0,
                    "wrist_speed": 0,
                    "pose_visible": False,
                    "frame_size": f"{WIDTH}x{HEIGHT}",
                    "spawn_interval": SPAWN_INTERVAL,
                    "gravity": GRAVITY,
                    "blade_threshold": BLADE_SPEED_THRESHOLD,
                    "min_person_ratio": MIN_PERSON_RATIO,
                    "detection_conf": POSE_DETECTION_CONF,
                    "tracking_conf": POSE_TRACKING_CONF,
                    "last_event": "camera_error",
                    "active_camera_index": preferred_idx_val,
                    "active_camera_name": get_camera_name(preferred_idx_val),
                    "active_camera_backend": preferred_backend_name,
                    "stream_source": STATE.stream_source,
                }
        return

    last_frame_time = time.time()
    fps = 0.0

    try:
        while not STOP_EVENT.is_set():
            if CAMERA_SWITCH_EVENT.is_set():
                CAMERA_SWITCH_EVENT.clear()
                with STATE.lock:
                    next_camera_index = STATE.active_camera_index
                    next_camera_backend_name = STATE.active_camera_backend
                    STATE.frame = None
                debug(f"Switching camera to index {next_camera_index}.")
                cap.release()
                cap = open_camera(next_camera_index, backend_code_from_label(next_camera_backend_name))
                if cap is None:
                    error_frame = build_error_frame("Selected camera unavailable.")
                    encoded = encode_frame(error_frame)
                    if encoded:
                        with STATE.lock:
                            STATE.frame = encoded
                            STATE.stats = {
                                "status": "camera_error",
                                "score": game.score,
                                "lives": game.lives,
                                "fruits": len(game.fruits),
                                "fps": 0,
                                "wrist_speed": round(game.last_speed, 0),
                                "pose_visible": False,
                                "frame_size": f"{WIDTH}x{HEIGHT}",
                                "spawn_interval": SPAWN_INTERVAL,
                                "gravity": GRAVITY,
                                "blade_threshold": BLADE_SPEED_THRESHOLD,
                                "min_person_ratio": MIN_PERSON_RATIO,
                                "detection_conf": POSE_DETECTION_CONF,
                                "tracking_conf": POSE_TRACKING_CONF,
                                "last_event": "camera_error",
                                "active_camera_index": next_camera_index,
                                "active_camera_name": get_camera_name(next_camera_index),
                                "active_camera_backend": next_camera_backend_name,
                                "stream_source": "local",
                            }
                    time.sleep(0.2)
                    continue
                last_frame_time = time.time()
                fps = 0.0
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            frame = cv2.flip(frame, 1)
            now = time.time()
            dt = now - last_frame_time
            if dt > 0:
                instant_fps = 1.0 / dt
                fps = instant_fps if fps == 0 else fps * 0.9 + instant_fps * 0.1
            last_frame_time = now

            mp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_frame.flags.writeable = False
            results = game.pose.process(mp_frame)
            wrist_positions = get_wrist_positions(results, frame.shape)
            game.update(frame, wrist_positions, now)
            draw_pose_overlay(frame, results, now)

            output_frame = frame
            if DISPLAY_SCALE != 1.0:
                output_frame = cv2.resize(
                    frame,
                    (int(frame.shape[1] * DISPLAY_SCALE), int(frame.shape[0] * DISPLAY_SCALE)),
                    interpolation=cv2.INTER_LINEAR,
                )

            stats = build_stats(game, fps, output_frame.shape, results.pose_landmarks is not None, now, "running")
            encoded = encode_frame(output_frame)
            if not encoded:
                continue
            with STATE.lock:
                STATE.frame = encoded
                STATE.stats = stats
    finally:
        cap.release()


def start_capture_thread() -> None:
    if STATE.started:
        return
    STATE.started = True
    thread = threading.Thread(target=capture_loop, daemon=True)
    thread.start()


def stream_frames():
    delay = 1.0 / max(1.0, STREAM_FPS)
    while True:
        with STATE.lock:
            frame = STATE.frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(frame)).encode("ascii") + b"\r\n\r\n" + frame + b"\r\n"
        )
        time.sleep(delay)


app = Flask(__name__)


@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


@app.get("/stream")
def stream():
    return Response(stream_frames(), content_type="multipart/x-mixed-replace;boundary=frame")


@app.get("/stats")
def stats():
    with STATE.lock:
        payload = dict(STATE.stats) if STATE.stats else {"status": "starting"}
    return jsonify(payload)


@app.get("/cameras")
def cameras():
    with STATE.lock:
        payload = list(STATE.available_cameras)
        active_camera_index = STATE.active_camera_index
    if not payload:
        payload = refresh_available_cameras()
    return jsonify({"cameras": payload, "active_camera_index": active_camera_index})


@app.post("/camera/select")
def select_camera():
    payload = request.get_json(silent=True) or {}
    raw_index = payload.get("index")
    if raw_index in ("", "auto", None):
        index = None
    else:
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "message": "Camera index must be an integer."}), 400
    result = request_camera_switch(index)
    status_code = 200 if result.get("ok") else 400
    return jsonify(result), status_code


@app.get("/logo.png")
def logo():
    if not LOGO_PATH.exists():
        return Response("Logo not found", status=404)
    return send_file(LOGO_PATH)


@app.get("/favicon.ico")
def favicon():
    if not LOGO_PATH.exists():
        return Response(status=404)
    return send_file(LOGO_PATH, mimetype="image/png")


def main() -> None:
    start_capture_thread()
    debug(f"Starting web server on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
