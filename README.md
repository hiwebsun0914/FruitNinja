# Fruit Ninja 体感版

基于摄像头姿态识别的网页水果忍者演示项目。程序会启动一个本地 Web 服务，在浏览器中显示实时视频、骨骼识别结果和游戏画面。

## 当前运行方式

项目当前建议使用 `conda` 环境 `pytorch_gpu_23.5.30` 运行：

```powershell
conda run -n pytorch_gpu_23.5.30 python cv_fruit_ninja.py
```

启动后打开：

```text
http://127.0.0.1:8888/
```

## 前端摄像头切换

页面中新增了 `Camera Source` 面板，可以直接在前端选择摄像头并切换，无需手动改代码。

- `Logi C270 HD WebCam`：当前识别到的外接摄像头
- `HD Webcam`：通常是笔记本内置摄像头

如果你希望默认优先使用某个摄像头，也可以在启动前设置环境变量：

```powershell
$env:FN_CAMERA_INDEX='0'
conda run -n pytorch_gpu_23.5.30 python cv_fruit_ninja.py
```

## 常用环境变量

- `FN_PORT`：服务端口，默认 `8888`
- `FN_CAMERA_INDEX`：优先尝试的摄像头索引
- `FN_CAMERA_BACKENDS`：摄像头后端顺序，例如 `DSHOW,MSMF`
- `FN_WIDTH` / `FN_HEIGHT`：采集分辨率
- `FN_DISPLAY_SCALE`：前端显示缩放比例

## 依赖

项目依赖主要包括：

- `opencv-python`
- `mediapipe`
- `numpy`
- `flask`

如果你在新的环境中运行，可以按需安装：

```powershell
pip install opencv-python mediapipe numpy flask
```

## 说明

- 建议在 Windows 下运行
- 首次启动时会扫描可用摄像头
- 如果页面能打开但没有体感效果，先确认人物已经进入镜头且姿态被识别到
