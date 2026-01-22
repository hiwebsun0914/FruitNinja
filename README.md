# Fruit Ninja 体感切水果

基于摄像头手势识别的体感切水果小游戏，适合作为计算机视觉与交互设计课程的演示项目。

## 创作团队

中山大学智能工程学院学生会 ISEGeekUp 制作团队。

## 功能说明

- **体感切水果**：使用摄像头检测手部动作，实现挥动切割水果的交互体验。
- **计分与结算**：切中水果会计分，游戏结束时展示结算界面。
- **界面与特效**：包含主界面、开始界面、分数展示与结算界面。

## 运行方式

> 运行前请确保已连接摄像头并安装 Python 3。建议在 Windows 上运行。

1. 安装依赖（如未安装 OpenCV）：

   ```bash
   pip install opencv-python
   ```

2. 启动游戏：

   ```bash
   python cv_fruit_ninja.py
   ```

## 运行参数

如需调整摄像头与窗口配置，可在 `cv_fruit_ninja.py` 中修改以下常见参数（不同环境可按需调整）：

- `camera_id`：摄像头编号（默认 0）。
- `frame_width` / `frame_height`：摄像头采集分辨率。
- `window_name`：窗口标题。

## 运行截图

以下为游戏运行时的部分界面截图：

![主界面](release/images/home-desc.png)
![开始游戏](release/images/new-game.png)
![得分界面](release/images/score.png)
![游戏结束](release/images/game-over.png)
