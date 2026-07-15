---
name: deface-diy
description: 医疗视频面部脱敏管线。基于 deface + onnxruntime-gpu + CUDA 13.0 + cuDNN 9.24 + 双 RTX 4090 NVENC 硬件编码的最快方案。支持单视频处理、文件夹批量处理、多种视频格式自动转换。包含环境审查、单视频双卡处理、批量双卡处理三个脚本。触发词: deface、面部脱敏、人脸脱敏、视频脱敏、脸部模糊、face anonymization、医疗视频处理。
agent_created: true
---

# Deface DIY — 医疗视频面部脱敏管线

基于 `deface` + `onnxruntime-gpu 1.27.0` + CUDA 13.0 + cuDNN 9.24 的双卡 RTX 4090 全速方案。
自动 MTS→MP4、分片双 GPU 并行、NVENC 硬件编码，输出固定为 MP4。

## 工作流总览

每次被触发时，按以下层次执行：

### 第一层：环境审查（首次或用户要求时）

运行审计脚本检查 GPU、CUDA、cuDNN、onnxruntime、Python、FFmpeg：

```bash
D:\face_anon_env\Scripts\python scripts/audit_env.py
```

如果审计发现 CUDA EP 未激活，按照 `references/cuda13_install.md` 的步骤修复。

### 第二层：单视频处理

用户指定一个视频文件时：

```bash
D:\face_anon_env\Scripts\python scripts/deface_single.py <视频路径>
```

行为：
- 自动检测格式，非 MP4 (MTS/AVI/MOV 等) 先用 ffmpeg 转封装
- 视频 > 2 秒：分片为两半，双 GPU 并行 deface
- 视频 ≤ 2 秒：单 GPU 直接处理
- 输出：`<原文件名>_anonymized.mp4`（同目录）
- 完成后自动清理临时文件

### 第三层：批量文件夹处理

用户指定一个目录或多个目录时：

```bash
D:\face_anon_env\Scripts\python scripts/deface_batch.py <目录1> [目录2 ...]
```

默认扫描目录：
- `D:\workspace\face\phase1_video`
- `D:\145videodata`

行为：
- 递归扫描所有视频文件
- 逐个按双卡管道处理
- 实时写入 `status.json` 供看板读取
- 完成后输出统计摘要

### 第四层：进度看板（可选）

启动 HTTP 看板服务：

```bash
# 在 deface_batch.py 运行时，另开终端：
cd <skill-directory>/..
D:\face_anon_env\Scripts\python -m http.server 8765
# 然后打开 http://localhost:8765/dashboard.html
```

看板 HTML 文件在 `assets/dashboard.html`，从 status.json 读取数据。
每 2 秒自动刷新，显示：
- 总体进度条和统计
- 当前视频双卡实时进度
- 最近完成的文件列表
- 预计剩余时间

## 技术栈

| 组件 | 版本 | 用途 |
|------|------|------|
| deface | 1.5.0 | 人脸检测 + 面部模糊 |
| onnxruntime-gpu | 1.27.0 | ONNX 模型推理 (CUDA 13 EP) |
| CUDA Toolkit | 13.0.48 | GPU 加速平台 |
| cuDNN | 9.24.0.43 | 深度神经网络加速库 |
| NVIDIA 驱动 | 610.74 | GPU 驱动 |
| NVENC | H.264 | 硬件视频编码 |
| Python | 3.13 | 脚本运行环境 |

## 管道性能

| 方案 | 单卡 | 双卡 | 11分钟视频 |
|------|------|------|-----------|
| DirectML (CPU 编码) | ~30 it/s | ~56 it/s | ~12 min |
| DirectML + NVENC | ~35 it/s | ~65 it/s | ~10 min |
| **CUDA 13 Native + NVENC** | **~39 it/s** | **~72 it/s** | **~4.5 min** |

## 脚本参考

### scripts/audit_env.py
环境审查脚本。检查 GPU、CUDA Toolkit、cuDNN、onnxruntime、deface、Python、FFmpeg、磁盘、内存。
独立运行，不依赖其他文件。输出每项 ✅/⚠️/❌ 状态。

### scripts/deface_single.py
单视频双卡处理脚本。支持 `--no-dual` 强制单 GPU 模式。
自动 MTS→MP4 转封装，双卡分片并行 deface，合并输出，清理临时文件。

### scripts/deface_batch.py
批量双卡处理脚本。接受目录参数或使用默认目录。
递归扫描所有视频格式，逐个双卡处理，实时状态回写 status.json。

### references/cuda13_install.md
完整的 CUDA 13.0 + cuDNN 9.24 + onnxruntime-gpu 1.27.0 安装步骤。
包含驱动检查、CUDA Toolkit 下载安装、cuDNN 配置、Python 环境搭建、验证步骤和版本兼容矩阵。

### references/troubleshooting.md
常见问题：MTS 格式不支持、输出视频损坏、CUDA EP 回退 CPU、Error 126 DLL 缺失、
速度慢、双卡闲置、NVENC 不可用、分片合并跳帧。每个问题含现象、原因、解决方案。

## 依赖清单 (requirements.txt)

```
deface>=1.5.0
onnxruntime-gpu==1.27.0
onnx>=1.17
numpy>=1.24
opencv-python-headless>=4.8
imageio>=2.31
imageio-ffmpeg>=0.4
```

外部依赖：ffmpeg (系统 PATH)、CUDA 13 Toolkit、cuDNN 9.24。

## 安全检查

此管线执行以下系统级操作：
- 读取本地视频文件（输入目录扫描）
- 写入处理后视频（同目录 `_anonymized.mp4`）
- 执行 ffmpeg 命令行（转封装、分片、合并）
- 执行 deface CLI（人脸检测 + 模糊）
- 调用 onnxruntime GPU 推理
- 写入 status.json 文件（进度状态）
- 调用 subprocess.Popen 管理子进程

**不做的事情**：
- 不上传任何数据到网络（完全离线）
- 不修改输入文件
- 不删除用户原始文件（仅清理临时文件 `_p1.mp4`, `_p2.mp4`, `_a1.mp4`, `_a2.mp4`, `_merge.txt`）
- 不访问注册表、系统设置、网络
