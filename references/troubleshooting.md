# 常见问题与解决方案

## Deface

### 问题: MTS 文件无法处理
**现象**: deface 报错 `video format not recognized` 或 `text/plain`
**原因**: deface 的 imageio 后端不支持 MTS 容器格式
**解决**: 先用 ffmpeg 转封装为 MP4：
```bash
ffmpeg -i input.MTS -c copy -map 0:v -map 0:a output.mp4
```
然后对 MP4 进行 deface。脚本会自动处理此步骤。

### 问题: 输出视频损坏 (moov atom missing)
**现象**: ffprobe 报错 "moov atom not found"
**原因**: deface 进程被提前终止，未能完成 MP4 尾部写入
**解决**: 删除损坏文件，重新运行。确保不中断 deface 进程。

### 问题: face detection skipped
**现象**: deface 输出 `0 faces detected`，但视频明显有人脸
**原因**: 人脸角度过大、遮挡严重、或光照不足
**解决**: 无需调整，centerface 模型对正面/侧面人脸效果良好。极少数极端角度确实无法检测。

## CUDA / GPU

### 问题: CUDA EP falls back to CPU
**现象**: onnxruntime 日志显示 `CUDAExecutionProvider` 回退到 `CPUExecutionProvider`
**原因**: 
  1. `cublas64_13.dll` 或 `cublasLt64_13.dll` 缺失
  2. cuDNN DLL 版本不匹配
  3. CUDA Toolkit 版本与 onnxruntime-gpu 版本不兼容
**解决**:
  1. 确保 cuDNN 9.x DLL 在 onnxruntime/capi/ 或 CUDA Toolkit/bin/ 中
  2. 确保 cublas64_13.dll 在 onnxruntime/capi/ 中
  3. 运行 `audit_env.py` 逐项排查

### 问题: Error 126 (DLL not found)
**现象**: `[ONNXRuntimeError] : 1 : FAIL : LoadLibrary failed with error 126`
**原因**: onnxruntime 的 CUDA provider DLL 找不到依赖的 cublas/cudnn/cudart DLL
**解决**: 将所有 CUDA 13 Toolkit 和 cuDNN 9 DLL 复制到 onnxruntime/capi/ 目录

### 问题: 速度比预期慢
**现象**: 单卡只有 ~30 it/s 而不是 ~40 it/s
**原因**: 可能回退到了 CPU 或 DirectML
**检查**: `D:\face_anon_env\Scripts\python -c "import onnxruntime; print(onnxruntime.get_available_providers())"`
**确保**: 列表中包含 `CUDAExecutionProvider` 且排在第一个

## 双卡

### 问题: 双卡其中一个 GPU 空闲
**现象**: nvidia-smi 显示只有一张卡在工作
**原因**: deface 未使用 `CUDA_VISIBLE_DEVICES` 指定 GPU，两个进程冲突了同一张卡
**解决**: 使用 `deface_single.py` 或 `deface_batch.py` 脚本，它们自动管理 GPU 分配

### 问题: 分片后合并的视频有跳帧
**现象**: 两段视频拼接处有画面跳跃
**原因**: ffmpeg `-c copy` 只能在关键帧处切割
**解决**: 影响极小（<1秒），可接受。如需完美合并，使用 `-ss` 关键帧对齐或重编码拼接点。

## 编码

### 问题: h264_nvenc 不可用
**现象**: ffmpeg 报错 `unknown encoder 'h264_nvenc'`
**原因**: NVIDIA GPU 硬件编码器未安装或驱动过旧
**解决**: 更新 NVIDIA 驱动到最新。如需回退 CPU 编码，使用 `--cpu` 参数。
