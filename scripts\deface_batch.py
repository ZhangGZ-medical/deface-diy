#!/usr/bin/env python
"""
Dual RTX 4090 + CUDA 13 native + NVENC deface batch processor
=============================================================
Optimized pipeline:
  1. Scan target dir(s) for video files (mp4/mts/avi/mov/mkv/...)
  2. Auto-convert non-MP4 -> MP4 (stream copy, or re-encode fallback)
  3. Split each video into 2 halves
  4. Run deface on both halves in parallel (CUDA EP on GPU0 + GPU1)
  5. Merge anonymized halves back into single MP4
  6. Clean up temp files

Requires:
  - D:\\face_anon_env (Python venv with deface + onnxruntime-gpu)
  - CUDA 13.0 + cuDNN 9.x in PATH
  - NVENC-capable GPU (RTX 4090)
  - ffmpeg in PATH

Usage:
  python deface_batch.py [target_dir] [target_dir2 ...]
  python deface_batch.py D:\\145videodata
  python deface_batch.py D:\\workspace\\face\\phase1_video D:\\145videodata
"""

import os, sys, time, subprocess, glob as glob_mod
from pathlib import Path
from datetime import timedelta


# ============= Configuration =============
PYTHON = r"D:\face_anon_env\Scripts\python.exe"
DEFACE_CMD = r"D:\face_anon_env\Scripts\deface"
FFMPEG_CFG = '{"codec":"h264_nvenc"}'
BACKEND = "onnxrt"
EP = "CUDAExecutionProvider"
VIDEO_EXTENSIONS = {".mp4", ".mts", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v", ".ts"}


def setup_env():
    """Ensure CUDA 13 DLLs are findable"""
    cuda_base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
    for sub in ["bin", "bin\\x64"]:
        d = os.path.join(cuda_base, sub)
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except (AttributeError, OSError):
                pass
    path_add = os.pathsep.join([
        os.path.join(cuda_base, "bin"),
        os.path.join(cuda_base, "bin", "x64"),
    ])
    os.environ["PATH"] = path_add + os.pathsep + os.environ.get("PATH", "")


def scan_videos(root_dir):
    """Find all video files recursively"""
    videos = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS:
                videos.append(os.path.join(root, f))
    return sorted(videos)


def get_duration(path):
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=15
        )
        return float(r.stdout.strip() or 0)
    except:
        return 0


def to_mp4(src):
    """Convert non-MP4 to MP4 (stream copy preferred, re-encode fallback)"""
    if src.lower().endswith(".mp4"):
        return src
    dst = os.path.splitext(src)[0] + ".mp4"
    if os.path.exists(dst):
        return dst
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-c", "copy", "-map", "0:v", "-map", "0:a?", dst],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=180
    )
    if r.returncode != 0:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-c:v", "libx264", "-c:a", "aac",
             "-preset", "fast", dst],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=300
        )
    return dst


def split_video(mp4_path, out_dir):
    """Split video into 2 halves for dual GPU"""
    dur = get_duration(mp4_path)
    if dur <= 2:
        base = os.path.splitext(os.path.basename(mp4_path))[0]
        p1 = os.path.join(out_dir, f"{base}_p1.mp4")
        subprocess.run(["ffmpeg", "-y", "-i", mp4_path, "-c", "copy", p1],
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60)
        return p1, p1

    mid = dur / 2
    base = os.path.splitext(os.path.basename(mp4_path))[0]
    p1 = os.path.join(out_dir, f"{base}_p1.mp4")
    p2 = os.path.join(out_dir, f"{base}_p2.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp4_path, "-t", str(mid), "-c", "copy", p1],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp4_path, "-ss", str(mid), "-c", "copy", p2],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60
    )
    return p1, p2


def deface_part(input_path, output_path, gpu_id):
    """Run deface on one part with CUDA EP on specific GPU"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cuda_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64",
    ]
    env["PATH"] = os.pathsep.join(cuda_paths) + os.pathsep + env.get("PATH", "")
    cmd = [
        DEFACE_CMD, input_path,
        "-k",
        "--backend", BACKEND,
        "--ep", EP,
        "--ffmpeg-config", FFMPEG_CFG,
        "-o", output_path,
    ]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)


def process_video(video_path):
    """Process one video with dual GPU pipeline"""
    vname = os.path.basename(video_path)
    parent = os.path.dirname(video_path)
    base_name = os.path.splitext(vname)[0]
    output = os.path.join(parent, f"{base_name}_anonymized.mp4")
    start_time = time.time()

    # Convert to MP4
    mp4_path = to_mp4(video_path)

    # Split
    p1, p2 = split_video(mp4_path, parent)
    a1 = p1.replace("_p1.mp4", "_a1.mp4")
    a2 = p2.replace("_p2.mp4", "_a2.mp4")

    if p1 == p2:
        # Very short video, just single GPU
        env = os.environ.copy()
        cuda_paths = [r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin",
                      r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\x64"]
        env["PATH"] = os.pathsep.join(cuda_paths) + os.pathsep + env.get("PATH", "")
        r = subprocess.run(
            [DEFACE_CMD, p1, "-k", "--backend", BACKEND, "--ep", EP,
             "--ffmpeg-config", FFMPEG_CFG, "-o", output],
            env=env, capture_output=True, text=True, timeout=3600
        )
        ok = r.returncode == 0
        elapsed = time.time() - start_time
        out_size = os.path.getsize(output) / 1024 / 1024 if os.path.exists(output) else 0
        for f in [p1, mp4_path]:
            if f != video_path:
                try: os.remove(f)
                except: pass
        return ok, vname, elapsed, out_size
    else:
        # Dual GPU
        proc0 = deface_part(p1, a1, gpu_id=0)
        proc1 = deface_part(p2, a2, gpu_id=1)

        target_mb = os.path.getsize(p1) + os.path.getsize(p2)
        while proc0.poll() is None or proc1.poll() is None:
            sz0 = os.path.getsize(a1) if os.path.exists(a1) else 0
            sz1 = os.path.getsize(a2) if os.path.exists(a2) else 0
            pct = min(99, int((sz0 + sz1) / max(1, target_mb) * 100))
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"\r    [{bar}] {pct}%", end="", flush=True)
            time.sleep(1)

        print()
        rc0, rc1 = proc0.returncode, proc1.returncode
        if rc0 != 0 or rc1 != 0:
            return False, vname, time.time() - start_time, 0

        # Merge
        list_file = os.path.join(parent, "_merge.txt")
        with open(list_file, "w") as f:
            f.write(f"file '{a1}'\nfile '{a2}'\n")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
             "-c", "copy", output],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60
        )

        # Cleanup
        for f in [p1, p2, a1, a2, list_file]:
            try: os.remove(f)
            except: pass
        if mp4_path != video_path:
            try: os.remove(mp4_path)
            except: pass

        elapsed = time.time() - start_time
        out_size = os.path.getsize(output) / 1024 / 1024 if os.path.exists(output) else 0
        return True, vname, elapsed, out_size


def main():
    setup_env()

    if len(sys.argv) > 1:
        target_dirs = sys.argv[1:]
    else:
        target_dirs = [
            r"D:\workspace\face\phase1_video",
            r"D:\145videodata"
        ]

    all_videos = []
    for d in target_dirs:
        if os.path.exists(d):
            vids = scan_videos(d)
            print(f"[SCAN] {d}: {len(vids)} videos")
            all_videos.extend(vids)
        else:
            print(f"[SKIP] Directory not found: {d}")

    if not all_videos:
        print("No videos found.")
        return

    print(f"\n[START] {len(all_videos)} videos to process")
    print(f"[GPU] Dual RTX 4090, CUDA 13 native, NVENC")
    print()

    completed = 0
    errors = 0
    t0 = time.time()

    for i, video_path in enumerate(all_videos):
        vname = os.path.basename(video_path)
        print(f"[{i+1}/{len(all_videos)}] {vname}")
        try:
            ok, name, elapsed, out_size = process_video(video_path)
            if ok:
                completed += 1
                print(f"  OK | {elapsed:.0f}s | {out_size:.1f}MB")
            else:
                errors += 1
                print(f"  FAIL | {elapsed:.0f}s")
        except Exception as e:
            errors += 1
            print(f"  ERROR: {e}")

    total_time = time.time() - t0
    elapsed_str = str(timedelta(seconds=int(total_time)))
    print(f"\n[DONE] Completed: {completed}, Errors: {errors}, Total: {elapsed_str}")


if __name__ == "__main__":
    main()
