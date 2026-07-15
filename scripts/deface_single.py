#!/usr/bin/env python
"""
Single video deface with CUDA 13 native + NVENC + auto MTS→MP4 conversion.
Usage:
  python deface_single.py <video_path>
  python deface_single.py "D:\\workspace\\face\\video.MTS"
  python deface_single.py video.mp4 --no-dual  # force single GPU
"""
import os, sys, time, subprocess

DEFACE_CMD = r"D:\face_anon_env\Scripts\deface"
FFMPEG_CFG = '{"codec":"h264_nvenc"}'
BACKEND = "onnxrt"
EP = "CUDAExecutionProvider"


def setup_env():
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
    if src.lower().endswith(".mp4"):
        return src
    dst = os.path.splitext(src)[0] + ".mp4"
    if os.path.exists(dst):
        return dst
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-c", "copy", "-map", "0:v", "-map", "0:a?", dst],
        capture_output=True, text=True, timeout=180
    )
    if r.returncode != 0:
        subprocess.run(
            ["ffmpeg", "-y", "-i", src, "-c:v", "libx264", "-c:a", "aac",
             "-preset", "fast", dst],
            capture_output=True, text=True, timeout=300
        )
    return dst


def split_video(mp4_path, out_dir):
    dur = get_duration(mp4_path)
    if dur <= 2:
        base = os.path.splitext(os.path.basename(mp4_path))[0]
        p1 = os.path.join(out_dir, f"{base}_p1.mp4")
        subprocess.run(["ffmpeg", "-y", "-i", mp4_path, "-c", "copy", p1],
                       capture_output=True, timeout=60)
        return p1, p1

    mid = dur / 2
    base = os.path.splitext(os.path.basename(mp4_path))[0]
    p1 = os.path.join(out_dir, f"{base}_p1.mp4")
    p2 = os.path.join(out_dir, f"{base}_p2.mp4")
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp4_path, "-t", str(mid), "-c", "copy", p1],
        capture_output=True, timeout=60
    )
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp4_path, "-ss", str(mid), "-c", "copy", p2],
        capture_output=True, timeout=60
    )
    return p1, p2


def deface_part(input_path, output_path, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    cmd = [DEFACE_CMD, input_path, "-k", "--backend", BACKEND, "--ep", EP,
           "--ffmpeg-config", FFMPEG_CFG, "-o", output_path]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python deface_single.py <video_path> [--no-dual]")
        sys.exit(1)

    video_path = sys.argv[1]
    no_dual = "--no-dual" in sys.argv

    if not os.path.exists(video_path):
        print(f"ERROR: File not found: {video_path}")
        sys.exit(1)

    setup_env()
    vname = os.path.basename(video_path)
    parent = os.path.dirname(video_path) or "."
    base_name = os.path.splitext(vname)[0]
    output = os.path.join(parent, f"{base_name}_anonymized.mp4")
    t0 = time.time()

    # Convert
    print(f"[1/4] Convert to MP4: {vname}")
    mp4 = to_mp4(video_path)

    if no_dual:
        print(f"[2/4] Single GPU deface (CUDA 13)...")
        env = os.environ.copy()
        t1 = time.time()
        r = subprocess.run(
            [DEFACE_CMD, mp4, "-k", "--backend", BACKEND, "--ep", EP,
             "--ffmpeg-config", FFMPEG_CFG, "-o", output],
            env=env
        )
        if r.returncode != 0:
            print("ERROR: deface failed")
            sys.exit(1)
        elapsed = time.time() - t1
    else:
        # Split
        print(f"[2/4] Split video...")
        p1, p2 = split_video(mp4, parent)
        a1 = p1.replace("_p1.mp4", "_a1.mp4")
        a2 = p2.replace("_p2.mp4", "_a2.mp4")

        if p1 == p2:
            print(f"[3/4] Short video, single GPU deface...")
            env = os.environ.copy()
            t1 = time.time()
            r = subprocess.run(
                [DEFACE_CMD, p1, "-k", "--backend", BACKEND, "--ep", EP,
                 "--ffmpeg-config", FFMPEG_CFG, "-o", output], env=env
            )
            if r.returncode != 0:
                print("ERROR: deface failed")
                sys.exit(1)
            elapsed = time.time() - t1
        else:
            print(f"[3/4] Dual GPU deface (GPU0 + GPU1)...")
            t1 = time.time()
            proc0 = deface_part(p1, a1, 0)
            proc1 = deface_part(p2, a2, 1)
            target_mb = os.path.getsize(p1) + os.path.getsize(p2)

            while proc0.poll() is None or proc1.poll() is None:
                sz0 = os.path.getsize(a1) if os.path.exists(a1) else 0
                sz1 = os.path.getsize(a2) if os.path.exists(a2) else 0
                pct = min(99, int((sz0 + sz1) / max(1, target_mb) * 100))
                bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
                print(f"\r    [{bar}] {pct}%", end="", flush=True)
                time.sleep(1)

            print()
            elapsed = time.time() - t1
            if proc0.returncode != 0 or proc1.returncode != 0:
                print("ERROR: deface failed on one or both GPUs")
                sys.exit(1)

            print(f"[4/4] Merging...")
            list_file = os.path.join(parent, "_merge.txt")
            with open(list_file, "w") as f:
                f.write(f"file '{a1}'\nfile '{a2}'\n")
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
                 "-c", "copy", output],
                capture_output=True, timeout=60
            )

            # Cleanup
            for f in [p1, p2, a1, a2, list_file]:
                try: os.remove(f)
                except: pass

    # Cleanup converted MP4
    if mp4 != video_path:
        try: os.remove(mp4)
        except: pass

    out_size = os.path.getsize(output) / 1024 / 1024
    total_time = time.time() - t0
    print(f"\nDone! {output} ({out_size:.1f}MB, {total_time:.0f}s)")


if __name__ == "__main__":
    main()
