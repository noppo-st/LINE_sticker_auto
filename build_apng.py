import time
import math
import shutil
import subprocess
from pathlib import Path
from typing import List
from PIL import Image

# =========================
# パス設定
# =========================

ROOT = Path(__file__).resolve().parent

FFMPEG = ROOT / "tools" / "ffmpeg.exe"

VIDEOS_DIR = ROOT / "videos"
DIST_DIR = ROOT / "dist"
FRAMES_DIR = DIST_DIR / "frames"
SUBMIT_DIR = ROOT / "submit"
DONE_DIR = ROOT / "videos_done"
FAILED_DIR = ROOT / "videos_failed"

FPS = 8
MAX_BYTES = 1_000_000


# =========================
# 共通関数
# =========================

def ensure_dirs():
    for d in [VIDEOS_DIR, DIST_DIR, FRAMES_DIR, SUBMIT_DIR, DONE_DIR, FAILED_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def run(cmd: List[str]):
    print("RUN:", " ".join(map(str, cmd)))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        raise RuntimeError("Command failed")


def save_png24(im: Image.Image, path: Path):
    im.save(path, format="PNG", optimize=True)


# =========================
# 透過処理
# =========================

def chroma_key_green(img: Image.Image, tol=40):
    img = img.convert("RGBA")
    px = img.load()
    w, h = img.size

    for y in range(h):
        for x in range(w):
            r, g, b, a = px[x, y]
            if g > r + tol and g > b + tol:
                px[x, y] = (0, 0, 0, 0)

    return img


def apply_transparency(frames_dir: Path):
    for p in frames_dir.glob("*.png"):
        im = Image.open(p)
        im = chroma_key_green(im)
        save_png24(im, p)


# =========================
# mp4 → GIF → PNG
# =========================

def mp4_to_gif(mp4: Path, gif: Path):
    run([
        str(FFMPEG),
        "-y",
        "-i", str(mp4),
        "-vf",
        f"fps={FPS},scale=320:270:flags=lanczos,"
        "split[s0][s1];"
        "[s0]palettegen=stats_mode=diff:max_colors=128[p];"
        "[s1][p]paletteuse=dither=bayer",
        str(gif)
    ])


def gif_to_frames(gif: Path, out_dir: Path):
    run([
        str(FFMPEG),
        "-y",
        "-i", str(gif),
        str(out_dir / "%04d.png")
    ])


# =========================
# APNG生成（自動圧縮）
# =========================
def build_stamp(frames_dir: Path, out_path: Path, loops: int):

    frames = sorted(frames_dir.glob("*.png"))

    fps = 5
    valid_seconds = [4, 3, 2, 1]

    temp = frames_dir / "_stamp"
    temp.mkdir(exist_ok=True)

    for sec in valid_seconds:

        needed = sec * fps  # 5,10,15,20枚

        if len(frames) < needed:
            continue

        idxs = [round(i*(len(frames)-1)/(needed-1)) for i in range(needed)]
        use = [frames[i] for i in idxs]

        for f in temp.glob("*.png"):
            f.unlink()

        for i, f in enumerate(use, 1):
            im = Image.open(f).convert("RGBA")
            save_png24(im, temp / f"{i:04d}.png")

        run([
            str(FFMPEG),
            "-y",
            "-framerate", str(fps),
            "-i", str(temp / "%04d.png"),
            "-plays", str(loops),
            "-pix_fmt", "rgba",
            "-f", "apng",
            str(out_path)
        ])

        size = out_path.stat().st_size
        print("Stamp size:", size, "seconds:", sec)

        if size <= MAX_BYTES:
            break

def build_main(frames_dir: Path, batch_dir: Path):

    frames = sorted(frames_dir.glob("*.png"))

    fps = 5
    valid_seconds = [4, 3, 2, 1]

    main_path = batch_dir / "main.png"

    temp = batch_dir / "_main"
    temp.mkdir(exist_ok=True)

    for sec in valid_seconds:

        needed = sec * fps  # 5,10,15,20枚

        if len(frames) < needed:
            continue

        idxs = [round(i*(len(frames)-1)/(needed-1)) for i in range(needed)]
        use = [frames[i] for i in idxs]

        for f in temp.glob("*.png"):
            f.unlink()

        for i, f in enumerate(use, 1):

            im = Image.open(f).convert("RGBA")

            # 240x240に全体縮小（切り取りなし）
            w, h = im.size
            scale = min(240 / w, 240 / h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            resized = im.resize((new_w, new_h), Image.LANCZOS)

            canvas = Image.new("RGBA", (240, 240), (0, 0, 0, 0))
            x = (240 - new_w) // 2
            y = (240 - new_h) // 2
            canvas.paste(resized, (x, y), resized)

            save_png24(canvas, temp / f"{i:04d}.png")

        run([
            str(FFMPEG),
            "-y",
            "-framerate", str(fps),
            "-i", str(temp / "%04d.png"),
            "-plays", "1",
            "-pix_fmt", "rgba",
            "-f", "apng",
            str(main_path)
        ])

        size = main_path.stat().st_size
        print("Main size:", size, "seconds:", sec)

        if size <= MAX_BYTES:
            break

def build_tab(frames_dir: Path, batch_dir: Path):
    frames = sorted(frames_dir.glob("*.png"))
    mid = frames[len(frames)//2]
    im = Image.open(mid).convert("RGBA")
    im = im.resize((96,74),Image.LANCZOS)
    save_png24(im,batch_dir/"tab.png")


# =========================
# 単体処理
# =========================

def process_mp4(mp4: Path, batch_dir: Path):

    name = mp4.stem
    work = FRAMES_DIR / name
    work.mkdir(parents=True, exist_ok=True)

    gif = work / "temp.gif"

    mp4_to_gif(mp4, gif)
    gif_to_frames(gif, work)
    apply_transparency(work)

    frames = list(work.glob("*.png"))
    if not frames:
        return None

    duration = len(frames) / FPS
    loops = 1 if duration >= 4 else max(1,int(4/duration))

    stamp_path = batch_dir / f"{name}.png"
    build_stamp(work, stamp_path, loops)

    return work


# =========================
# 自動監視
# =========================

def main():

    ensure_dirs()

    print("📡 自動監視開始")

    processed = set()

    while True:

        mp4s = sorted(VIDEOS_DIR.glob("*.mp4"),
                      key=lambda p: p.stat().st_ctime)

        new_files = [m for m in mp4s if m not in processed]

        if new_files:

            batch_dir = SUBMIT_DIR / f"batch_{time.strftime('%Y%m%d_%H%M%S')}"
            batch_dir.mkdir(parents=True, exist_ok=True)

            first_frames = None

            for i, mp4 in enumerate(new_files):

                try:
                    frames_dir = process_mp4(mp4, batch_dir)

                    if i == 0:
                        first_frames = frames_dir

                    shutil.move(str(mp4), DONE_DIR / mp4.name)

                except Exception as e:
                    print("❌ エラー:", e)
                    shutil.move(str(mp4), FAILED_DIR / mp4.name)

                processed.add(mp4)

            if first_frames:
                build_main(first_frames, batch_dir)
                build_tab(first_frames, batch_dir)

            print("✅ batch 完了")

        time.sleep(5)


if __name__ == "__main__":
    main()
