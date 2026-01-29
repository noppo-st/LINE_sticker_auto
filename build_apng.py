# -*- coding: utf-8 -*-
"""
LINE sticker APNG builder (Windows / PowerShell)

やること（完成版の仕様）
- mp4 -> frames(RGBA) -> 背景黒の除去（外周から連結している黒だけ） -> APNG(.png)
- 本体の黒（黒目・首輪など）は消さない（外周連結のみflood fill）
- PNG-24相当（RGBA truecolor）で保存（パレット化で審査NGを避ける）
- 1MB超過なら、(scale_w / colors / fps) のプリセットを順に試して落とす
- 秒数は必ず「ぴったり」にする（審査エラー対策）
    * 動画長 <= 2.0 秒 : loops=2 固定、合計 4秒ぴったり（= 2秒ぶんを2回）
    * 動画長 >  2.0 秒 : loops=1 固定、合計 3秒 or 4秒ぴったり（近い方）
- submit package を作る：upload.zip + main.png(APNG 240x240) + tab.png(PNG 96x74)

Requirements:
- ffmpeg, ffprobe are in PATH
- pip install pillow
"""

from __future__ import annotations

import csv
import time
import shutil
import zipfile
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

from PIL import Image


# ----------------------------
# Paths
# ----------------------------
ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT / "videos"
DIST_DIR = ROOT / "dist"
FRAMES_DIR = DIST_DIR / "frames"
APNG_DIR = DIST_DIR / "apng"
SUBMIT_DIR = ROOT / "submit"
REPORT_CSV = DIST_DIR / "report.csv"


# ----------------------------
# LINE constraints / targets
# ----------------------------
STICKER_MAX_W = 320
STICKER_MAX_H = 270
MAIN_SIZE = (240, 240)   # main: APNG
TAB_SIZE = (96, 74)      # tab: PNG

MAX_BYTES = 1_000_000  # 1MB

# 「秒数ぴったり」作成のためのフレーム範囲
MIN_FRAMES = 5
MAX_FRAMES = 20

# main は常に 4秒ぴったり（loops=1固定）
MAIN_SECONDS = 4

# tab の半透明が原因で「透過されていない」扱いになるケースがあるため
# tabだけ α2値化して mid(1..254) を消す（本体フレームは基本しない）
ALPHA_BIN_THRESHOLD = 160
BINARIZE_TAB = True
BINARIZE_MAIN = True
BINARIZE_STICKER_FRAMES = False  # ← 本体が透ける場合があるので通常はFalse推奨

# ----------------------------
# Background removal: "edge-connected black"
# ----------------------------
# 黒判定（控えめ）
EDGE_BLACK_LUMA_MAX = 24   # 低いほど「黒扱い」が減る（本体を守る）
EDGE_BLACK_RGB_MAX = 28

# 境界を確実に透明にするための透明枠(px)
BORDER_TRANSPARENT_PX = 1

# 被写体が外周に接して「本体まで消える」保険（少し縮小）
SAFE_SHRINK = 0.96  # 0.94〜0.98で調整


# ----------------------------
# Presets (sticker)
# ----------------------------
# fps / scale_w / colors（この順で試して1MB以下を狙う）
# colors を落とすほど軽くなる（ただし画質は落ちる）
PRESETS: List[Tuple[int, int, int]] = [
    (8, 320, 64),
    (8, 320, 56),
    (7, 320, 56),
    (6, 320, 56),
    (6, 320, 48),
    (5, 320, 48),

    (8, 288, 56),
    (7, 288, 56),
    (6, 288, 48),
    (5, 288, 48),

    (6, 272, 48),
    (5, 272, 48),
    (4, 272, 48),
    (4, 256, 48),
    (4, 256, 40),
]

# main（4秒固定）のプリセット
# fps を下げる/framesを減らす/scaleを下げる/colorsを落とす、で 1MB以下へ
MAIN_PRESETS: List[Tuple[int, int, int, int]] = [
    # fps, scale_w, colors, pad
    (5, 240, 64, 2),   # 4*5=20 frames
    (4, 240, 64, 2),   # 16 frames
    (4, 224, 56, 2),
    (4, 224, 48, 2),
    (3, 224, 48, 2),   # 12 frames
    (3, 208, 48, 2),
    (3, 208, 40, 2),
]


# ----------------------------
# Utils
# ----------------------------
def run(cmd: List[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """cp932問題回避（UTF-8で読み、読めない文字は置換）"""
    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )


def ensure_dirs() -> None:
    for d in [VIDEOS_DIR, DIST_DIR, FRAMES_DIR, APNG_DIR, SUBMIT_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def check_tools() -> None:
    missing = []
    for t in ["ffmpeg", "ffprobe"]:
        if shutil.which(t) is None:
            missing.append(t)
    if missing:
        raise RuntimeError(
            f"Missing tools in PATH: {missing}\n"
            f"PowerShellで where ffmpeg / where ffprobe を確認してください"
        )


def safe_stem(_name: str) -> str:
    """日本語等は英数字に置換して basename にする（時刻）"""
    ts = time.strftime("%Y%m%d_%H%M%S")
    return f"video_{ts}"


def clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def luma(r: int, g: int, b: int) -> int:
    return int(0.2126 * r + 0.7152 * g + 0.0722 * b)


def save_png24_rgba(img: Image.Image, path: Path) -> None:
    """
    必ず PNG-24 相当（RGBA truecolor）で保存。
    optimize=True だと環境によってパレット化することがあるので避ける。
    """
    img = img.convert("RGBA")
    img.save(path, "PNG", optimize=False, compress_level=9)


def border_alpha_minmax(img_rgba: Image.Image, border: int = 1) -> Tuple[int, int]:
    a = img_rgba.getchannel("A")
    w, h = img_rgba.size
    b = border
    vals = []
    for x in range(w):
        for y in range(min(b, h)):
            vals.append(a.getpixel((x, y)))
            vals.append(a.getpixel((x, h - 1 - y)))
    for y in range(h):
        for x in range(min(b, w)):
            vals.append(a.getpixel((x, y)))
            vals.append(a.getpixel((w - 1 - x, y)))
    if not vals:
        return (0, 0)
    return (min(vals), max(vals))


def force_transparent_border(img_rgba: Image.Image, border: int = 1) -> Image.Image:
    """外周 border px を強制透明に"""
    img = img_rgba.copy()
    w, h = img.size
    px = img.load()
    for x in range(w):
        for y in range(border):
            r, g, b, _a = px[x, y]
            px[x, y] = (r, g, b, 0)
            r, g, b, _a = px[x, h - 1 - y]
            px[x, h - 1 - y] = (r, g, b, 0)
    for y in range(h):
        for x in range(border):
            r, g, b, _a = px[x, y]
            px[x, y] = (r, g, b, 0)
            r, g, b, _a = px[w - 1 - x, y]
            px[w - 1 - x, y] = (r, g, b, 0)
    return img


def safe_shrink_to_make_margin(img_rgba: Image.Image, shrink: float = 0.96) -> Image.Image:
    """
    画像を少し縮小して中央に配置（透明キャンバス）。
    被写体が外周に接しているときの「本体が消える」対策。
    """
    w, h = img_rgba.size
    nw = max(1, int(w * shrink))
    nh = max(1, int(h * shrink))
    small = img_rgba.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ox = (w - nw) // 2
    oy = (h - nh) // 2
    canvas.alpha_composite(small, (ox, oy))
    return canvas

def binarize_alpha(img_rgba: Image.Image, threshold: int = 160) -> Image.Image:
    """半透明(1..254)を無くす：threshold以上は255、未満は0"""
    im = img_rgba.copy()
    a = im.getchannel("A")
    a2 = a.point(lambda v: 255 if v >= threshold else 0)
    im.putalpha(a2)
    return im


def edge_connected_black_to_transparent(img_rgba: Image.Image) -> Image.Image:
    """
    外周から連結している“黒だけ”を透明化（flood fill）。
    ※本体の黒（黒目・首輪など）は「外周に接続していない限り」消えない
    """
    img = img_rgba.convert("RGBA").copy()
    w, h = img.size
    px = img.load()

    def is_blackish(r: int, g: int, b: int) -> bool:
        if luma(r, g, b) <= EDGE_BLACK_LUMA_MAX:
            return True
        if max(r, g, b) <= EDGE_BLACK_RGB_MAX:
            return True
        return False

    visited = [[False] * h for _ in range(w)]
    stack: List[Tuple[int, int]] = []

    # 外周から開始
    for x in range(w):
        stack.append((x, 0))
        stack.append((x, h - 1))
    for y in range(h):
        stack.append((0, y))
        stack.append((w - 1, y))

    while stack:
        x, y = stack.pop()
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        if visited[x][y]:
            continue
        visited[x][y] = True

        r, g, b, a = px[x, y]
        if a == 0:
            # 透明は背景領域として連結扱いでOK
            pass
        else:
            if not is_blackish(r, g, b):
                continue

        # 背景（黒 or 透明）として透明化
        px[x, y] = (r, g, b, 0)

        # 4近傍
        stack.append((x + 1, y))
        stack.append((x - 1, y))
        stack.append((x, y + 1))
        stack.append((x, y - 1))

    return img


def estimate_border_bg_color(img: Image.Image, border: int = 2, a_min: int = 200) -> Tuple[int, int, int]:
    """
    外周(border px)の「不透明寄り(alpha>=a_min)」から背景色を推定（平均RGB）
    """
    im = img.convert("RGBA")
    px = im.load()
    w, h = im.size
    vals: List[Tuple[int, int, int]] = []

    def add(x: int, y: int):
        r, g, b, a = px[x, y]
        if a >= a_min:
            vals.append((r, g, b))

    for x in range(w):
        for y in range(border):
            add(x, y)
            add(x, h - 1 - y)
    for y in range(h):
        for x in range(border):
            add(x, y)
            add(w - 1 - x, y)

    if not vals:
        return (0, 0, 0)

    r = sum(v[0] for v in vals) / len(vals)
    g = sum(v[1] for v in vals) / len(vals)
    b = sum(v[2] for v in vals) / len(vals)
    return (int(r), int(g), int(b))


def border_connected_bg_to_transparent(img_rgba: Image.Image, tol: int = 25) -> Image.Image:
    """
    外周に接続している「背景色に近い画素」だけ透明化する。
    ※黒背景以外（暗いグラデ等）に効く“保険”。
    tol を上げるほど消える範囲が増える（上げ過ぎ注意）
    """
    img = img_rgba.convert("RGBA").copy()
    w, h = img.size
    px = img.load()

    bg = estimate_border_bg_color(img, border=2, a_min=200)

    def close_to_bg(r: int, g: int, b: int) -> bool:
        return (abs(r - bg[0]) <= tol and abs(g - bg[1]) <= tol and abs(b - bg[2]) <= tol)

    visited = [[False] * h for _ in range(w)]
    stack: List[Tuple[int, int]] = []

    for x in range(w):
        stack.append((x, 0))
        stack.append((x, h - 1))
    for y in range(h):
        stack.append((0, y))
        stack.append((w - 1, y))

    while stack:
        x, y = stack.pop()
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        if visited[x][y]:
            continue
        visited[x][y] = True

        r, g, b, a = px[x, y]
        if a == 0:
            pass
        else:
            if not close_to_bg(r, g, b):
                continue

        px[x, y] = (r, g, b, 0)

        stack.append((x + 1, y))
        stack.append((x - 1, y))
        stack.append((x, y + 1))
        stack.append((x, y - 1))

    return img


    while stack:
        x, y = stack.pop()
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        if visited[x][y]:
            continue
        visited[x][y] = True

        r, g, b, a = px[x, y]

        # 透明は通路としてOK（ただし「透明＝背景」なのでそのままで良い）
        if a == 0:
            pass
        else:
            # 背景色に近いものだけを「背景」とみなす（黒目などは通常ここで弾かれる）
            if not close_to_bg(r, g, b):
                continue

        px[x, y] = (r, g, b, 0)

        stack.append((x + 1, y))
        stack.append((x - 1, y))
        stack.append((x, y + 1))
        stack.append((x, y - 1))

    return img

 
    """
    外周から連結している「黒っぽい領域」だけを透明化（flood fill）。
    本体内部の黒（黒目・首輪）は消さない。
    """
    img = img_rgba.copy()
    w, h = img.size
    px = img.load()

    def is_blackish(r: int, g: int, b: int) -> bool:
        if luma(r, g, b) <= EDGE_BLACK_LUMA_MAX:
            return True
        if max(r, g, b) <= EDGE_BLACK_RGB_MAX:
            return True
        return False

    visited = [[False] * h for _ in range(w)]
    stack: List[Tuple[int, int]] = []

    # 外周から開始
    for x in range(w):
        stack.append((x, 0))
        stack.append((x, h - 1))
    for y in range(h):
        stack.append((0, y))
        stack.append((w - 1, y))

    while stack:
        x, y = stack.pop()
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        if visited[x][y]:
            continue
        visited[x][y] = True

        r, g, b, a = px[x, y]

        # 透明は背景領域として連結扱いOK
        if a != 0 and (not is_blackish(r, g, b)):
            continue

        # 背景として透明化
        px[x, y] = (r, g, b, 0)

        # 4近傍
        stack.append((x + 1, y))
        stack.append((x - 1, y))
        stack.append((x, y + 1))
        stack.append((x, y - 1))

    return img


def fit_within_sticker_max(frames_dir: Path) -> None:
    """抽出PNGを 320x270 以内に収める（透明キャンバスで中央配置）"""
    pngs = sorted(frames_dir.glob("*.png"))
    for p in pngs:
        im = Image.open(p).convert("RGBA")
        w, h = im.size
        scale = min(STICKER_MAX_W / w, STICKER_MAX_H / h, 1.0)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        if (nw, nh) != (w, h):
            im = im.resize((nw, nh), Image.Resampling.LANCZOS)

        canvas = Image.new("RGBA", (STICKER_MAX_W, STICKER_MAX_H), (0, 0, 0, 0))
        ox = (STICKER_MAX_W - im.size[0]) // 2
        oy = (STICKER_MAX_H - im.size[1]) // 2
        canvas.alpha_composite(im, (ox, oy))
        save_png24_rgba(canvas, p)


def composite_on_transparent(src_rgba: Image.Image, size: Tuple[int, int], pad: int) -> Image.Image:
    """
    srcを size の透明キャンバスに、できるだけ余白少なく（padだけ確保）
    """
    w, h = size
    src = src_rgba.copy()
    bbox = src.getbbox()
    if bbox:
        src = src.crop(bbox)

    max_w = max(1, w - pad * 2)
    max_h = max(1, h - pad * 2)
    sw, sh = src.size
    scale = min(max_w / sw, max_h / sh, 1.0)
    nw = max(1, int(sw * scale))
    nh = max(1, int(sh * scale))
    if (nw, nh) != (sw, sh):
        src = src.resize((nw, nh), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ox = (w - src.size[0]) // 2
    oy = (h - src.size[1]) // 2
    canvas.alpha_composite(src, (ox, oy))
    return canvas


def extract_frames_ffmpeg(mp4: Path, out_dir: Path, fps: int, scale_w: int) -> None:
    """mp4 → 連番PNG抽出（RGBA）"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    pattern = str(out_dir / "%04d.png")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(mp4),
        "-vf", f"fps={fps},scale={scale_w}:-1:flags=lanczos,format=rgba",
        pattern,
    ]
    p = run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg extract failed:\n{p.stderr}")


def get_duration_seconds(mp4: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        str(mp4),
    ]
    p = run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{p.stderr}")
    try:
        return float(p.stdout.strip())
    except Exception:
        return 0.0


def build_apng_from_frames(frames_dir: Path, fps: int, loops: int, out_apng: Path) -> None:
    """
    連番PNG → APNG（拡張子は .png でOK）
    loops: APNGのplays回数（0は無限）なので 1 or 2 などを入れる
    """
    pattern = str(frames_dir / "%04d.png")
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-plays", str(loops),
        "-pix_fmt", "rgba",
        "-f", "apng",
        str(out_apng),
    ]
    p = run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg apng failed:\n{p.stderr}")


def ensure_exact_frame_count(frames_dir: Path, frames_target: int) -> None:
    """
    %04d.png を frames_target 枚に揃える。
    - 多い: 間引き
    - 少ない: 最後のフレームを複製
    """
    pngs = sorted(frames_dir.glob("*.png"))
    if not pngs:
        return

    # 多い → 間引き
    if len(pngs) > frames_target:
        idxs = [round(i * (len(pngs) - 1) / (frames_target - 1)) for i in range(frames_target)]
        kept = [pngs[i] for i in idxs]
        kept_bytes = [p.read_bytes() for p in kept]

        for old in frames_dir.glob("*.png"):
            old.unlink()
        for i, b in enumerate(kept_bytes, start=1):
            (frames_dir / f"{i:04d}.png").write_bytes(b)

    # 少ない → 最後を複製
    pngs = sorted(frames_dir.glob("*.png"))
    if len(pngs) < frames_target:
        last_bytes = pngs[-1].read_bytes()
        for i in range(len(pngs) + 1, frames_target + 1):
            (frames_dir / f"{i:04d}.png").write_bytes(last_bytes)


def apply_transparency_pipeline(frames_dir: Path) -> None:
    """
    フレームに対して：
    - 必要なら shrink（外周と被写体が接している時の保険）
    - ①外周連結の黒だけ透明化（本体の黒目・首輪は守られる）
    - ②それでも背景が残る“場合だけ” 背景色推定で追加除去（弱め）
    - 外周1pxを完全透明化
    - PNG-24(RGBA truecolor)で保存
    ※本体が透けるのを避けるため、基本は二値化しない（必要な場合のみ）
    """
    pngs = sorted(frames_dir.glob("*.png"))
    for p in pngs:
        im = Image.open(p).convert("RGBA")

        # 外周に不透明がある＝被写体が縁に接している可能性 → 少し縮小して安全マージン
        bmin, bmax = border_alpha_minmax(im, border=1)
        if bmax > 0:
            im = safe_shrink_to_make_margin(im, shrink=SAFE_SHRINK)

        # ① 本命：外周連結「黒だけ」除去（本体の黒は外周に繋がってない限り残る）
        im = edge_connected_black_to_transparent(im)

        # ② 保険：まだ背景が残っている場合だけ、背景色推定で“弱め”に追加除去
        #    ※毎回やると本体の暗部が消えやすいので、条件付きにする
        bg = estimate_border_bg_color(im, border=2, a_min=200)
        # 背景が「そこそこ暗い」時だけ実行（真っ黒/暗背景対策用）
        if (bg[0] + bg[1] + bg[2]) <= 120:
            im = border_connected_bg_to_transparent(im, tol=18)  # tolは弱め（大きいと本体が危険）

        # 外周の取りこぼしを確実に消す
        im = force_transparent_border(im, border=BORDER_TRANSPARENT_PX)

        # 半透明を消したいときだけ（通常False推奨）
        if BINARIZE_STICKER_FRAMES:
            im = binarize_alpha(im, threshold=ALPHA_BIN_THRESHOLD)

        save_png24_rgba(im, p)

def reduce_colors_in_frames(frames_dir: Path, colors: int) -> None:
    """
    frames_dir 内の連番PNGを減色して軽量化する（αは保持）
    最後は必ず PNG-24(RGBA truecolor)で保存する（審査対策）
    """
    pngs = sorted(frames_dir.glob("*.png"))
    if not pngs:
        return

    for p in pngs:
        im = Image.open(p).convert("RGBA")
        alpha = im.getchannel("A")

        # RGBだけ減色 → αを戻す
        rgb = im.convert("RGB")
        pal = rgb.quantize(
            colors=max(2, min(int(colors), 256)),
            method=Image.MEDIANCUT,
            dither=Image.NONE
        )
        back = pal.convert("RGBA")
        back.putalpha(alpha)

        save_png24_rgba(back, p)


def choose_seconds_and_loops(dur: float) -> Tuple[int, int]:
    """
    秒数とloopsを決める（必ず「ぴったり」作れる前提）
    - dur<=2.0 → seconds=4, loops=2（2秒ぶんを2回で4秒）
    - dur>2.0  → loops=1、secondsは 3 or 4 の近い方
    """
    if dur <= 2.0:
        return 4, 2
    # 3と4の近い方（審査で 1..4 秒に収める）
    # 例: 2.7→3, 3.6→4
    seconds = 4 if dur >= 3.5 else 3
    return seconds, 1


def frames_target_for(seconds: int, fps: int, loops: int) -> int:
    """
    合計秒数を seconds に「ぴったり」合わせるための frames_target を返す。
    total_duration = (frames_target / fps) * loops == seconds
    => frames_target = seconds * fps / loops
    """
    target = seconds * fps
    if target % loops != 0:
        # 割り切れない場合は呼び出し側でfpsを変える（この関数では失敗扱い）
        return -1
    return target // loops


def build_one(mp4: Path, out_basename: str) -> "ResultRow":
    dur = get_duration_seconds(mp4)
    seconds, loops = choose_seconds_and_loops(dur)

    last_err = ""
    last_size = 0
    last_path = ""

    for (fps, scale_w, colors) in PRESETS:
        ft = frames_target_for(seconds, fps, loops)
        if ft < 0:
            continue
        if not (MIN_FRAMES <= ft <= MAX_FRAMES):
            continue

        work_frames = FRAMES_DIR / out_basename
        extract_frames_ffmpeg(mp4, work_frames, fps=fps, scale_w=scale_w)

        fit_within_sticker_max(work_frames)
        ensure_exact_frame_count(work_frames, ft)

        apply_transparency_pipeline(work_frames)
        reduce_colors_in_frames(work_frames, colors)

        out_apng = APNG_DIR / f"{out_basename}.png"

        try:
            build_apng_from_frames(work_frames, fps=fps, loops=loops, out_apng=out_apng)
        except Exception as e:
            last_err = str(e)
            continue

        size = out_apng.stat().st_size if out_apng.exists() else 0
        last_size = size
        last_path = str(out_apng)

        detail = (
            f"dur={dur:.2f}s target={seconds}s frames={ft} fps={fps} loops={loops} "
            f"colors={colors} scale_w={scale_w} size={size}B"
        )

        if size <= MAX_BYTES:
            return ResultRow(
                src_filename=mp4.name,
                out_basename=out_basename,
                status="ok",
                detail=detail,
                bytes=size,
                apng_path=str(out_apng),
                seconds=seconds,
                frames=ft,
                fps=fps,
                loops=loops,
                order=0, 
            )

    # ここまで来たらサイズ超過 or 失敗
    if last_path and Path(last_path).exists():
        return ResultRow(
            src_filename=mp4.name,
            out_basename=out_basename,
            status="ng",
            detail=f"OVER 1MB or failed presets. last={last_size}B err={last_err}",
            bytes=last_size,
            apng_path=last_path,
            seconds=seconds,
            frames=0,
            fps=0,
            loops=loops,
            order=0,
        )

    return ResultRow(

        src_filename=mp4.name,
        out_basename=out_basename,
        status="ng",
        detail=f"failed: {last_err}",
        bytes=0,
        apng_path="",
        seconds=seconds,
        frames=0,
        fps=0,
        loops=loops,
    )


def extract_frames_from_apng_with_ffmpeg(apng_path: Path, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):
        old.unlink()

    pattern = str(out_dir / "%04d.png")
    cmd = [
        "ffmpeg", "-y",
        "-f", "apng",
        "-i", str(apng_path),
        pattern,
    ]
    p = run(cmd)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg apng->frames failed:\n{p.stderr}")
    return sorted(out_dir.glob("*.png"))


def make_tab_png_from_last_frame(last_frame_rgba: Image.Image, batch_dir: Path) -> Path:
    tab = composite_on_transparent(last_frame_rgba, TAB_SIZE, pad=2)

    bmin, bmax = border_alpha_minmax(tab, border=1)
    if bmax > 0:
        tab = safe_shrink_to_make_margin(tab, shrink=SAFE_SHRINK)

    tab = force_transparent_border(tab, border=BORDER_TRANSPARENT_PX)

    if BINARIZE_TAB:
        tab = binarize_alpha(tab, threshold=ALPHA_BIN_THRESHOLD)

    out_tab = batch_dir / "tab.png"
    save_png24_rgba(tab, out_tab)
    return out_tab


def make_main_apng_from_apng(apng_path: Path, batch_dir: Path) -> Path:
    """
    main.png: 必ず 4秒ぴったりの “動く” APNG を作る。
    - loops=1 固定
    - MAIN_PRESETS を試して 1MB 以下を狙う
    """
    frames_src_dir = batch_dir / "_tmp_main_extract"
    frames = extract_frames_from_apng_with_ffmpeg(apng_path, frames_src_dir)
    if not frames:
        raise RuntimeError("No frames extracted for main")

    out_main = batch_dir / "main.png"
    last_err = ""
    last_size = 0

    loops = 1
    seconds = MAIN_SECONDS

    for (fps, scale_w, colors, pad) in MAIN_PRESETS:
        ft = frames_target_for(seconds, fps, loops)
        if ft < 0:
            continue
        ft = clamp(ft, MIN_FRAMES, MAX_FRAMES)

        main_frames_dir = batch_dir / "_main_frames"
        main_frames_dir.mkdir(parents=True, exist_ok=True)
        for old in main_frames_dir.glob("*.png"):
            old.unlink()

        total = len(frames)
        if total >= ft:
            idxs = [round(i * (total - 1) / (ft - 1)) for i in range(ft)]
        else:
            idxs = list(range(total)) + [total - 1] * (ft - total)

        # mainフレーム作成
        for i, idx in enumerate(idxs, start=1):
            fr = Image.open(frames[idx]).convert("RGBA")

            # まず mainサイズの透明キャンバスへ
            main_rgba = composite_on_transparent(fr, MAIN_SIZE, pad=pad)

            bmin, bmax = border_alpha_minmax(main_rgba, border=1)
            if bmax > 0:
                main_rgba = safe_shrink_to_make_margin(main_rgba, shrink=SAFE_SHRINK)

            main_rgba = edge_connected_black_to_transparent(main_rgba)
            main_rgba = force_transparent_border(main_rgba, border=BORDER_TRANSPARENT_PX)

            if BINARIZE_MAIN:
                main_rgba = binarize_alpha(main_rgba, threshold=ALPHA_BIN_THRESHOLD)

            save_png24_rgba(main_rgba, main_frames_dir / f"{i:04d}.png")

        # 念のため枚数を固定
        ensure_exact_frame_count(main_frames_dir, ft)

        # 減色（mainも軽量化）
        reduce_colors_in_frames(main_frames_dir, colors)

        # 出力
        try:
            build_apng_from_frames(main_frames_dir, fps=fps, loops=loops, out_apng=out_main)
        except Exception as e:
            last_err = str(e)
            continue

        size = out_main.stat().st_size if out_main.exists() else 0
        last_size = size

        if size <= MAX_BYTES:
            return out_main

    raise RuntimeError(f"main over 1MB or failed. last={last_size}B err={last_err}")


# ----------------------------
# Result
# ----------------------------
@dataclass
class ResultRow:
    src_filename: str
    out_basename: str
    status: str          # ok/ng
    detail: str
    bytes: int
    apng_path: str
    seconds: int
    frames: int
    fps: int
    loops: int
    order: int 

# ----------------------------
# submit package
# ----------------------------
def create_submit_package(rows: List[ResultRow]) -> Optional[Path]:
    ok_rows = [r for r in rows if r.status == "ok" and r.apng_path]
    if not ok_rows:
        print("[submit] no ok rows -> skip submit package")
        return None

    ts = time.strftime("%Y%m%d_%H%M%S")
    batch = SUBMIT_DIR / f"batch_{ts}"
    upload_dir = batch / "upload"
    apng_dir = batch / "apng"
    upload_dir.mkdir(parents=True, exist_ok=True)
    apng_dir.mkdir(parents=True, exist_ok=True)

    # 1) upload/apng へコピー
    copied_apngs: List[Path] = []
    for r in ok_rows:
        src = Path(r.apng_path)
        if not src.exists():
            continue

        dst_upload = upload_dir / src.name
        shutil.copy2(src, dst_upload)

        dst_apng = apng_dir / src.name
        shutil.copy2(src, dst_apng)
        copied_apngs.append(dst_apng)

    if not copied_apngs:
        print("[submit] no apng copied -> skip submit package")
        return None

    # list.csv
    with (batch / "list.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_filename", "apng_file", "bytes", "detail"])
        for r in ok_rows:
            w.writerow([r.src_filename, Path(r.apng_path).name, r.bytes, r.detail])

        # 2) main/tab 作成（最初の動画を必ず使う）
    try:
        # ok_rows を「入力順（order）」でソートして先頭を採用
        ok_rows_sorted = sorted(ok_rows, key=lambda r: getattr(r, "order", 10**9))

        base_apng = Path(ok_rows_sorted[0].apng_path)
        if not base_apng.exists():
            raise RuntimeError(f"base apng not found: {base_apng}")

        # main（動くAPNG）
        make_main_apng_from_apng(base_apng, batch_dir=batch)

        # tab（最後フレームから静止PNG）
        tmp = batch / "_tmp_tab_extract"
        frames = extract_frames_from_apng_with_ffmpeg(base_apng, tmp)
        last = Image.open(frames[-1]).convert("RGBA")
        make_tab_png_from_last_frame(last, batch_dir=batch)

        (batch / "main_tab_ok.txt").write_text(f"source={base_apng.name}\n", encoding="utf-8")
        print(f"[submit] main/tab created in {batch}")
    except Exception as e:
        (batch / "main_tab_error.txt").write_text(str(e), encoding="utf-8")
        print("[submit] main/tab FAILED -> wrote main_tab_error.txt")


    # 3) ZIP作成（upload配下＋main/tab）
    zip_path = batch / "upload.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in upload_dir.glob("*.png"):
            z.write(p, arcname=p.name)
        for p in [batch / "main.png", batch / "tab.png"]:
            if p.exists():
                z.write(p, arcname=p.name)

    print(f"SUBMIT: {batch}")
    print(f"✅ 提出用ZIP: {zip_path}")
    return batch

# ----------------------------
# main
# ----------------------------
def main() -> None:
    ensure_dirs()
    check_tools()

    mp4s = sorted(VIDEOS_DIR.glob("*.mp4"))
    if not mp4s:
        print(f"No mp4 in {VIDEOS_DIR}")
        return

    rows: List[ResultRow] = []

    for idx, mp4 in enumerate(mp4s, start=1):
        out_base = safe_stem(mp4.name)

        try:
            r = build_one(mp4, out_base)

            # ★順番を保持したい場合：ResultRowに order があるならここで入れる
            if hasattr(r, "order"):
                r.order = idx

            rows.append(r)

            if r.status == "ok":
                print(f"{idx:02d} {mp4.name} ok {r.detail} -> {Path(r.apng_path).name}")
            else:
                print(f"{idx:02d} {mp4.name} NG {r.detail}")

        except Exception as e:
            # ★ここで e を使う（except の外では使わない！）
            print(f"{idx:02d} {mp4.name} NG: {e}")

            rows.append(ResultRow(
                src_filename=mp4.name,
                out_basename=out_base,
                status="ng",
                detail=str(e),
                bytes=0,
                apng_path="",
                seconds=0,
                frames=0,
                fps=0,
                loops=0,
            ))

            print(f"{mp4.name} NG")

    # report.csv
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    with REPORT_CSV.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["src_filename", "out_basename", "status", "detail", "bytes", "apng_path", "seconds", "frames", "fps", "loops"])
        for r in rows:
            w.writerow([r.src_filename, r.out_basename, r.status, r.detail, r.bytes, r.apng_path, r.seconds, r.frames, r.fps, r.loops])

    print("DONE")
    print(f"APNG: {APNG_DIR}")
    print(f"REPORT: {REPORT_CSV}")

    # submit
    create_submit_package(rows)


if __name__ == "__main__":
    main()
