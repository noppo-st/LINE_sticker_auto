# 🎬 LINEアニメーションスタンプ自動作成ツール（Windows用）

このツールは、MP4動画を入れるだけで、LINEアニメーションスタンプ用ファイルを自動で作ります。

---

# 🖥 最初にやること（1回だけ）

## ① LINE_sticker_auto.zipをダウンロードする。
1. LINE_sticker_auto.zip を右クリック
2.「すべて展開」を展開する。
3. LINE_sticker_autoフォルダーをデスクトップに置く（お勧め）

---

## ② build_apng.py を入れる
1. GitHubページで build_apng.py をクリック  
2. 右上の「Download」を押す  
3. ダウンロードした build_apng.py をデスクトップのLINE_sticker_auto フォルダの中に入れる

---

## ③ Python 3.11 を入れる（重要）
※ 必ず「Python 3.11」をインストールしてください。
1. 次のページを開く  
   https://www.python.org/downloads/release/python-3119/
2. 「Windows installer (64-bit)」をクリック
3. インストール時に　✔ Add Python to PATHにチェックを入れる
4. Install Now を押す

---

## ④ 必要な部品を入れる
1. LINE_sticker_auto フォルダを開く
2. 右クリック →「PowerShellをここで開く」
3. pip install pillow　を入力してEnter：
成功と出ればOKです。

---

## ⑤ ffmpeg を入れる
1. https://www.gyan.dev/ffmpeg/builds/ を開く  
2. release builds をクリック  
3. zipをダウンロードして解凍  
4. 中にある「ffmpeg.exe」を  LINE_sticker_auto フォルダの中の
tools フォルダに入れる

---

# 🚀 使い方
1. videosフォルダにMP4動画を入れる  
2. PowerShellで次を入力してEnter：

python build_apng.py

3. 完成ファイルは、submitフォルダに作られます  

---

# 🎥 動画の条件（重要）
・形式は mp4  
・長さは 1～4秒がおすすめ  
・キャラクターが画面のフチに触れない  
・背景は #00FF00（明るい緑）  
・本体が透過されていないか確認
---

# ⚠ 透過処理の確認

submitフォルダの画像を開き、

✔ 体が消えていない  
✔ 緑が残っていない  

ことを確認してください。
