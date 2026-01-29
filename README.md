LINE アニメーションスタンプ作成ツール（APNG自動生成）
このツールは、MP4動画からLINEアニメーションスタンプ用APNGを自動生成する
Windows向けのPythonスクリプトです。

できること
・MP4 → 透過PNG → APNG を自動作成
・背景は透過、本体は透過させない処理
・LINE審査仕様（秒数・サイズ・容量）に自動対応
・main.png / tab.png / upload.zip を自動生成

動作環境
・Windows 10 / 11
・Python 3.10 以上
・ffmpeg / ffprobe（PATH設定済み）
・ライブラリ：Pillow

事前準備（最初に一度だけ）
１．Python をインストール
    参照：	https://www.python.org/
２．ffmpeg をインストールし PATH を通す
    参照：	https://qiita.com/Tadataka_Takahashi/items/9dcb0cf308db6f5dc31b
３．Pillow をインストール
    参照：	https://ai-kenkyujo.com/programming/language/python/pil-install/

使い方（超重要）
１．LINE_sticker_auto.zip をダウンロード
２．右クリックして「すべて展開」
３．展開したフォルダを開く
４．videos フォルダに MP4動画 を入れる（main,tab作成用動画を1.mv4とし最初に入れる）
５．PowerShellでフォルダを開き、下記のプロンプトをコピペ

cd LINE_sticker_auto
python build_apng.py


出力先
    dist/apng/        ← スタンプ用APNG
    submit/batch\_xxxx/
    ├ upload.zip    ← LINE Creators Market提出用
    ├ main.png
    └ tab.png

注意事項（必ず読んでください）
    本ツールは 個人利用・学習目的 に限り自由に使用できます
    商用利用・再配布・改変後の再配布は禁止です
    本ツールの使用によるトラブル・損害について作者は責任を負いません

免責
    本ツールは無保証で提供されます。
    使用はすべて自己責任でお願いします。ｍｐ４からlineスタンプ用apngを自動作成
