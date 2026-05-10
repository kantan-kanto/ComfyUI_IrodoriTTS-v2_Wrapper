# ComfyUI_IrodoriTTS-v2_Wrapper

このリポジトリは [ComfyUI_IrodoriTTS_Wrapper](https://github.com/jupo-ai/ComfyUI_IrodoriTTS_Wrappe) の非公式 v2 専用 fork です。
Irodori-TTS-500M-v2 用に変更したもので、元プロジェクトの公式版ではありません。
upstream が公式に v2 対応した場合、この fork は archive する可能性があります。

[IrodoriTTS-v2](https://github.com/Aratako/Irodori-TTS)のComfyUI用カスタムノードです。
v2用のcodecは `Aratako/Semantic-DACVAE-Japanese-32dim` 固定です。
モデル本体の自動ダウンロードは行いません。
codec weights と tokenizer は、未配置・未キャッシュの場合に Hugging Face から自動取得されます。

## install
1. custom_nodesフォルダ上でgit clone
2. 仮想環境を有効化した上で `pip install -r ComfyUI_IrodoriTTS-v2_Wrapper/requirements.txt`
3. checkpointsフォルダに[IrodoriTTS-v2モデル](https://huggingface.co/Aratako/Irodori-TTS-500M-v2/blob/main/model.safetensors)を手動で配置する
   - 配置例: `ComfyUI/models/checkpoints/Irodori-TTS-500M-v2.safetensors`
4. codec weights と tokenizer は初回ロード時に自動取得されます
   - codec: [`Aratako/Semantic-DACVAE-Japanese-32dim`](https://huggingface.co/Aratako/Semantic-DACVAE-Japanese-32dim) の `weights.pth`、約410MB
   - tokenizer: [`llm-jp/llm-jp-3-150m`](https://huggingface.co/llm-jp/llm-jp-3-150m)、約6MB
   - codec を手動配置したい場合は、`ComfyUI/models/checkpoints/Semantic-DACVAE-Japanese-32dim-weights.pth` または `ComfyUI/models/vae/Semantic-DACVAE-Japanese-32dim-weights.pth` に置くと自動取得を避けられます


## ノード一覧
- IrodoriTTS-v2 Model Loader
  - モデルを読み込みます

- IrodoriTTS-v2Sampler
  - テキストの入力と実際の生成を行います

- IrodoriTTS-v2 Referenec Audio
- IrodoriTTS-v2 Advanced CFG
- IrodoriTTS-v2 Rescale Config
  - オプション設定用のカスタムノードです

- IrodoriTTS-v2 Emoji Selector
  - IrodoriTTS-v2で使用できる絵文字一覧です。
  - ボタンクリックで、クリップボードに絵文字をコピーします

## ライセンスと謝辞

この fork は元リポジトリ [ComfyUI_IrodoriTTS_Wrapper](https://github.com/jupo-ai/ComfyUI_IrodoriTTS_Wrappe) を基にしています。
元リポジトリの著作権表示および MIT License は `LICENSE` に保持しています。

Original project:
- Repository: https://github.com/jupo-ai/ComfyUI_IrodoriTTS_Wrappe
- Copyright (c) 2026 jupo-ai
- License: MIT License

This fork:
- Repository: https://github.com/kantan-kanto/ComfyUI_IrodoriTTS-v2_Wrapper
- Modifications Copyright (c) 2026 kantan-kanto
- License: MIT License
