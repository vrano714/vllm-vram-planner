# vLLM Configurator & VRAM Planner

vLLMのGPUメモリ管理（モデル重み、CUDA Graph、KVキャッシュ、マージン）と各種オプション（Tensor Parallel, enforce-eager, kv-cache-dtype, prefix-caching等）をシミュレーションし、サービングコマンドを自動生成するWebツールです。

Try here -> https://vllm-vram-planner.vrano.net/

---

## 主な機能

- `--gpu-memory-utilization` × VRAM容量 × 枚数で割当の総容量を指定可能
- Hugging FaceのモデルURL・モデルIDで対象モデルを指定して、自動で必要容量を計算
- コンテキスト長とKVキャッシュ形式を指定して自動で必要容量を計算

---

## 開発 & ローカル実行

```bash
# 依存関係のインストール
npm install

# 開発サーバーの起動 (デフォルト: http://localhost:3000)
npm run dev

# プロダクションビルド
npm run build

# ビルド成果物のプレビュー
npm run preview
```

---

## デプロイ（静的ホスティング）

完全なSPA（静的HTML/JS/CSS）としてビルドされるため、サーバーサイドのバックエンドなしに任意の静的ホスティング環境でそのまま配信可能です。

- **Vercel / Netlify / Cloudflare Pages**:
  - Build command: `npm run build`
  - Output directory: `dist`
- **GitHub Pages**:
  - `dist` ディレクトリの内容を配信。

