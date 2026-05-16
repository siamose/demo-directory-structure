# [Project Name]

> POC development environment with Dev Container, Pandera, Hydra, MLflow, and Streamlit.

## Stack

| Tool | 役割 |
|---|---|
| [Pandera](https://pandera.readthedocs.io/) | 入力データのスキーマバリデーション |
| [Hydra](https://hydra.cc/) | 設定管理・マルチラン実験 |
| [MLflow](https://mlflow.org/) | 実験管理・モデルアーティファクト保存 |
| [Streamlit](https://streamlit.io/) | POCダッシュボード・ステークホルダー向け可視化 |

## Getting Started

### Dev Container（推奨）

1. VS Code に [Dev Containers 拡張](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) をインストール
2. このリポジトリを開き `Reopen in Container` を実行
3. `uv sync` が自動実行され依存関係がインストールされる

### ローカル環境

```bash
pip install uv
uv sync
```

## Usage

### 実験を実行する

```bash
# 単一実行
uv run python scripts/run_train.py

# マルチラン（ハイパーパラメータスイープ）
uv run python scripts/run_train.py --multirun experiment.params.learning_rate=0.001,0.01,0.1
```

### MLflow UI を起動する

```bash
uv run mlflow ui --port 5000
# → http://localhost:5000
```

### Streamlit アプリを起動する

```bash
uv run streamlit run src/poc/app/main.py
# → http://localhost:8501
```

### Docker Compose で一括起動する

```bash
cp .env.example .env
docker compose up
```

## Project Structure

```
.
├── .devcontainer/   # Dev Container設定（開発用）
├── .streamlit/      # Streamlit設定
├── conf/            # Hydra設定ファイル
├── data/            # データ（gitignore）
├── model/           # シリアライズ済みモデル（ローカルキャッシュ）
├── notebooks/       # 探索的分析用
├── outputs/         # Hydra単一実行の出力（gitignore）
├── multirun/        # Hydraマルチランの出力（gitignore）
├── mlruns/          # MLflowトラッキング（gitignore）
├── scripts/         # 実行エントリポイント
├── src/poc/         # メインパッケージ
│   ├── schema/      # Panderaスキーマ
│   ├── pipeline/    # データロード・前処理
│   ├── models/      # 学習ロジック
│   └── app/         # Streamlitアプリ
├── tests/
├── Dockerfile       # 本番・共有用イメージ
└── compose.yml      # MLflow + Streamlit 一括起動
```

## Environment Variables

`.env.example` をコピーして `.env` を作成してください。

```bash
cp .env.example .env
```
