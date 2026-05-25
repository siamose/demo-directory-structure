# {{cookiecutter.project_name}}

{{cookiecutter.description}}

## 🔮 デモ

<!-- WARNING: Streamlit Community Cloud へのデプロイ後に URL を記入してください -->
<!-- 👉 [Streamlit アプリを開く]({{cookiecutter.streamlit_app_url}}) -->

## Stack

| Tool | Role | Design Notes |
|---|---|---|
| [LightGBM](https://lightgbm.readthedocs.io/) | 分類・回帰モデル | |
| [Pandera](https://pandera.readthedocs.io/) | 入力データスキーマ検証 | |
| [Hydra](https://hydra.cc/) | 設定管理 | |
| [MLflow](https://mlflow.org/) | 実験管理 | |
| [Optuna](https://optuna.org/) | ハイパーパラメータ最適化 | |
| [Streamlit](https://streamlit.io/) | インタラクティブ UI | |
| [uv](https://docs.astral.sh/uv/) | パッケージ管理 | |

## Prerequisites

Python 3.12 と [uv](https://docs.astral.sh/uv/getting-started/installation/) が必要です。

```bash
uv sync
```

## Getting Started

```bash
# 0. データを取得して data/raw/ に配置
#    WARNING: データソース URL と必要なファイル名をここに記載してください

# 1. 学習（1回）
uv run python scripts/train.py

# 2. ハイパーパラメータ最適化（Optuna）
#    WARNING: conf/config.yaml の hydra.sweeper.direction を確認してから実行
#    WARNING: --multirun を付けると n_trials 回学習が走る
uv run python scripts/train.py --multirun

# 3. アプリ起動
uv run streamlit run src/{{cookiecutter.project_slug}}/app/main.py
```

詳細な手順は [プロジェクト利用ガイド](docs/project_workflow.md) を参照。

## Directory Structure

```
.
├── conf/                    # Hydra 設定ファイル
│   ├── config.yaml          # メイン設定（データパス・MLflow・Optuna）
│   └── experiment/          # 実験プリセット
├── data/
│   ├── raw/                 # 生データ（.gitignore 対象）
│   └── processed/
│       └── demo/            # Streamlit Cloud 用デモデータ（コミット対象）
├── docs/                    # 実装記録・ガイド
├── model/
│   └── demo/                # Streamlit Cloud 用デモモデル（コミット対象）
├── notebooks/               # EDA・可視化ノートブック
├── scripts/
│   └── train.py             # 学習エントリポイント（前処理を内包）
└── src/{{cookiecutter.project_slug}}/
    ├── app/
    │   └── main.py          # Streamlit アプリ（A/B 比較シミュレーター）
    └── schema/              # Pandera スキーマ定義
```

## Data Attribution

<!-- WARNING: 使用データセットのライセンス・帰属情報を記載してください -->
