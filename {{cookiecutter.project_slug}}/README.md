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

## 参考文献

### 📖 書籍

- 浅野 純季, 木村 真也, 田中 冬馬, 武藤 克大, 栁 泉穂（2025）『先輩データサイエンティストからの指南書 ―実務で生き抜くためのエンジニアリングスキル』技術評論社. ISBN: 978-4-297-15100-3. [→ 公式ページ](https://gihyo.jp/book/2025/978-4-297-15100-3)

### 🍪 Cookiecutter

- [Python: Cookiecutter でプロジェクトのテンプレートを作る（ブログ）](https://blog.amedama.jp/entry/2016/01/09/111215)

### 🐳 Dev Container

- [Get Started with Dev Containers in VS Code（YouTube）](https://youtu.be/b1RavPr_878?si=3sIDOcyZxPuc4djl)
- [Dev Containersとは？Dockerを使った開発環境構築の決定版【図解で完全理解】（Zenn）](https://zenn.dev/yamato_snow/articles/fcb3cf8cf0ad03#discuss)
- [Dev Containerについて基礎を学習する（Qiita）](https://qiita.com/smr1/items/137b912de86c4947ead0)

### ✅ Pandera

- [Panderaをマスターしよう（1. 基本編）（Qiita）](https://qiita.com/GGravitons/items/981f439f687df0dc0be1)
- [Panderaの基本から応用まで（Zenn）](https://zenn.dev/zenn_tkc/articles/e07b34716237b6)
- [Pandera使ってみた（Pandasの型検証）（note）](https://note.com/yuuki_iwasaki/n/n4c15dcade0e9)

### ⚙️ Hydra × MLflow × Optuna

- [設定管理ツール Hydra で内部構造ごと書き換える（Zenn）](https://zenn.dev/gesonanko/articles/417d43669cf2af)
- [【機械学習】Hydraで「再現できない実験」とおさらば（Qiita）](https://qiita.com/shun_hobby/items/eecffd36b1fb827ae7f9)
- [実験管理が簡単に行えるmlflow trackingをローカル環境上で試してみた（Classmethod）](https://dev.classmethod.jp/articles/mlflow-tracking/)
- [MLflowで実験管理入門（フューチャー技術ブログ）](https://future-architect.github.io/articles/20200626/)
- [【13日目】MLflow で実験管理を始めよう（Zenn）](https://zenn.dev/churadata/articles/961bc10fd19ef6)
- [機械学習の煩雑なパラメーター管理の決定版 Hydra・MLflow・Optunaの組み合わせ（ログミー）](https://logmi.jp/main/technology/325087)

### 🖥️ Streamlit

- [Streamlitとは？Pythonで可視化Webアプリを作ろう（AI Academy Media）](https://aiacademy.jp/media/?p=6929)
- [案件先で学んだPythonのStreamlitスキルを公開してみた（note）](https://note.com/cograph_data/n/n167a5bffb6e1)
