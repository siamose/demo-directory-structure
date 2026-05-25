# プロジェクト利用ガイド

このテンプレートを使って新しい PoC を立ち上げるときの手順を時系列でまとめます。

---

## 1. プロジェクト生成

```bash
cookiecutter gh:your-username/00_demo_directory_structure
# → project_name / description / streamlit_app_url などを入力
#    streamlit_app_url はデプロイ後に記入するため空欄で OK
```

## 2. 依存関係のインストール

```bash
cd {{cookiecutter.project_slug}}
uv sync
```

## 3. データの配置

- 使用するデータセットを `data/raw/` に配置する
- 最低限 `train.csv` と `test.csv` が必要
- `sample_submission.csv` があれば submission 保存に使用
- データソースの URL や取得方法を `README.md` に記載しておく

## 4. スキーマの定義（WARNING）

`src/{{cookiecutter.project_slug}}/schema/__init__.py` を開き、以下を変更する：

- [ ] `FeatureSchema` のフィールド名をデータのカラム名に合わせる
- [ ] 型（`Series[int]`, `Series[float]`, `Series[str]`）を確認する
- [ ] 値域（`ge`, `le`, `isin`）をデータの実測値から設定する
- [ ] `TrainSchema` の目的変数カラム名・型を変更する

## 5. 設定の更新（WARNING）

`conf/config.yaml` を開き、以下を変更する：

- [ ] `data.target_col` を目的変数のカラム名に変更する
- [ ] `hydra.sweeper.direction` を `maximize` または `minimize` に設定する
- [ ] `hydra.sweeper.n_trials` を適切な回数に設定する（初回は 5 のまま可）

`conf/experiment/default.yaml` を開き、必要に応じてパラメータ初期値を調整する。

## 6. 学習スクリプトの調整（WARNING）

`scripts/train.py` を開き、以下を変更する：

- [ ] `FEATURE_COLS` をモデルの入力特徴量に合わせる
- [ ] `CAT_COLS` をカテゴリ変数に合わせる（ない場合は空リスト）
- [ ] `lgb_params` の `objective` / `metric` をタスクに合わせる
  - 二値分類: `"binary"` / `"auc"`
  - 多クラス:  `"multiclass"` / `"multi_logloss"`
  - 回帰:      `"regression"` / `"rmse"`
- [ ] 評価指標（`roc_auc_score` 等）をタスクに合わせる

## 7. 1回学習・動作確認

```bash
uv run python scripts/train.py
# mlruns/ に実験ログが記録される
# model/ に lgbm_fold{n}.pkl と categories.json が保存される
```

## 8. ハイパーパラメータ最適化（任意）

```bash
# WARNING: --multirun を付けると conf/config.yaml の n_trials 回学習が走る
#          direction の設定を必ず確認してから実行すること
uv run python scripts/train.py --multirun
```

## 9. Streamlit UI の調整（WARNING）

`src/{{cookiecutter.project_slug}}/app/main.py` を開き、以下を変更する：

- [ ] `FEATURE_COLS` をモデルの入力特徴量に合わせる（train.py と一致させる）
- [ ] `CAT_COLS` をカテゴリ変数に合わせる
- [ ] `render_inputs()` のウィジェットを変更する
  - `st.selectbox` : カテゴリ変数
  - `st.slider`    : 数値変数（連続値）
  - `st.number_input` : 数値変数（精度が必要な場合）
- [ ] `init_session_state()` のデフォルト値を変更する
- [ ] `copy_a_to_b()` のキー名を変更する
- [ ] `gauge_chart()` は二値分類の確率表示用。回帰タスクでは別グラフを検討する

## 10. ローカル動作確認

```bash
uv run streamlit run src/{{cookiecutter.project_slug}}/app/main.py
```

## 11. Streamlit Cloud デプロイ前の準備

- [ ] ベスト run のモデルファイル（`lgbm_fold{1..n}.pkl`）を `model/demo/` にコピーする
- [ ] `model/demo/categories.json` をコピーする（カテゴリ変数がある場合）
- [ ] `data/processed/demo/submission.csv` をコピーする（提出スコア確認ページがある場合）
- [ ] 上記ファイルをコミット & プッシュする
  - `model/demo/` と `data/processed/demo/` は `.gitignore` の例外として管理済み

## 12. Streamlit Community Cloud でデプロイ

1. [share.streamlit.io](https://share.streamlit.io) でリポジトリを接続する
2. Main file path: `src/{{cookiecutter.project_slug}}/app/main.py`
3. デプロイ後の URL を `README.md` の デモセクションに記入する

```markdown
<!-- README.md の デモ セクション -->
👉 [Streamlit アプリを開く](https://your-app-url.streamlit.app/)
```

## 13. モデル更新時の手順

再学習でモデルを更新した場合は、デモ用ファイルも更新が必要。

- [ ] 再学習 → ベスト run を選定（MLflow の実験管理ページで確認）
- [ ] `model/demo/` を上書きコピー
- [ ] `data/processed/demo/submission.csv` を上書きコピー（必要な場合）
- [ ] コミット & プッシュ
- [ ] Streamlit Cloud 管理画面で **Reboot app** を実行

> **注意**: ローカルの `model/` と `model/demo/` は別管理です。
> 学習を回すと `model/` は自動更新されますが、`model/demo/` は**手動コピー**が必要です。
