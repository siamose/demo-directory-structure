"""学習スクリプト（Hydra + MLflow + LightGBM）

WARNING: FEATURE_COLS / CAT_COLS / 評価指標をプロジェクトに合わせて変更してください。
         変更が必要な箇所は WARNING コメントでマークしています。

実行方法:
    uv run python scripts/train.py                   # 1回学習
    uv run python scripts/train.py --multirun        # Optuna 最適化（n_trials 回）

Hydra でパラメータを上書きする場合:
    uv run python scripts/train.py experiment.params.learning_rate=0.01
    uv run python scripts/train.py experiment.params.num_leaves=127
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

# scripts/ → プロジェクトルート → conf/
_CONF_DIR = str(Path(__file__).parents[1] / "conf")

import hydra
import hydra.utils
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score  # WARNING: タスクに応じて変更（回帰なら mean_squared_error 等）
from sklearn.model_selection import StratifiedKFold  # WARNING: 回帰タスクは KFold に変更

# ── 定数 ──────────────────────────────────────────────────────────────────────
# WARNING: 以下をプロジェクトのデータに合わせて変更してください

# 特徴量列（conf/config.yaml の target_col とは別に、モデル入力を定義する）
FEATURE_COLS: list[str] = [
    "feature_1",  # WARNING: 変更必須
    "feature_2",  # WARNING: 変更必須
]

# カテゴリ変数（LightGBM に categorical_feature として渡すカラム名）
# カテゴリ変数がない場合は空リスト [] にする
CAT_COLS: list[str] = []  # WARNING: カテゴリ変数があれば追加


# ── カテゴリカル処理 ──────────────────────────────────────────────────────────

def apply_categories(
    df: pd.DataFrame,
    categories: dict[str, list],
) -> pd.DataFrame:
    """保存済みのカテゴリ定義を DataFrame に適用する。

    学習時・推論時で同じコードが使われることを保証する。
    未知のカテゴリは NaN になる。
    """
    df = df.copy()
    for col, cats in categories.items():
        df[col] = pd.Categorical(df[col], categories=cats)
    return df


# ── 学習 ──────────────────────────────────────────────────────────────────────

def train_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    params: DictConfig,
    model_dir: Path,
) -> tuple[np.ndarray, list[float]]:
    """StratifiedKFold × LightGBM で学習し、各 fold のモデルを pickle 保存する。

    WARNING: 回帰タスクの場合は以下を変更してください
    - StratifiedKFold → KFold
    - objective: "binary" → "regression"
    - metric: "auc" → "rmse"
    - roc_auc_score → mean_squared_error 等

    Returns
    -------
    test_preds : np.ndarray
        全 fold の予測値を平均したもの
    oof_scores : list[float]
        各 fold のスコア
    """
    lgb_params = {
        # WARNING: objective と metric をタスクに合わせて変更してください
        # 二値分類: objective="binary",    metric="auc"
        # 多クラス: objective="multiclass", metric="multi_logloss"
        # 回帰:     objective="regression", metric="rmse"
        "objective": "binary",
        "metric": "auc",
        "learning_rate": params.learning_rate,
        "num_leaves": params.num_leaves,
        "min_child_samples": params.min_child_samples,
        "subsample": params.subsample,
        "colsample_bytree": params.colsample_bytree,
        "reg_alpha": params.reg_alpha,
        "reg_lambda": params.reg_lambda,
        "verbose": -1,
        "random_state": params.random_state,
    }

    skf = StratifiedKFold(
        n_splits=params.n_splits,
        shuffle=True,
        random_state=params.random_state,
    )
    test_preds = np.zeros(len(X_test))
    oof_scores: list[float] = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        cat_feature = CAT_COLS if CAT_COLS else "auto"
        model = lgb.train(
            lgb_params,
            lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_feature),
            num_boost_round=1000,
            valid_sets=[lgb.Dataset(X_val, label=y_val, categorical_feature=cat_feature)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=100),
            ],
        )

        # WARNING: 評価指標をタスクに応じて変更してください
        score = roc_auc_score(y_val, model.predict(X_val))
        oof_scores.append(score)
        test_preds += model.predict(X_test) / params.n_splits

        with open(model_dir / f"lgbm_fold{fold}.pkl", "wb") as f:
            pickle.dump(model, f)

        mlflow.log_metric(f"fold{fold}_score", score, step=fold)
        print(f"  Fold {fold}: score = {score:.4f}  (best_iter={model.best_iteration})")

    return test_preds, oof_scores


# ── エントリーポイント ────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path=_CONF_DIR, config_name="config")
def main(cfg: DictConfig) -> float:
    root = Path(hydra.utils.get_original_cwd())
    model_dir = root / "model"
    model_dir.mkdir(exist_ok=True)

    mlflow.set_tracking_uri((root / cfg.mlflow.tracking_uri).as_uri())
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    hydra_cfg = HydraConfig.get()
    is_multirun = hydra_cfg.mode.name == "MULTIRUN"
    run_name = (
        f"{cfg.experiment.name}_trial{hydra_cfg.job.num}"
        if is_multirun
        else cfg.experiment.name
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(dict(cfg.experiment.params))

        # ── 1. データ読み込み・バリデーション ─────────────────────────────
        # WARNING: スキーマ import のモジュール名を確認してください
        from {{cookiecutter.project_slug}}.schema import InferenceSchema, TrainSchema

        target_col = cfg.data.target_col
        train_cols = ["id"] + FEATURE_COLS + [target_col]
        test_cols = ["id"] + FEATURE_COLS

        train = pd.read_csv(
            root / cfg.data.input_path / "train.csv", encoding="utf-8-sig"
        )[train_cols]
        TrainSchema.validate(train)

        test = pd.read_csv(
            root / cfg.data.input_path / "test.csv", encoding="utf-8-sig"
        )[test_cols]
        InferenceSchema.validate(test[FEATURE_COLS])

        print(f"train: {train.shape}, test: {test.shape}")

        # ── 2. カテゴリ定義を作成して保存 ─────────────────────────────────
        categories: dict[str, list] = {}
        for col in CAT_COLS:
            train[col] = train[col].astype("category")
            categories[col] = train[col].cat.categories.tolist()

        categories_path = model_dir / "categories.json"
        with open(categories_path, "w", encoding="utf-8") as f:
            json.dump(categories, f, ensure_ascii=False, indent=2)
        print(f"categories.json saved → {categories_path}")

        if CAT_COLS:
            test = apply_categories(test, categories)

        # ── 3. 学習 ────────────────────────────────────────────────────────
        X = train[FEATURE_COLS]
        y = train[target_col]
        X_test = test[FEATURE_COLS]

        print(f"\nTraining ({cfg.experiment.params.n_splits}-fold CV)...")
        test_preds, oof_scores = train_cv(X, y, X_test, cfg.experiment.params, model_dir)

        mean_score = float(np.mean(oof_scores))
        std_score = float(np.std(oof_scores))
        mlflow.log_metric("oof_score_mean", mean_score)
        mlflow.log_metric("oof_score_std", std_score)
        print(f"\nOOF score: {mean_score:.4f} ± {std_score:.4f}")

        # ── 4. submission 保存 ────────────────────────────────────────────
        sub_path = root / cfg.data.output_path / "submission.csv"
        sub_path.parent.mkdir(parents=True, exist_ok=True)

        submission = test[["id"]].copy()
        submission[target_col] = test_preds
        submission.to_csv(sub_path, index=False)

        mlflow.log_artifact(str(sub_path))
        print(f"Submission saved → {sub_path}")

    return mean_score


if __name__ == "__main__":
    main()
