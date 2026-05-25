"""Streamlit アプリ — A/B シミュレーター & 実験管理

WARNING: FEATURE_COLS / CAT_COLS / render_inputs() のウィジェットを
         プロジェクトのデータ・タスクに合わせて変更してください。

起動方法:
    uv run streamlit run src/{{cookiecutter.project_slug}}/app/main.py

事前に以下を実行しておくこと:
    uv run python scripts/train.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── パス ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parents[3]   # src/pkg/app/main.py → 3階層上 = プロジェクトルート
MODEL_DIR = ROOT / "model"         # ローカル学習済みモデル（gitignore 対象）
DEMO_DIR  = ROOT / "model" / "demo"  # デモ用モデル（git にコミット済み）

# ── 定数 ─────────────────────────────────────────────────────────────────────
# WARNING: 以下をプロジェクトの特徴量・カテゴリに合わせて変更してください

FEATURE_COLS: list[str] = [
    "feature_1",  # WARNING: 変更必須
    "feature_2",  # WARNING: 変更必須
]

# カテゴリ変数（categories.json から読み込む対象）
CAT_COLS: list[str] = []  # WARNING: カテゴリ変数があれば追加


# ── モデル・カテゴリ読み込み ───────────────────────────────────────────────────

@st.cache_resource
def load_resources() -> tuple[list | None, dict | None]:
    """学習済みモデルと categories.json を読み込む。

    優先順位：
      1. model/          ローカルで学習した最新モデル（gitignore 対象）
      2. model/demo/     git にコミットされたデモ用モデル（Streamlit Cloud 用）
    どちらも存在しない場合は (None, None) を返す。
    """
    n_folds = 4  # WARNING: 学習時の n_splits に合わせて変更
    fold_paths = [MODEL_DIR / f"lgbm_fold{i}.pkl" for i in range(1, n_folds + 1)]
    cat_path = MODEL_DIR / "categories.json"

    if not all(p.exists() for p in fold_paths) or not cat_path.exists():
        fold_paths = [DEMO_DIR / f"lgbm_fold{i}.pkl" for i in range(1, n_folds + 1)]
        cat_path = DEMO_DIR / "categories.json"

    if not all(p.exists() for p in fold_paths) or not cat_path.exists():
        return None, None

    models = []
    for p in fold_paths:
        with open(p, "rb") as f:
            models.append(pickle.load(f))

    with open(cat_path, encoding="utf-8") as f:
        categories = json.load(f)

    return models, categories


# ── 推論 ─────────────────────────────────────────────────────────────────────

@st.cache_data
def predict(inputs_json: str) -> float:
    """特徴量の JSON 文字列を受け取り、4 fold 平均の予測値を返す。

    WARNING: 二値分類以外のタスクでは戻り値の解釈を変更してください。
    """
    models, categories = load_resources()
    inputs = json.loads(inputs_json)
    df = pd.DataFrame([inputs])
    for col, cats in (categories or {}).items():
        df[col] = pd.Categorical(df[col], categories=cats)
    X = df[FEATURE_COLS]
    return float(np.mean([m.predict(X)[0] for m in models]))


# ── UI パーツ ─────────────────────────────────────────────────────────────────

@st.cache_data
def gauge_chart(value: float, title: str) -> go.Figure:
    """予測確率をゲージグラフで表示する（二値分類用）。

    WARNING: 回帰タスクなど 0〜1 範囲外の出力には不向きです。
             タスクに応じて go.Bar や go.Scatter に変更してください。
    """
    if value < 0.3:
        bar_color = "#2ecc71"
    elif value < 0.7:
        bar_color = "#f39c12"
    else:
        bar_color = "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"valueformat": ".3f", "font": {"size": 36}},
        title={"text": title, "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 1], "tickformat": ".1f"},
            "bar": {"color": bar_color, "thickness": 0.3},
            "steps": [
                {"range": [0.0, 0.3], "color": "#d5f5e3"},
                {"range": [0.3, 0.7], "color": "#fef9e7"},
                {"range": [0.7, 1.0], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "gray", "width": 2},
                "thickness": 0.75,
                "value": 0.5,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=60, b=10))
    return fig


def render_inputs(scenario: str, categories: dict) -> dict:
    """1シナリオ分の入力ウィジェットを描画し、入力値を dict で返す。

    WARNING: ウィジェットとキー名をプロジェクトの特徴量に合わせて変更してください。
    - st.selectbox : カテゴリ変数
    - st.slider    : 数値変数（連続値）
    - st.number_input : 数値変数（精度が必要な場合）
    """
    s = f"_{scenario}"

    # WARNING: 以下のウィジェットをプロジェクトの特徴量に差し替えてください ──────
    feature_2 = st.selectbox(
        "feature_2（カテゴリ）",
        options=categories.get("feature_2", ["A", "B", "C"]),
        key=f"feature_2{s}",
    )
    feature_1 = st.slider(
        "feature_1（数値）",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        key=f"feature_1{s}",
    )
    # ────────────────────────────────────────────────────────────────────────

    return {
        "feature_1": feature_1,
        "feature_2": feature_2,
    }


def init_session_state(categories: dict) -> None:
    """セッション変数のデフォルト値を設定する（初回のみ）。

    WARNING: キー名と初期値をプロジェクトの特徴量に合わせて変更してください。
    """
    defaults: dict = {
        "feature_1": 0.5,
        "feature_2": categories.get("feature_2", ["A"])[0],
    }
    for scenario in ("a", "b"):
        for key, val in defaults.items():
            state_key = f"{key}_{scenario}"
            if state_key not in st.session_state:
                st.session_state[state_key] = val


def copy_a_to_b() -> None:
    """シナリオ A の入力値をシナリオ B にコピーする。"""
    # WARNING: キー名をプロジェクトの特徴量に合わせて変更してください
    keys = ["feature_1", "feature_2"]
    for key in keys:
        st.session_state[f"{key}_b"] = st.session_state[f"{key}_a"]


# ── ページ：シミュレーター ────────────────────────────────────────────────────

def page_simulator() -> None:
    st.title("🔮 シミュレーター")
    # WARNING: キャプションをプロジェクトの内容に合わせて変更してください
    st.caption("特徴量を指定して予測値をリアルタイム確認できます。")

    models, categories = load_resources()

    if models is None:
        st.warning(
            "モデルが見つかりません。先に以下を実行してください。\n\n"
            "```\nuv run python scripts/train.py\n```"
        )
        return

    init_session_state(categories or {})

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("シナリオ A")
        inputs_a = render_inputs("a", categories or {})

    with col_b:
        sub_col, btn_col = st.columns([3, 1])
        sub_col.subheader("シナリオ B")
        if btn_col.button("← A をコピー", use_container_width=True):
            copy_a_to_b()
            st.rerun()
        inputs_b = render_inputs("b", categories or {})

    st.divider()

    pred_a = predict(json.dumps(inputs_a, ensure_ascii=False))
    pred_b = predict(json.dumps(inputs_b, ensure_ascii=False))
    diff = pred_b - pred_a

    g_col_a, g_col_b, g_col_diff = st.columns([2, 2, 1])

    with g_col_a:
        st.plotly_chart(gauge_chart(pred_a, "シナリオ A"), use_container_width=True)

    with g_col_b:
        st.plotly_chart(gauge_chart(pred_b, "シナリオ B"), use_container_width=True)

    with g_col_diff:
        st.write("")
        st.write("")
        st.metric(label="差 (B − A)", value=f"{diff:+.3f}", delta=None)
        label = "B の方が高い" if diff > 0 else "A の方が高い"
        st.caption(label)


# ── ページ：実験管理 ──────────────────────────────────────────────────────────

def page_experiments() -> None:
    st.title("📊 実験管理")

    st.subheader("MLflow 実験結果")
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())

    try:
        runs = mlflow.search_runs(search_all_experiments=True)
        if runs.empty:
            st.info("`uv run python scripts/train.py` を実行するとここに結果が表示されます。")
        else:
            metric_cols = [c for c in runs.columns if c.startswith("metrics.")]
            param_cols = [c for c in runs.columns if c.startswith("params.")]
            show_cols = (
                ["tags.mlflow.runName", "status", "start_time"]
                + metric_cols
                + param_cols
            )
            show_cols = [c for c in show_cols if c in runs.columns]
            st.dataframe(
                runs[show_cols].sort_values("start_time", ascending=False),
                use_container_width=True,
            )

            score_col = "metrics.oof_score_mean"
            if score_col in runs.columns and runs[score_col].notna().any():
                st.subheader("OOF スコア推移")
                runs_sorted = runs.dropna(subset=[score_col]).sort_values("start_time")
                fig = go.Figure(go.Scatter(
                    x=runs_sorted["start_time"],
                    y=runs_sorted[score_col],
                    mode="lines+markers",
                    hovertemplate="Score: %{y:.4f}<extra></extra>",
                ))
                fig.update_layout(
                    xaxis_title="実行日時", yaxis_title="OOF Score",
                    height=320, paper_bgcolor="white", plot_bgcolor="white",
                )
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"MLflow の読み込みに失敗しました: {e}")

    st.subheader("Submission スコア分布")
    sub_path = ROOT / "data" / "processed" / "submission.csv"
    if not sub_path.exists():
        sub_path = ROOT / "data" / "processed" / "demo" / "submission.csv"

    if sub_path.exists():
        sub = pd.read_csv(sub_path)
        c1, c2, c3 = st.columns(3)
        c1.metric("件数", f"{len(sub):,}")
        c2.metric("平均スコア", f"{sub.iloc[:, 1].mean():.4f}")
        c3.metric("中央値", f"{sub.iloc[:, 1].median():.4f}")

        fig2 = go.Figure(go.Histogram(
            x=sub.iloc[:, 1], nbinsx=100,
            marker_color="#3366cc",
        ))
        fig2.update_layout(
            xaxis_title="予測値", yaxis_title="件数",
            height=320, paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("`uv run python scripts/train.py` を実行すると submission.csv が生成されます。")


# ── メイン ────────────────────────────────────────────────────────────────────

st.set_page_config(
    # WARNING: ページタイトルをプロジェクト名に変更してください
    page_title="{{cookiecutter.project_name}}",
    page_icon="🔮",
    layout="wide",
)

page = st.sidebar.radio(
    "ページ",
    ["🔮 シミュレーター", "📊 実験管理"],
    label_visibility="collapsed",
)

if page == "🔮 シミュレーター":
    page_simulator()
else:
    page_experiments()
