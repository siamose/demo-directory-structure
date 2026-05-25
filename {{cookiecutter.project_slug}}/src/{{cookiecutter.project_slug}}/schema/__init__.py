"""Pandera スキーマ定義

WARNING: フィールド名・型・値域はプロジェクトのデータに合わせて必ず変更してください。
         TrainSchema と InferenceSchema は最低限定義が必要です。
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.typing import Series


class FeatureSchema(pa.DataFrameModel):
    """モデルで使う特徴量の共通スキーマ（基底クラス）。

    WARNING: 以下のフィールドをプロジェクトのデータに合わせて変更してください。
    - フィールド名をデータのカラム名に揃える
    - 型（Series[int], Series[float], Series[str]）を確認する
    - 値域（ge, le, isin）をデータの実測値から設定する

    使用可能なフィールド定義の例::

        # 数値特徴量
        age: Series[int] = pa.Field(ge=0, le=120)
        score: Series[float] = pa.Field(ge=0.0, le=1.0)

        # カテゴリ特徴量
        category: Series[str] = pa.Field(isin=["A", "B", "C"])

        # 必須でない（nullable）フィールド
        optional_value: Series[float] = pa.Field(nullable=True)
    """

    # WARNING: 以下をプロジェクトの特徴量に差し替えてください
    feature_1: Series[float] = pa.Field(description="数値特徴量の例")
    feature_2: Series[str] = pa.Field(
        isin=["A", "B", "C"],
        description="カテゴリ特徴量の例",
    )

    class Config:
        strict = True   # 未知のカラムを拒否
        coerce = True   # 型を自動キャスト


class TrainSchema(FeatureSchema):
    """学習データ用スキーマ。

    FeatureSchema に id と目的変数を追加する。

    WARNING: target_col をプロジェクトの目的変数名に変更してください。
    """

    id: Series[int] = pa.Field(ge=0, description="行識別子")
    # WARNING: カラム名と型をプロジェクトの目的変数に合わせて変更してください
    # 二値分類: Series[int] = pa.Field(isin=[0, 1])
    # 回帰:     Series[float] = pa.Field(ge=0.0)
    target: Series[int] = pa.Field(isin=[0, 1], description="目的変数")

    class Config(FeatureSchema.Config):
        strict = True
        coerce = True


class InferenceSchema(FeatureSchema):
    """推論入力用スキーマ。

    Streamlit のシミュレーターがモデルに渡す DataFrame を検証する。
    id・目的変数は含まない。特徴量のみ。
    """

    class Config(FeatureSchema.Config):
        strict = True
        coerce = True
