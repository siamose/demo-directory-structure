import pandera as pa
from pandera.typing import Series


class InputSchema(pa.DataFrameModel):
    """Pandera schema for validating input data.

    Customize field definitions to match your actual data contract.
    """

    # Example fields — replace with your actual columns
    # id: Series[int] = pa.Field(gt=0, description="Unique identifier")
    # value: Series[float] = pa.Field(nullable=False)
    # category: Series[str] = pa.Field(isin=["A", "B", "C"])

    class Config:
        strict = True  # reject unknown columns
        coerce = True  # auto-cast types where possible
