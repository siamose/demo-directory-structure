import pandas as pd
from pandera.typing import DataFrame

from poc.schema import InputSchema


@InputSchema.check_types
def load_and_validate(path: str) -> DataFrame[InputSchema]:
    df = pd.read_csv(path)
    return df
