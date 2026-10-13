import pandas as pd
from mintic import eda, ensemble, kmeans

DATASET_PATH = "data/"


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
