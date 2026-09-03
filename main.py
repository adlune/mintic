import pandas as pd

DATASET_PATH = "data/IMDb_Top_700_Movies_2026.csv"


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
