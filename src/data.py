"""
loads the 10kGNAD csv-files into dataframes. creates subsets of the dataframes for the model runs (10%, 20%... of training-data). no preprocessing.
"""
import pandas as pd
from pathlib import Path
from config import TRAINING_PATH, TEST_PATH


# -------------------- HELPERS -------------------- 
def split_csv(path: Path) -> list:
    """
    pandas read_csv wont work directly because the text after the ; split in the csv-file also contains a ; sometimes. 
    because of that i made this helper function, where the rows are split correctly (category ; text) in a list per row. those lists are saved in the splitted_csv list
    """
    splitted_csv = []
    with open(path, "r", encoding="utf-8") as f:
        for row in f:
            clean_row = row.strip()
            splitted_row = clean_row.split(";", 1)
            splitted_csv.append(splitted_row)

    return splitted_csv


# -------------------- LOAD DATA --------------------
def load_data(path: Path) -> pd.DataFrame:
    """
    load data from the 10kgnad csv-file into a dataframe
    """
    ordered_data = split_csv(path)
    df = pd.DataFrame(ordered_data, columns=["category", "text"])
    return df


# -------------------- GET SUBSET OF TRAINING-DATA --------------------
def get_data_subset(df: pd.DataFrame, size: float, seed: int) -> pd.DataFrame:
    """
    creates a subset from the given dataframe. uses stratify so the distribution of the category stays the same to keep clean data.
    runs new everytime so the 10% subset is not included in the 20% subset.
    
    Args:
        df: dataframe from which to create the subset
        size: a float which determines the size of the subset (0.1 = 10%, 1 = 100%)
        seed: a int, which is set in the config.py, so it is reproduceable

    Returns:
        DataFrame: a subset of a given % from the trainig-data dataframe
    """
    subset = df.groupby("category", group_keys=False).sample(frac=size, random_state=seed)
    return subset


# -------------------- CHECKS --------------------
if __name__ == "__main__":
    train_df = load_data(TRAINING_PATH)
    test_df = load_data(TEST_PATH)

    print("--- TRAIN-DATA ---")
    print(train_df)
    print("\n--- TEST-DATA ---")
    print(test_df)

    print("\n-- TRAIN-DATA-CATEGORY-DIST --")
    print(train_df["category"].value_counts())
    print("\n-- TEST-DATA-CATEGORY-DIST --")
    print(test_df["category"].value_counts())

    print("\n-- ARTICLE-LENGTH-MEAN (CHARACTERS) --")
    print("train:", round(train_df["text"].str.len().mean()))
    print("test: ", round(test_df["text"].str.len().mean()))

    train_subset = get_data_subset(train_df, 0.1, 1)
    print("\n--- TRAIN-DATA-SUBSET (10%) ---")
    print(train_subset.shape)
    print(train_subset["category"].value_counts())