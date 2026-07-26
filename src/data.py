import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from config import TRAINING_PATH, TEST_PATH


# -------------------- HELPERS --------------------
# pandas read_csv wont work directly, because the text after the ; split in the csv also contains a ; sometimes.
# because of that i made this helper function, where the rows are split correctly (category ; text) in a list per row. those lists are saved in the splitted_csv list 
def split_csv(path: Path) -> list:
    splitted_csv = []
    with open(path, "r", encoding="utf-8") as f:
        for row in f:
            clean_row = row.strip()
            splitted_row = clean_row.split(";", 1)
            splitted_csv.append(splitted_row)

    return splitted_csv


# -------------------- LOAD DATA --------------------
# load data from the 10kgnad csv into a dataframe
def load_data(path: Path) -> pd.DataFrame:
    ordered_data = split_csv(path)
    df = pd.DataFrame(ordered_data, columns=["category", "text"])
    return df


# -------------------- GET SUBSET OF TRAINING-DATA --------------------
# creates a subset of the trainigdata dataframe (10%, 20%...) for the model training with a given percentage of the traininddata
# uses stratify to keep the category distribution
# uses a seed to ensure reproducibility
def get_data_subset(size: float, seed: int) -> pd.DataFrame:
    """
    
    """




# -------------------- INIT DF --------------------
train_df = load_data(TRAINING_PATH)
test_df = load_data(TEST_PATH)


# -------------------- CHECKS --------------------
print("--- TRAIN-DATA ---")
print(train_df)
print("\n--- TEST-DATA ---")
print(test_df)


# -------------------- STATS --------------------
print("\n-- TRAIN-DATA-CATEGORY-DIST --")
print(train_df["category"].value_counts())
print("\n-- TEST-DATA-CATEGORY-DIST --")
print(test_df["category"].value_counts())
print("\n-- TRAIN-DATA-ARTICLE-LENGTH-MEAN-(CHARACTERS) --")
print(round(train_df["text"].str.len().mean()))
print("\n-- TEST-DATA-ARTICLE-LENGTH-MEAN-(CHARACTERS) --")
print(round(test_df["text"].str.len().mean()))