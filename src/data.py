import pandas as pd
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
# load training data from CSV to DataFrame
def load_train() -> pd.DataFrame:
    correct_train = split_csv(TRAINING_PATH)
    train_df = pd.DataFrame(correct_train, columns=["category", "text"])
    return train_df

# load test data from CSV to DataFrame
def load_test() -> pd.DataFrame:
    correct_test = split_csv(TEST_PATH)
    test_df = pd.DataFrame(correct_test, columns=["category", "text"])
    return test_df


# -------------------- CHECKS --------------------
print("--- TRAIN-DATA ---")
print(load_train())
print("\n--- TEST-DATA ---")
print(load_test())