from pathlib import Path


# -------------------- CONFIG --------------------
# general configuration for all modules
# four groups: PATHS, EXPERIMENT DESIGN, CLASSIC, BERT
# PATHS : where training- and testdata is located, where to put results and graphs
# EXPERIMENT DESIGN: which % to run, how many repetitions, seed control
# CLASSIC: config for the classic model (log. reg.)
# BERT: config for the Bert model




# -------------------- PATHS --------------------
BASE_DIR = Path(__file__).resolve().parent.parent # set the bachelor directory as base dir. path is based from the location of this file
TRAINING_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
TRAIN_LEMMAS_PATH = PROCESSED_DIR / "train_lemmas.parquet"
TEST_LEMMAS_PATH = PROCESSED_DIR / "test_lemmas.parquet"




# -------------------- EXPERIMENT DESIGN --------------------




# -------------------- CLASSIC --------------------




# -------------------- BERT --------------------



