"""
general configuration for all modules, no logic just values four groups: PATHS, EXPERIMENT DESIGN, CLASSIC, BERT
PATHS : where training- and testdata is located, where to put results and graphs
EXPERIMENT DESIGN: which % to run, how many repetitions, seed control
CLASSIC: config for the classic model (log. reg.)
BERT: config for the Bert model
"""
from pathlib import Path


# -------------------- PATHS --------------------
BASE_DIR = Path(__file__).resolve().parent.parent # set the bachelor directory as base dir. path is based from the location of this file
TRAINING_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed" # generated data (preprocessed dataframes)
TRAIN_LEMMAS_PATH = PROCESSED_DIR / "train_lemmas.parquet" # need to delete data if preprocessing gets changed
TEST_LEMMAS_PATH = PROCESSED_DIR / "test_lemmas.parquet"


# -------------------- EXPERIMENT DESIGN --------------------



# -------------------- CLASSIC --------------------
MIN_DF = 1 # no vocabulary filter, vocabulary size should depend on the training subset size alone
MAX_FEATURES = None # same reason, a fixed cap would cut harder on small subsets than on large ones
C_VALUE = 1.0 # sklearn default, leaves regualization on
MAX_ITER = 1000 # default of 100 often stops before convergence (before the improvement per step gets small enough) on high-dimensional TF-IDF data.


# -------------------- BERT --------------------