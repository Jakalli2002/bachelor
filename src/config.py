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
RESULTS_DIR = BASE_DIR / "results"
RESULTS_PATH = RESULTS_DIR / "results.csv"


# -------------------- EXPERIMENT DESIGN --------------------
# fractions of the training data used per run. dense in the 0.5-1% range because the first measurements suggest the two curves cross somewhere between 1% and 10%.
# capped at 5% for now, everything above takes too long on CPU (10% alone is ~40 min per BERT run).
STEPS = [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.25, 0.50, 1.00]

# one run per seed and step. the seed decides which articles end up in the subset (and for BERT also the classifier head init and the batch order).
# the same seed is used for both approaches so they train on identical articles.
SEEDS = [1, 2, 3]


# -------------------- CLASSIC --------------------
MIN_DF = 1 # no vocabulary filter, vocabulary size should depend on the training subset size alone
MAX_FEATURES = None # same reason, a fixed cap would cut harder on small subsets than on large ones
C_VALUE = 1.0 # sklearn default, leaves regualization on
MAX_ITER = 1000 # default of 100 often stops before convergence (before the improvement per step gets small enough) on high-dimensional TF-IDF data.


# -------------------- BERT --------------------
MODEL_NAME = "deepset/gbert-base" # german BERT, used as reference in Chan/Schweter/Möller 2020
MAX_LENGTH = 256 # BERT caps at 512 anyway, 42% of articles exceed even that, 77% exceed 256. attention cost grows quadratically, so 256 is roughly 4x faster than 512. news articles put the topic in the first paragraph, so the cut is acceptable.
BATCH_SIZE = 16 # 16 for colab run 8 for cpu run up to 5%
EPOCHS = 3 # standard range for BERT fine-tuning (Sun et al. 2019 use 4)
LEARNING_RATE = 2e-5 # standard for fine-tuning: the pretrained layers should only be nudged, not overwritten
WEIGHT_DECAY = 0.01
CATEGORIES = ["Etat", "Inland", "International", "Kultur", "Panorama", "Sport", "Web", "Wirtschaft", "Wissenschaft"]

