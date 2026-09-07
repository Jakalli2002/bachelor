"""
runs the experiment. for every training size and every seed, both approaches are trained on the exact same subset and evaluated on the full test set.
writes one row per run to results/results.csv, appending after each run so a crash does not lose everything. - results are processed in evaluation.py
"""
import csv
import time
from sklearn.metrics import accuracy_score, f1_score
from bert import run_bert
from classic import get_preprocessed, run_classic
from config import (CATEGORIES, RESULTS_PATH, SEEDS, STEPS, TEST_LEMMAS_PATH, TEST_PATH, TRAIN_LEMMAS_PATH, TRAINING_PATH)
from data import get_data_subset, load_data


# -------------------- HELPERS --------------------
FIELDS = ["approach", "fraction", "n_train", "seed", "accuracy", "f1_macro", "runtime"] + CATEGORIES # fields for the result.csv

def write_row(row: dict) -> None:
    """
    appends one result row to the csv. the header is only written if the file does not exist yet, 
    so the experiment can be interrupted and continued without overwriting earlier runs or repeating the header.
    """
    is_new = not RESULTS_PATH.exists()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RESULTS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)

def build_row(approach: str, fraction: float, seed: int, subset, y_true, y_pred, runtime: float) -> dict:
    """
    collects everything that describes one run: identifiers, metrics, runtime and the category distribution of the training subset.
    f1 is macro averaged, so every category counts the same regardless of how often it occurs - the dataset is unbalanced (Panorama has about 3x as many articles as Kultur).
    """
    row = {
        "approach": approach,
        "fraction": fraction,
        "n_train": len(subset),
        "seed": seed,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "runtime": round(runtime, 2),
    }

    counts = subset["category"].value_counts().to_dict()
    for category in CATEGORIES:
        row[category] = counts.get(category, 0)

    return row

def load_done_runs() -> set:
    """
    reads which runs are already in the results file, so an interrupted experiment can be continued without repeating or duplicating runs.
    returns a set of (approach, fraction, seed) tuples.
    """
    if not RESULTS_PATH.exists():
        return set()

    done = set()
    with open(RESULTS_PATH, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add((row["approach"], float(row["fraction"]), int(row["seed"])))
    return done


# -------------------- EXPERIMENT --------------------
def run_experiment() -> None:
    """
    main loop over training sizes and seeds. the subset is drawn once per fraction and seed, then handed to both approaches, so any difference between them
    cannot come from different training articles. both are always evaluated against the full test set, which never changes.
    runtime covers training and prediction, measured the same way for both approaches.
    """
    train_df = get_preprocessed(load_data(TRAINING_PATH), TRAIN_LEMMAS_PATH)
    test_df = get_preprocessed(load_data(TEST_PATH), TEST_LEMMAS_PATH)
    y_true = test_df["category"].values

    done = load_done_runs()

    for fraction in STEPS:
        for seed in SEEDS:
            if ("classic", fraction, seed) in done and ("bert", fraction, seed) in done:
                print(f"skip {fraction:.1%}, seed {seed} (already done)")
                continue

            subset = get_data_subset(train_df, fraction, seed)
            print(f"\n=== {fraction:.1%} ({len(subset)} articles), seed {seed} ===")

            # classic
            if ("classic", fraction, seed) not in done:
                start = time.perf_counter()
                preds = run_classic(subset["lemmas"], subset["category"], test_df["lemmas"])
                runtime = time.perf_counter() - start
                row = build_row("classic", fraction, seed, subset, y_true, preds, runtime)
                write_row(row)
                print(f"classic: acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f} ({runtime:.1f}s)")

            # bert
            if ("bert", fraction, seed) not in done:
                start = time.perf_counter()
                preds = run_bert(subset["text"], subset["category"], test_df["text"], seed=seed)
                runtime = time.perf_counter() - start
                row = build_row("bert", fraction, seed, subset, y_true, preds, runtime)
                write_row(row)
                print(f"bert: acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f} ({runtime:.1f}s)")


# -------------------- TEStING --------------------
if __name__ == "__main__":
    run_experiment()