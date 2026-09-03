"""
gets the result csv-file and turns the raw data into the learning curve:
mean and standard deviation per approach and training size, intersection between the curves and the plot.
"""
import matplotlib.pyplot as plt
import pandas as pd
from config import RESULTS_PATH, RESULTS_DIR


# -------------------- HELPERS --------------------
def load_results() -> pd.DataFrame:
    """
    loads the results from the csv ans sums the seeds. returns one row per approch and fraction:
    mean and standard deviation of accuracy and f1, mean runtime, number of runs
    """
    df = pd.read_csv(RESULTS_PATH)

    results = df.groupby(["approach", "fraction"]).agg(
        n_train=("n_train", "first"),
        acc_mean=("accuracy", "mean"),
        acc_std=("accuracy", "std"),
        f1_mean=("f1_macro", "mean"),
        f1_std=("f1_macro", "std"),
        runtime_mean=("runtime", "mean"),
        n_runs=("seed", "count"),
    ).reset_index()

    return results


# -------------------- PLOT --------------------
def plot_learning_curves(results: pd.DataFrame) -> None:
    """
    plots the two learning-curves (accuracy and f1). one curve (mean of the 3 runs with 3 seeds) per approach with a shaded curve for deviation.
    the x axis is logarithmic so the small steps (0.5% - 1%) are good visible.
    """
    plots = [
        ("acc_mean", "acc_std", "Accuracy", "learning_curve_accuracy.png"),
        ("f1_mean", "f1_std", "F1-Score (macro)", "learning_curve_f1.png"),
    ]

    approaches = [
        ("classic", "tab:blue", "TF-IDF + Log. Reg."),
        ("bert", "tab:orange", "BERT (gbert-base)"),
    ]

    for mean_col, std_col, label, filename in plots:
        fig, ax = plt.subplots(figsize=(11, 5))

        for approach, color, name in approaches:
            sub = results[results["approach"] == approach].sort_values("fraction")
            x = sub["fraction"].values
            mean = sub[mean_col].values
            std = sub[std_col].values

            ax.plot(x, mean, marker="o", color=color, label=name)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

        ax.set_xscale("log")

        ticks = results[results["approach"] == "classic"].sort_values("fraction")
        ax.set_xticks(ticks["fraction"].values)
        ax.set_xticklabels([f"{f:.1%}\n({n})" for f, n in zip(ticks["fraction"], ticks["n_train"])])
        ax.minorticks_off()

        ax.set_xlabel("Anteil der Trainingsdaten (Anzahl Artikel)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(RESULTS_DIR / filename, dpi=150)
        plt.close(fig)


# -------------------- TEStING --------------------
if __name__ == "__main__":
    results = load_results()
    #print(results.to_string(index=False))
    plot_learning_curves(results)