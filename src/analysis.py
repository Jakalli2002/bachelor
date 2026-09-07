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

        tick_fractions = [0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
        ticks = results[(results["approach"] == "classic") &
                        (results["fraction"].isin(tick_fractions))].sort_values("fraction")
        ax.set_xticks(ticks["fraction"].values)
        ax.set_xticklabels([f"{f:.1%}".replace(".", ",") for f in ticks["fraction"]])
        ax.minorticks_off()

        ax.set_xlabel("Anteil der Trainingsdaten")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.3)
        ax.legend()

        fig.tight_layout()
        fig.savefig(RESULTS_DIR / filename, dpi=150)
        plt.close(fig)


# -------------------- TABLE --------------------
def build_tables(results: pd.DataFrame) -> None:
    """
    prints three comparison tables (accuracy, f1, runtime) with one row per training
    size and one column per approach. accuracy and f1 include the standard deviation
    across seeds; runtime is the mean in seconds.
    """
    tables = [
        ("acc_mean", "acc_std", "Accuracy", True),
        ("f1_mean", "f1_std", "F1-Score (macro)", True),
        ("runtime_mean", None, "Laufzeit (s)", False),
    ]

    for mean_col, std_col, title, as_percent in tables:
        pivot = results.pivot(index="fraction", columns="approach", values=mean_col)
        n_train = results[results["approach"] == "classic"].set_index("fraction")["n_train"]

        print(f"\n--- {title} ---")
        print(f"{'Anteil':>8} {'Artikel':>8} {'Klassisch':>16} {'BERT':>16}")

        for fraction in pivot.index:
            if as_percent:
                std = results.pivot(index="fraction", columns="approach", values=std_col)
                classic = f"{pivot.loc[fraction, 'classic']:.1%} ± {std.loc[fraction, 'classic']:.1%}"
                bert = f"{pivot.loc[fraction, 'bert']:.1%} ± {std.loc[fraction, 'bert']:.1%}"
            else:
                classic = f"{pivot.loc[fraction, 'classic']:.1f}"
                bert = f"{pivot.loc[fraction, 'bert']:.1f}"

            print(f"{fraction:>8.1%} {n_train[fraction]:>8} {classic:>16} {bert:>16}")


# -------------------- TEStING --------------------
if __name__ == "__main__":
    results = load_results()
    #plot_learning_curves(results)
    #print(build_tables(results))