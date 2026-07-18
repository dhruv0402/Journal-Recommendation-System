import matplotlib.pyplot as plt


def plot_journal_confidence(results: dict):
    journals = [j["journal"] for j in results["ranked_journals"]]
    confidences = [j["confidence"] for j in results["ranked_journals"]]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(journals, confidences)

    plt.ylim(0, 1.0)
    plt.ylabel("Confidence")
    plt.title("Journal Recommendation Confidence")

    # annotate bars
    for bar, score in zip(bars, confidences):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{score:.2f}",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.show()