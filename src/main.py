import pandas as pd
from pathlib import Path

from src.core.pipeline import run_pipeline
from src.phase2.optimized_phase2 import run_phase2_fast as run_phase2, preload_phase2
from src.phase1.abstract_duplicate_checker import check_against_dataset


# -------------------------------
# Resolve project root safely
# -------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data/master_journals_expanded.csv"


def read_multiline_input():
    print("(Paste abstract. Type END on a new line to finish)\n")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    return "\n".join(lines)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing file: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print("Preloading Phase 2 FAISS...")
    preload_phase2(df)

    while True:
        title = input("Enter journal/article title: ").strip()

        if not title:
            print("Error: Title cannot be empty.")
            continue

        # ---------------- Phase 1 ----------------
        phase1 = run_pipeline(title)
        journal_predictions = phase1.get("journal_predictions", [])

        if journal_predictions and journal_predictions[0].get(
            "confidence", ""
        ).startswith("Strong"):
            print("\nA journal with a very similar scope already exists.")
            print("Please enter a NEW article title.\n")
            continue

        # ---------------- Phase 2 ----------------
        print("\nPhase 2 required. Please enter article abstract:\n")
        user_abstract = read_multiline_input()

        if not user_abstract.strip():
            print("Error: Abstract cannot be empty.")
            continue

        # -------- Abstract Duplicate Check (Phase 1) --------
        duplication = check_against_dataset(user_abstract, df["abstract"].dropna().tolist())
        if duplication["verdict"] == "DUPLICATE":
            print(f"\n❌ Abstract already exists in dataset (duplication confidence: {duplication['confidence'] * 100:.1f}%).")
            print("Please enter a new article abstract.\n")
            continue
        elif duplication["verdict"] == "NEAR_DUPLICATE":
            print(f"\n⚠️ Warning: Abstract is highly similar to existing work (duplication confidence: {duplication['confidence'] * 100:.1f}%).")

        phase2 = run_phase2(user_abstract)

        if not phase2:
            print("\nFINAL DECISION:")
            print("Novel journal scope.")
            break

        semantic = phase2.get("semantic_validation")

        if semantic:
            print("\n[Semantic Analysis]")
            print(f"Domain / Heading        : {semantic.get('topic_heading')}")
            print(f"Topic Alignment Score   : {semantic.get('topic_alignment')}")
            print(f"Embedding Similarity    : {semantic.get('embedding_similarity')}")

            print("\n[Detected Techniques]")
            techniques = semantic.get("techniques", [])
            if techniques:
                for tech in techniques:
                    if isinstance(tech, dict):
                        name = tech.get("name", str(tech))
                        conf = tech.get("confidence")
                        if conf is not None:
                            print(f"- {name} (confidence: {conf})")
                        else:
                            print(f"- {name}")
                    else:
                        # tech is a string
                        print(f"- {tech}")
            else:
                print("No strong techniques detected.")

            print("\n[Overall Confidence]")
            conf = semantic.get("confidence", {})
            print(conf.get("verdict", "N/A"))
            print(conf.get("explanation", ""))

        print("\nFINAL DECISION:")
        rec = phase2.get("submission_recommendation")

        if rec and rec.get("primary_journal"):
            print("\nSUBMISSION RECOMMENDATION:")
            print(f"Primary Journal : {rec['primary_journal']}")
            if rec["alternate_journals"]:
                print("Alternate Journals:")
                for j in rec["alternate_journals"]:
                    print(f"- {j}")
        print(f"Confidence      : {rec['confidence']}")
        print(f"Reason          : {rec['explanation']}")

        break


if __name__ == "__main__":
    main()
