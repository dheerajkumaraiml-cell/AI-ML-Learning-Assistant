import json
from collections import Counter
from pathlib import Path

# Dataset location
INPUT_FILE = Path("dataset/cleaned_dataset.jsonl")


def analyze_dataset():

    topics = Counter()
    levels = Counter()

    total = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)

                total += 1

                topic = data.get("topic", "Unknown")
                level = data.get("level", "Unknown")

                topics[topic] += 1
                levels[level] += 1

            except json.JSONDecodeError:
                continue

    # -----------------------------
    # Display results
    # -----------------------------

    print("\n" + "=" * 50)
    print("DATASET ANALYSIS")
    print("=" * 50)

    print(f"\nTotal Examples: {total}")

    print("\n" + "-" * 50)
    print("TOPIC DISTRIBUTION")
    print("-" * 50)

    for topic, count in topics.most_common():

        percentage = (count / total) * 100

        print(
            f"{topic:<35} {count:>6} ({percentage:.2f}%)"
        )

    print("\n" + "-" * 50)
    print("LEVEL DISTRIBUTION")
    print("-" * 50)

    for level, count in levels.most_common():

        percentage = (count / total) * 100

        print(
            f"{level:<20} {count:>6} ({percentage:.2f}%)"
        )


if __name__ == "__main__":
    analyze_dataset()