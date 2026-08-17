import json
from collections import Counter

INPUT_FILE = "dataset/normalized_dataset.jsonl"


def analyze_other():

    topics = Counter()

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                data = json.loads(line)

                if data.get("category") == "Other":

                    topic = data.get("topic", "Unknown")

                    topics[topic] += 1

            except json.JSONDecodeError:
                continue

    print("\n" + "=" * 60)
    print("TOP 'OTHER' TOPICS")
    print("=" * 60)

    total = sum(topics.values())

    print(f"\nTotal Other examples: {total}\n")

    for topic, count in topics.most_common(100):

        percentage = (count / total) * 100

        print(
            f"{topic:<45} "
            f"{count:>5} "
            f"({percentage:.2f}%)"
        )


if __name__ == "__main__":
    analyze_other()