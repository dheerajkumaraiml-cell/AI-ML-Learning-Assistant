import json
import re
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# WINDOWS UTF-8 SUPPORT
# ============================================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ============================================================
# FILE PATHS
# ============================================================

TRAIN_FILE = Path(
    "dataset/splits/train.jsonl"
)

VALIDATION_FILE = Path(
    "dataset/splits/validation.jsonl"
)

TEST_FILE = Path(
    "dataset/splits/test.jsonl"
)

REPORT_DIR = Path(
    "dataset/leakage_reports"
)

JSON_REPORT = (
    REPORT_DIR / "near_duplicate_report.json"
)

TEXT_REPORT = (
    REPORT_DIR / "near_duplicate_report.txt"
)


# ============================================================
# SETTINGS
# ============================================================

SIMILARITY_THRESHOLD = 0.85

MAX_RESULTS_PER_COMPARISON = 50

MIN_NGRAM = 3
MAX_NGRAM = 5


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    records = []

    print(f"Loading: {path}")

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as error:

                raise ValueError(
                    f"Invalid JSON in {path} "
                    f"on line {line_number}: {error}"
                )

            if not isinstance(record, dict):

                raise ValueError(
                    f"Line {line_number} in {path} "
                    f"is not a JSON object."
                )

            records.append(record)

    print(
        f"  Loaded: {len(records):,}"
    )

    return records


# ============================================================
# NORMALIZE QUESTION
# ============================================================

def normalize_question(text):

    if text is None:
        return ""

    text = str(text).lower()

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# EXACT OVERLAP
# ============================================================

def exact_overlap(records_a, records_b):

    questions_a = {}

    for index, record in enumerate(records_a):

        question = normalize_question(
            record.get(
                "question",
                ""
            )
        )

        if question:

            questions_a.setdefault(
                question,
                []
            ).append(index)

    overlaps = []

    for index, record in enumerate(records_b):

        question = normalize_question(
            record.get(
                "question",
                ""
            )
        )

        if question in questions_a:

            for index_a in questions_a[
                question
            ]:

                overlaps.append(
                    {
                        "a_index": index_a,
                        "b_index": index,
                        "question": record.get(
                            "question",
                            ""
                        )
                    }
                )

    return overlaps


# ============================================================
# TF-IDF SIMILARITY
# ============================================================

def find_similar_questions(
    records_a,
    records_b
):

    questions_a = [
        normalize_question(
            record.get(
                "question",
                ""
            )
        )
        for record in records_a
    ]

    questions_b = [
        normalize_question(
            record.get(
                "question",
                ""
            )
        )
        for record in records_b
    ]

    print(
        "  Creating TF-IDF representation..."
    )

    all_questions = (
        questions_a
        + questions_b
    )

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(
            MIN_NGRAM,
            MAX_NGRAM
        ),
        lowercase=False,
        min_df=2
    )

    vectorizer.fit(
        all_questions
    )

    matrix_a = vectorizer.transform(
        questions_a
    )

    matrix_b = vectorizer.transform(
        questions_b
    )

    print(
        "  Calculating similarity..."
    )

    similarity_matrix = cosine_similarity(
        matrix_a,
        matrix_b,
        dense_output=False
    )

    similarity_matrix = (
        similarity_matrix.tocoo()
    )

    results = []

    for row, column, score in zip(
        similarity_matrix.row,
        similarity_matrix.col,
        similarity_matrix.data
    ):

        score = float(score)

        if score < SIMILARITY_THRESHOLD:
            continue

        record_a = records_a[row]
        record_b = records_b[column]

        results.append(
            {
                "similarity": round(
                    score,
                    4
                ),
                "source_index": int(row),
                "target_index": int(column),

                "source_question": (
                    record_a.get(
                        "question",
                        ""
                    )
                ),

                "target_question": (
                    record_b.get(
                        "question",
                        ""
                    )
                ),

                "source_category": (
                    record_a.get(
                        "category",
                        "Unknown"
                    )
                ),

                "target_category": (
                    record_b.get(
                        "category",
                        "Unknown"
                    )
                ),

                "source_level": (
                    record_a.get(
                        "level",
                        "Unknown"
                    )
                ),

                "target_level": (
                    record_b.get(
                        "level",
                        "Unknown"
                    )
                )
            }
        )

    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return results[
        :MAX_RESULTS_PER_COMPARISON
    ]


# ============================================================
# ANALYZE ONE PAIR
# ============================================================

def analyze_pair(
    records_a,
    records_b,
    name_a,
    name_b
):

    print()
    print("=" * 70)
    print(
        f"Checking {name_a} <-> {name_b}"
    )
    print("=" * 70)

    exact = exact_overlap(
        records_a,
        records_b
    )

    print(
        f"Exact normalized overlap: "
        f"{len(exact)}"
    )

    similar = find_similar_questions(
        records_a,
        records_b
    )

    print(
        f"Near-duplicate candidates "
        f"(>= {SIMILARITY_THRESHOLD}): "
        f"{len(similar)}"
    )

    return {
        "exact_overlap": exact,
        "similar_pairs": similar
    }


# ============================================================
# WRITE TEXT REPORT
# ============================================================

def write_text_report(report):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with TEXT_REPORT.open(
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "DEEP DATASET LEAKAGE AUDIT\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        file.write(
            "IMPORTANT:\n"
        )

        file.write(
            "This report is analysis only.\n"
        )

        file.write(
            "No dataset records were modified or deleted.\n\n"
        )

        file.write(
            f"Similarity threshold: "
            f"{SIMILARITY_THRESHOLD}\n"
        )

        file.write(
            f"Maximum results per comparison: "
            f"{MAX_RESULTS_PER_COMPARISON}\n\n"
        )

        # ----------------------------------------------------
        # Each comparison
        # ----------------------------------------------------

        comparisons = [
            (
                "TRAIN <-> VALIDATION",
                report["train_validation"]
            ),
            (
                "TRAIN <-> TEST",
                report["train_test"]
            ),
            (
                "VALIDATION <-> TEST",
                report["validation_test"]
            )
        ]

        for title, data in comparisons:

            file.write(
                "\n" + "=" * 70 + "\n"
            )

            file.write(
                title + "\n"
            )

            file.write(
                "=" * 70 + "\n\n"
            )

            file.write(
                "Exact overlap: "
                f"{len(data['exact_overlap'])}\n"
            )

            file.write(
                "Suspicious pairs: "
                f"{len(data['similar_pairs'])}\n\n"
            )

            for number, pair in enumerate(
                data["similar_pairs"],
                start=1
            ):

                file.write(
                    f"[{number}]\n"
                )

                file.write(
                    f"Similarity: "
                    f"{pair['similarity']}\n"
                )

                file.write(
                    f"Source index: "
                    f"{pair['source_index']}\n"
                )

                file.write(
                    f"Target index: "
                    f"{pair['target_index']}\n\n"
                )

                file.write(
                    "SOURCE QUESTION:\n"
                )

                file.write(
                    str(
                        pair["source_question"]
                    )
                    + "\n\n"
                )

                file.write(
                    "TARGET QUESTION:\n"
                )

                file.write(
                    str(
                        pair["target_question"]
                    )
                    + "\n\n"
                )

                file.write(
                    "SOURCE CATEGORY: "
                    f"{pair['source_category']}\n"
                )

                file.write(
                    "TARGET CATEGORY: "
                    f"{pair['target_category']}\n"
                )

                file.write(
                    "SOURCE LEVEL: "
                    f"{pair['source_level']}\n"
                )

                file.write(
                    "TARGET LEVEL: "
                    f"{pair['target_level']}\n"
                )

                file.write(
                    "\n" + "-" * 70 + "\n"
                )

    print()
    print(
        f"Text report saved:"
    )

    print(
        f"  {TEXT_REPORT}"
    )


# ============================================================
# WRITE JSON REPORT
# ============================================================

def write_json_report(report):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with JSON_REPORT.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"JSON report saved:"
    )

    print(
        f"  {JSON_REPORT}"
    )


# ============================================================
# PRINT TOP RESULTS
# ============================================================

def print_top_results(
    results,
    name_a,
    name_b
):

    print()
    print(
        "=" * 70
    )

    print(
        f"TOP SUSPICIOUS PAIRS: "
        f"{name_a} <-> {name_b}"
    )

    print(
        "=" * 70
    )

    if not results:

        print(
            "No suspicious pairs found."
        )

        return

    for number, pair in enumerate(
        results[:20],
        start=1
    ):

        print()
        print(
            f"[{number}] "
            f"Similarity: "
            f"{pair['similarity']}"
        )

        print(
            f"{name_a}:"
        )

        print(
            pair["source_question"]
        )

        print()

        print(
            f"{name_b}:"
        )

        print(
            pair["target_question"]
        )

        print(
            f"Categories: "
            f"{pair['source_category']} "
            f"<-> "
            f"{pair['target_category']}"
        )

        print(
            f"Levels: "
            f"{pair['source_level']} "
            f"<-> "
            f"{pair['target_level']}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "DEEP DATASET LEAKAGE AUDIT"
    )
    print("=" * 70)

    print()
    print(
        "This script will NOT modify your datasets."
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    train = load_jsonl(
        TRAIN_FILE
    )

    validation = load_jsonl(
        VALIDATION_FILE
    )

    test = load_jsonl(
        TEST_FILE
    )

    # --------------------------------------------------------
    # Train vs Validation
    # --------------------------------------------------------

    train_validation = analyze_pair(
        train,
        validation,
        "Train",
        "Validation"
    )

    # --------------------------------------------------------
    # Train vs Test
    # --------------------------------------------------------

    train_test = analyze_pair(
        train,
        test,
        "Train",
        "Test"
    )

    # --------------------------------------------------------
    # Validation vs Test
    # --------------------------------------------------------

    validation_test = analyze_pair(
        validation,
        test,
        "Validation",
        "Test"
    )

    # --------------------------------------------------------
    # Print examples
    # --------------------------------------------------------

    print_top_results(
        train_validation["similar_pairs"],
        "Train",
        "Validation"
    )

    print_top_results(
        train_test["similar_pairs"],
        "Train",
        "Test"
    )

    print_top_results(
        validation_test["similar_pairs"],
        "Validation",
        "Test"
    )

    # --------------------------------------------------------
    # Create report
    # --------------------------------------------------------

    report = {
        "configuration": {
            "similarity_threshold": (
                SIMILARITY_THRESHOLD
            ),
            "ngram_range": [
                MIN_NGRAM,
                MAX_NGRAM
            ],
            "max_results_per_comparison": (
                MAX_RESULTS_PER_COMPARISON
            )
        },

        "dataset_sizes": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test)
        },

        "train_validation": train_validation,

        "train_test": train_test,

        "validation_test": validation_test
    }

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    write_json_report(
        report
    )

    write_text_report(
        report
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "LEAKAGE AUDIT COMPLETED"
    )
    print("=" * 70)

    print()
    print(
        "Your datasets were NOT modified."
    )

    print()
    print(
        "Open this file to inspect the results:"
    )

    print(
        f"  {TEXT_REPORT}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()