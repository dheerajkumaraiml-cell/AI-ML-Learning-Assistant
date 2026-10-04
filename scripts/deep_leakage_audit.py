import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ============================================================
# WINDOWS UTF-8
# ============================================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("dataset/splits_final_candidate")

TRAIN_FILE = BASE_DIR / "train.jsonl"
VALIDATION_FILE = BASE_DIR / "validation.jsonl"
TEST_FILE = BASE_DIR / "test.jsonl"

REPORT_DIR = Path("dataset/leakage_reports")

REPORT_FILE = (
    REPORT_DIR / "deep_leakage_audit.json"
)

PAIR_REPORT_FILE = (
    REPORT_DIR / "deep_leakage_pairs.json"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_SEED = 42

# We use multiple thresholds so we can distinguish:
#
# >= 0.98  extremely suspicious
# >= 0.95  very suspicious
# >= 0.90  potentially near duplicate
#
# IMPORTANT:
# Similarity is NOT proof of leakage.
# It identifies records that require inspection.
# ============================================================

THRESHOLDS = [
    0.98,
    0.95,
    0.90
]

NGRAM_RANGE = (3, 5)

NEIGHBORS = 5

MIN_QUESTION_LENGTH = 20

MAX_REPORTED_PAIRS = 1000


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    records = []

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
                    f"line {line_number}: {error}"
                ) from error

            if not isinstance(record, dict):

                raise ValueError(
                    f"Record at {path}:{line_number} "
                    f"is not a JSON object."
                )

            records.append(record)

    return records


# ============================================================
# NORMALIZE QUESTION
# ============================================================

def normalize_question(text):

    if text is None:
        return ""

    text = str(text).lower()

    # Remove excessive whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LOAD ALL SPLITS
# ============================================================

def load_splits():

    print()
    print("=" * 70)
    print(
        "DEEP DATASET LEAKAGE AUDIT"
    )
    print("=" * 70)

    print()
    print(
        f"Input directory:"
    )

    print(
        f"  {BASE_DIR}"
    )

    train = load_jsonl(
        TRAIN_FILE
    )

    validation = load_jsonl(
        VALIDATION_FILE
    )

    test = load_jsonl(
        TEST_FILE
    )

    print()
    print(
        "Dataset sizes:"
    )

    print(
        f"  Train      : {len(train):,}"
    )

    print(
        f"  Validation : {len(validation):,}"
    )

    print(
        f"  Test       : {len(test):,}"
    )

    return train, validation, test


# ============================================================
# EXACT QUESTION OVERLAP
# ============================================================

def exact_question_overlap(
    first_records,
    second_records
):

    first_map = defaultdict(list)

    for index, record in enumerate(
        first_records
    ):

        question = normalize_question(
            record.get(
                "question",
                ""
            )
        )

        if question:

            first_map[
                question
            ].append(index)

    overlaps = []

    for second_index, record in enumerate(
        second_records
    ):

        question = normalize_question(
            record.get(
                "question",
                ""
            )
        )

        if not question:
            continue

        if question in first_map:

            overlaps.append(
                {
                    "second_index": second_index,
                    "first_indices": first_map[
                        question
                    ],
                    "question": question
                }
            )

    return overlaps


# ============================================================
# CREATE QUESTIONS
# ============================================================

def prepare_questions(records):

    questions = []

    indices = []

    for index, record in enumerate(
        records
    ):

        question = normalize_question(
            record.get(
                "question",
                ""
            )
        )

        if len(question) >= MIN_QUESTION_LENGTH:

            questions.append(
                question
            )

            indices.append(
                index
            )

    return questions, indices


# ============================================================
# NEAR-DUPLICATE SEARCH
# ============================================================

def search_near_duplicates(
    source_records,
    target_records,
    source_name,
    target_name
):

    print()
    print(
        f"Analyzing: "
        f"{source_name} <-> {target_name}"
    )

    source_questions, source_indices = (
        prepare_questions(
            source_records
        )
    )

    target_questions, target_indices = (
        prepare_questions(
            target_records
        )
    )

    print(
        f"  {source_name} questions: "
        f"{len(source_questions):,}"
    )

    print(
        f"  {target_name} questions: "
        f"{len(target_questions):,}"
    )

    if not source_questions:
        return []

    if not target_questions:
        return []

    # --------------------------------------------------------
    # Fit TF-IDF on BOTH datasets.
    #
    # This is important because we want the same vector
    # space for both sides of the comparison.
    # --------------------------------------------------------

    all_questions = (
        source_questions
        + target_questions
    )

    print(
        "  Creating TF-IDF representation..."
    )

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=NGRAM_RANGE,
        lowercase=False,
        min_df=2
    )

    matrix = vectorizer.fit_transform(
        all_questions
    )

    source_matrix = matrix[
        :len(source_questions)
    ]

    target_matrix = matrix[
        len(source_questions):
    ]

    print(
        f"  TF-IDF matrix: "
        f"{matrix.shape}"
    )

    # --------------------------------------------------------
    # Nearest neighbours.
    # --------------------------------------------------------

    neighbour_count = min(
        NEIGHBORS,
        len(target_questions)
    )

    model = NearestNeighbors(
        n_neighbors=neighbour_count,
        metric="cosine",
        algorithm="brute"
    )

    model.fit(
        target_matrix
    )

    distances, neighbours = (
        model.kneighbors(
            source_matrix
        )
    )

    results = []

    for row in range(
        len(source_questions)
    ):

        source_index = source_indices[
            row
        ]

        for position in range(
            len(neighbours[row])
        ):

            target_row = neighbours[
                row
            ][position]

            target_index = target_indices[
                target_row
            ]

            distance = distances[
                row
            ][position
            ]

            similarity = (
                1.0 - distance
            )

            if similarity >= min(
                THRESHOLDS
            ):

                results.append(
                    {
                        "source_split":
                            source_name,

                        "source_index":
                            source_index,

                        "target_split":
                            target_name,

                        "target_index":
                            target_index,

                        "similarity":
                            round(
                                float(
                                    similarity
                                ),
                                4
                            ),

                        "source_question":
                            source_records[
                                source_index
                            ].get(
                                "question",
                                ""
                            ),

                        "target_question":
                            target_records[
                                target_index
                            ].get(
                                "question",
                                ""
                            )
                    }
                )

    # --------------------------------------------------------
    # Sort highest similarity first.
    # --------------------------------------------------------

    results.sort(
        key=lambda item:
            item["similarity"],
        reverse=True
    )

    return results


# ============================================================
# COUNT BY THRESHOLD
# ============================================================

def threshold_counts(
    pairs
):

    counts = {}

    for threshold in THRESHOLDS:

        counts[
            str(threshold)
        ] = sum(
            1
            for pair in pairs
            if pair["similarity"] >= threshold
        )

    return counts


# ============================================================
# PRINT TOP PAIRS
# ============================================================

def print_top_pairs(
    pairs,
    source_name,
    target_name
):

    print()
    print(
        f"Top suspicious pairs: "
        f"{source_name} <-> {target_name}"
    )

    print("-" * 70)

    if not pairs:

        print(
            "No near-duplicate candidates found."
        )

        return

    for number, pair in enumerate(
        pairs[:20],
        start=1
    ):

        print()
        print(
            f"[{number}] "
            f"Similarity: "
            f"{pair['similarity']}"
        )

        print(
            f"Source: "
            f"{pair['source_question'][:300]}"
        )

        print(
            f"Target: "
            f"{pair['target_question'][:300]}"
        )


# ============================================================
# VERIFY RECORD COUNTS
# ============================================================

def verify_total_records(
    train,
    validation,
    test
):

    total = (
        len(train)
        + len(validation)
        + len(test)
    )

    expected = 37089

    print()
    print(
        "RECORD COUNT CHECK"
    )

    print("-" * 70)

    print(
        f"Expected: {expected:,}"
    )

    print(
        f"Actual  : {total:,}"
    )

    if total != expected:

        print(
            "WARNING: Total differs from "
            "the master dataset."
        )

        return False

    print(
        "PASS: All 37,089 records are present."
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    train, validation, test = (
        load_splits()
    )

    # --------------------------------------------------------
    # Count verification.
    # --------------------------------------------------------

    count_ok = verify_total_records(
        train,
        validation,
        test
    )

    # --------------------------------------------------------
    # Exact overlaps.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "EXACT QUESTION OVERLAP"
    )
    print("=" * 70)

    train_validation_exact = (
        exact_question_overlap(
            train,
            validation
        )
    )

    train_test_exact = (
        exact_question_overlap(
            train,
            test
        )
    )

    validation_test_exact = (
        exact_question_overlap(
            validation,
            test
        )
    )

    print(
        f"Train <-> Validation: "
        f"{len(train_validation_exact)}"
    )

    print(
        f"Train <-> Test: "
        f"{len(train_test_exact)}"
    )

    print(
        f"Validation <-> Test: "
        f"{len(validation_test_exact)}"
    )

    # --------------------------------------------------------
    # Near duplicate analysis.
    # --------------------------------------------------------

    tv_pairs = search_near_duplicates(
        train,
        validation,
        "train",
        "validation"
    )

    tt_pairs = search_near_duplicates(
        train,
        test,
        "train",
        "test"
    )

    vt_pairs = search_near_duplicates(
        validation,
        test,
        "validation",
        "test"
    )

    # --------------------------------------------------------
    # Print results.
    # --------------------------------------------------------

    print_top_pairs(
        tv_pairs,
        "train",
        "validation"
    )

    print_top_pairs(
        tt_pairs,
        "train",
        "test"
    )

    print_top_pairs(
        vt_pairs,
        "validation",
        "test"
    )

    # --------------------------------------------------------
    # Threshold summary.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "NEAR-DUPLICATE THRESHOLD SUMMARY"
    )
    print("=" * 70)

    print()
    print(
        "Train <-> Validation:"
    )

    for threshold, count in (
        threshold_counts(
            tv_pairs
        ).items()
    ):

        print(
            f"  >= {threshold}: {count}"
        )

    print()
    print(
        "Train <-> Test:"
    )

    for threshold, count in (
        threshold_counts(
            tt_pairs
        ).items()
    ):

        print(
            f"  >= {threshold}: {count}"
        )

    print()
    print(
        "Validation <-> Test:"
    )

    for threshold, count in (
        threshold_counts(
            vt_pairs
        ).items()
    ):

        print(
            f"  >= {threshold}: {count}"
        )

    # --------------------------------------------------------
    # Save report.
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {

        "configuration": {

            "similarity_thresholds":
                THRESHOLDS,

            "ngram_range":
                list(NGRAM_RANGE),

            "neighbors":
                NEIGHBORS,

            "min_question_length":
                MIN_QUESTION_LENGTH,

            "max_reported_pairs":
                MAX_REPORTED_PAIRS
        },

        "dataset_sizes": {

            "train":
                len(train),

            "validation":
                len(validation),

            "test":
                len(test),

            "total":
                len(train)
                + len(validation)
                + len(test)
        },

        "exact_overlap": {

            "train_validation":
                len(train_validation_exact),

            "train_test":
                len(train_test_exact),

            "validation_test":
                len(validation_test_exact)
        },

        "near_duplicate_counts": {

            "train_validation":
                threshold_counts(
                    tv_pairs
                ),

            "train_test":
                threshold_counts(
                    tt_pairs
                ),

            "validation_test":
                threshold_counts(
                    vt_pairs
                )
        },

        "status": {

            "record_count_ok":
                count_ok,

            "exact_overlap_free":
                (
                    len(train_validation_exact)
                    == 0
                    and
                    len(train_test_exact)
                    == 0
                    and
                    len(validation_test_exact)
                    == 0
                )
        }
    }

    with REPORT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Save suspicious pairs separately.
    # --------------------------------------------------------

    all_pairs = (
        tv_pairs
        + tt_pairs
        + vt_pairs
    )

    all_pairs.sort(
        key=lambda item:
            item["similarity"],
        reverse=True
    )

    all_pairs = all_pairs[
        :MAX_REPORTED_PAIRS
    ]

    with PAIR_REPORT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_pairs,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Final status.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "DEEP LEAKAGE AUDIT COMPLETED"
    )
    print("=" * 70)

    print()
    print(
        "Reports:"
    )

    print(
        f"  {REPORT_FILE}"
    )

    print(
        f"  {PAIR_REPORT_FILE}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Near-duplicate candidates are NOT "
        "automatically leakage."
    )

    print(
        "Inspect the highest-similarity pairs "
        "before deleting anything."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()