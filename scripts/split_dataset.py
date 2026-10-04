import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path


# ============================================================
# WINDOWS / TERMINAL UTF-8 SUPPORT
# ============================================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ============================================================
# CONFIGURATION
# ============================================================

# Master dataset.
# IMPORTANT: This file is READ ONLY.
INPUT_FILE = Path("dataset/merged_dataset.jsonl")

# Output directory.
OUTPUT_DIR = Path("dataset/splits")

# Output files.
TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALIDATION_FILE = OUTPUT_DIR / "validation.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"


# ------------------------------------------------------------
# Split ratios
# ------------------------------------------------------------

TRAIN_RATIO = 0.90
VALIDATION_RATIO = 0.05
TEST_RATIO = 0.05


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

RANDOM_SEED = 42


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_configuration():
    """
    Make sure the split ratios add up to 100%.
    """

    total = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if abs(total - 1.0) > 0.000001:
        raise ValueError(
            "TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO "
            "must equal 1.0"
        )


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(file_path):
    """
    Load a JSONL dataset.

    Each non-empty line must contain one JSON object.
    """

    records = []

    print()
    print("Loading dataset:")
    print(f"  {file_path}")

    with file_path.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1
        ):

            line = line.strip()

            # Ignore empty lines.
            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as error:

                raise ValueError(
                    f"Invalid JSON on line "
                    f"{line_number}: {error}"
                ) from error

            if not isinstance(record, dict):

                raise ValueError(
                    f"Line {line_number} is not "
                    f"a JSON object."
                )

            records.append(record)

    print(
        f"  Loaded records: {len(records):,}"
    )

    return records


# ============================================================
# SAVE JSONL
# ============================================================

def save_jsonl(records, file_path):
    """
    Save records to JSONL format.

    The original master dataset is never written here.
    """

    with file_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )


# ============================================================
# NORMALIZE QUESTION
# ============================================================

def normalize_question(question):
    """
    Light normalization used ONLY for leakage checking.

    This function does NOT modify the dataset.
    """

    if question is None:
        return ""

    question = str(question)

    # Lowercase.
    question = question.lower()

    # Normalize whitespace.
    question = " ".join(
        question.split()
    )

    return question


# ============================================================
# GET CATEGORY + LEVEL STRATUM
# ============================================================

def get_stratum(record):
    """
    Return the category + level grouping.

    Example:

        ("Machine Learning", "Beginner")
        ("NLP", "Advanced")
        ("Python", "Unknown")

    Missing values are preserved as Unknown/Other.
    """

    category = record.get(
        "category",
        "Other"
    )

    level = record.get(
        "level",
        "Unknown"
    )

    if category is None:
        category = "Other"

    if level is None:
        level = "Unknown"

    category = str(category).strip()
    level = str(level).strip()

    if not category:
        category = "Other"

    if not level:
        level = "Unknown"

    return category, level


# ============================================================
# SPLIT ONE GROUP
# ============================================================

def split_group(
    records,
    random_generator
):
    """
    Split one category + level group.

    Target:

        90% training
         5% validation
         5% testing

    Very small groups are handled conservatively.
    """

    records = records.copy()

    # Shuffle reproducibly.
    random_generator.shuffle(records)

    count = len(records)

    # --------------------------------------------------------
    # Very small groups
    # --------------------------------------------------------

    # With one record, a meaningful 90/5/5 split
    # is impossible.
    if count == 1:
        return records, [], []

    # With two records, keep both for training.
    if count == 2:
        return records, [], []

    # With three records, keep all for training.
    if count == 3:
        return records, [], []

    # --------------------------------------------------------
    # Normal groups
    # --------------------------------------------------------

    validation_count = round(
        count * VALIDATION_RATIO
    )

    test_count = round(
        count * TEST_RATIO
    )

    # For sufficiently large groups, make sure
    # validation and test receive at least one record.
    if count >= 20:

        validation_count = max(
            1,
            validation_count
        )

        test_count = max(
            1,
            test_count
        )

    # --------------------------------------------------------
    # Safety:
    # Always keep at least one training example.
    # --------------------------------------------------------

    if (
        validation_count
        + test_count
        >= count
    ):

        validation_count = min(
            validation_count,
            count - 1
        )

        remaining = (
            count
            - validation_count
            - 1
        )

        test_count = min(
            test_count,
            max(0, remaining)
        )

    train_count = (
        count
        - validation_count
        - test_count
    )

    # --------------------------------------------------------
    # Slice the group
    # --------------------------------------------------------

    train_records = records[
        :train_count
    ]

    validation_start = train_count

    validation_end = (
        validation_start
        + validation_count
    )

    validation_records = records[
        validation_start:validation_end
    ]

    test_records = records[
        validation_end:
    ]

    return (
        train_records,
        validation_records,
        test_records
    )


# ============================================================
# CREATE STRATIFIED SPLIT
# ============================================================

def create_split(records):
    """
    Create the train/validation/test split.

    Stratification key:

        category + level
    """

    random_generator = random.Random(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Create groups
    # --------------------------------------------------------

    groups = defaultdict(list)

    for record in records:

        stratum = get_stratum(record)

        groups[stratum].append(record)

    print()
    print(
        "Number of category+level groups:",
        len(groups)
    )

    # --------------------------------------------------------
    # Create empty output lists
    # --------------------------------------------------------

    train_records = []
    validation_records = []
    test_records = []

    # --------------------------------------------------------
    # Split every category + level group
    # --------------------------------------------------------

    for stratum in sorted(groups):

        group = groups[stratum]

        (
            train_group,
            validation_group,
            test_group
        ) = split_group(
            group,
            random_generator
        )

        train_records.extend(
            train_group
        )

        validation_records.extend(
            validation_group
        )

        test_records.extend(
            test_group
        )

    # --------------------------------------------------------
    # Shuffle final datasets
    # --------------------------------------------------------

    random_generator.shuffle(
        train_records
    )

    random_generator.shuffle(
        validation_records
    )

    random_generator.shuffle(
        test_records
    )

    return (
        train_records,
        validation_records,
        test_records
    )


# ============================================================
# RECORD COUNT VERIFICATION
# ============================================================

def verify_record_counts(
    original,
    train,
    validation,
    test
):
    """
    Verify that no records disappeared.

    Original count must equal:

        train + validation + test
    """

    original_count = len(
        original
    )

    split_count = (
        len(train)
        + len(validation)
        + len(test)
    )

    print()
    print(
        "Record count verification"
    )
    print("-" * 60)

    print(
        f"Original dataset : "
        f"{original_count:,}"
    )

    print(
        f"Train            : "
        f"{len(train):,}"
    )

    print(
        f"Validation       : "
        f"{len(validation):,}"
    )

    print(
        f"Test             : "
        f"{len(test):,}"
    )

    print(
        f"Combined splits  : "
        f"{split_count:,}"
    )

    if original_count != split_count:

        raise RuntimeError(
            "\nERROR: Record count mismatch!\n"
            f"Original: {original_count}\n"
            f"Splits: {split_count}"
        )

    print()
    print(
        "PASS: All records are accounted for."
    )


# ============================================================
# QUESTION SET
# ============================================================

def get_question_set(records):
    """
    Create a set containing normalized questions.
    """

    questions = set()

    for record in records:

        question = normalize_question(
            record.get(
                "question",
                ""
            )
        )

        if question:
            questions.add(question)

    return questions


# ============================================================
# LEAKAGE CHECK
# ============================================================

def verify_question_leakage(
    train,
    validation,
    test
):
    """
    Check whether identical normalized questions
    appear in multiple splits.

    This catches exact/lightly-normalized overlap.

    It does NOT detect semantic similarity.
    """

    print()
    print(
        "Checking question overlap"
    )
    print("-" * 60)

    train_questions = get_question_set(
        train
    )

    validation_questions = get_question_set(
        validation
    )

    test_questions = get_question_set(
        test
    )

    # --------------------------------------------------------
    # Train vs Validation
    # --------------------------------------------------------

    train_validation_overlap = (
        train_questions
        & validation_questions
    )

    # --------------------------------------------------------
    # Train vs Test
    # --------------------------------------------------------

    train_test_overlap = (
        train_questions
        & test_questions
    )

    # --------------------------------------------------------
    # Validation vs Test
    # --------------------------------------------------------

    validation_test_overlap = (
        validation_questions
        & test_questions
    )

    print(
        "Train <-> Validation overlap:",
        len(train_validation_overlap)
    )

    print(
        "Train <-> Test overlap:",
        len(train_test_overlap)
    )

    print(
        "Validation <-> Test overlap:",
        len(validation_test_overlap)
    )

    # --------------------------------------------------------
    # Fail if overlap exists
    # --------------------------------------------------------

    if train_validation_overlap:

        raise RuntimeError(
            "Question overlap detected between "
            "training and validation sets."
        )

    if train_test_overlap:

        raise RuntimeError(
            "Question overlap detected between "
            "training and test sets."
        )

    if validation_test_overlap:

        raise RuntimeError(
            "Question overlap detected between "
            "validation and test sets."
        )

    print()
    print(
        "PASS: No normalized question overlap "
        "was found between splits."
    )


# ============================================================
# DISTRIBUTION
# ============================================================

def get_distribution(
    records,
    field
):
    """
    Count values for a metadata field.
    """

    counter = Counter()

    for record in records:

        value = record.get(
            field,
            "Unknown"
        )

        if value is None:
            value = "Unknown"

        value = str(value).strip()

        if not value:
            value = "Unknown"

        counter[value] += 1

    return counter


# ============================================================
# PRINT DISTRIBUTION
# ============================================================

def print_distribution(
    split_name,
    records,
    field
):
    """
    Print category or level distribution.
    """

    counter = get_distribution(
        records,
        field
    )

    total = len(records)

    print()
    print(
        f"{split_name} - "
        f"{field.upper()}"
    )

    print("-" * 60)

    for value, count in counter.most_common():

        percentage = (
            count / total * 100
            if total > 0
            else 0
        )

        print(
            f"{value:<25} "
            f"{count:>7,} "
            f"({percentage:>6.2f}%)"
        )


# ============================================================
# PRINT ORIGINAL STRATA
# ============================================================

def print_stratum_summary(records):
    """
    Print category + level distribution
    of the original dataset.
    """

    groups = Counter(
        get_stratum(record)
        for record in records
    )

    print()
    print(
        "CATEGORY + LEVEL GROUPS"
    )

    print("-" * 60)

    for (
        category,
        level
    ), count in sorted(
        groups.items()
    ):

        print(
            f"{category:<25} "
            f"{level:<15} "
            f"{count:>7,}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    original,
    train,
    validation,
    test
):
    """
    Print final dataset split statistics.
    """

    total = len(
        original
    )

    print()
    print("=" * 70)
    print(
        "FINAL DATASET SPLIT SUMMARY"
    )
    print("=" * 70)

    datasets = [
        ("Training", train),
        ("Validation", validation),
        ("Testing", test),
    ]

    # --------------------------------------------------------
    # Basic counts
    # --------------------------------------------------------

    for name, records in datasets:

        percentage = (
            len(records)
            / total
            * 100
        )

        print(
            f"{name:<15} "
            f"{len(records):>8,} "
            f"({percentage:>6.2f}%)"
        )

    print("-" * 70)

    print(
        f"{'Total':<15} "
        f"{total:>8,} "
        f"(100.00%)"
    )

    # --------------------------------------------------------
    # Category distributions
    # --------------------------------------------------------

    for name, records in datasets:

        print_distribution(
            name,
            records,
            "category"
        )

    # --------------------------------------------------------
    # Level distributions
    # --------------------------------------------------------

    for name, records in datasets:

        print_distribution(
            name,
            records,
            "level"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "TRAIN / VALIDATION / TEST "
        "DATASET SPLITTER"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------

    validate_configuration()

    # --------------------------------------------------------
    # Check master dataset
    # --------------------------------------------------------

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nDataset not found:\n"
            f"{INPUT_FILE}\n\n"
            "Run this script from the project root directory."
        )

    # --------------------------------------------------------
    # Prevent accidental overwrite
    # --------------------------------------------------------

    existing_files = []

    for file_path in [
        TRAIN_FILE,
        VALIDATION_FILE,
        TEST_FILE
    ]:

        if file_path.exists():

            existing_files.append(
                file_path
            )

    if existing_files:

        print()
        print(
            "WARNING: Split files already exist:"
        )

        for file_path in existing_files:

            print(
                f"  {file_path}"
            )

        print()
        print(
            "No files were overwritten."
        )

        print(
            "If you intentionally want to regenerate "
            "the splits, delete the existing split files "
            "manually and run this script again."
        )

        return

    # --------------------------------------------------------
    # Load master dataset
    # --------------------------------------------------------

    records = load_jsonl(
        INPUT_FILE
    )

    if not records:

        raise ValueError(
            "The dataset contains no records."
        )

    # --------------------------------------------------------
    # Show original distribution
    # --------------------------------------------------------

    print_stratum_summary(
        records
    )

    # --------------------------------------------------------
    # Create split
    # --------------------------------------------------------

    print()
    print(
        "Creating stratified split..."
    )

    print(
        f"Training ratio   : "
        f"{TRAIN_RATIO:.0%}"
    )

    print(
        f"Validation ratio : "
        f"{VALIDATION_RATIO:.0%}"
    )

    print(
        f"Testing ratio    : "
        f"{TEST_RATIO:.0%}"
    )

    print(
        f"Random seed      : "
        f"{RANDOM_SEED}"
    )

    (
        train,
        validation,
        test
    ) = create_split(
        records
    )

    # --------------------------------------------------------
    # Verify record counts
    # --------------------------------------------------------

    verify_record_counts(
        records,
        train,
        validation,
        test
    )

    # --------------------------------------------------------
    # Verify question leakage
    # --------------------------------------------------------

    verify_question_leakage(
        train,
        validation,
        test
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save datasets
    # --------------------------------------------------------

    print()
    print(
        "Saving split files..."
    )

    save_jsonl(
        train,
        TRAIN_FILE
    )

    save_jsonl(
        validation,
        VALIDATION_FILE
    )

    save_jsonl(
        test,
        TEST_FILE
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print_final_summary(
        records,
        train,
        validation,
        test
    )

    # --------------------------------------------------------
    # Success message
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "DATASET SPLITTING COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print()
    print("Created files:")

    print(
        f"  {TRAIN_FILE}"
    )

    print(
        f"  {VALIDATION_FILE}"
    )

    print(
        f"  {TEST_FILE}"
    )

    print()
    print(
        "Master dataset was NOT modified:"
    )

    print(
        f"  {INPUT_FILE}"
    )

    print()
    print("Important:")

    print(
        "  - No records were intentionally deleted."
    )

    print(
        "  - Unknown-level records were preserved."
    )

    print(
        "  - Short-answer records were preserved."
    )

    print(
        "  - The split is reproducible."
    )

    print(
        "  - Category + level stratification was used."
    )

    print(
        "  - Normalized question overlap was checked."
    )

    print()
    print(
        "Next step: perform a deeper near-duplicate/"
        "semantic leakage audit before tokenization."
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()