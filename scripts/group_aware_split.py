import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# ============================================================
# WINDOWS UTF-8 SUPPORT
# ============================================================

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


# ============================================================
# PATHS
# ============================================================

MASTER_DATASET = Path(
    "dataset/merged_dataset.jsonl"
)

# NEW output directory.
# Your previous splits_grouped directory is NOT overwritten.
OUTPUT_DIR = Path(
    "dataset/splits_grouped_v2"
)

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALIDATION_FILE = OUTPUT_DIR / "validation.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"


# ============================================================
# REPORTS
# ============================================================

REPORT_DIR = Path(
    "dataset/leakage_reports"
)

GROUP_REPORT = (
    REPORT_DIR / "duplicate_groups_v2.json"
)

SPLIT_REPORT = (
    REPORT_DIR / "group_aware_split_report_v2.json"
)


# ============================================================
# SPLIT CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.90
VALIDATION_RATIO = 0.05
TEST_RATIO = 0.05

RANDOM_SEED = 42


# ============================================================
# NEAR-DUPLICATE CONFIGURATION
# ============================================================

SIMILARITY_THRESHOLD = 0.90

NGRAM_MIN = 3
NGRAM_MAX = 5

NEIGHBORS_PER_RECORD = 5

MIN_QUESTION_LENGTH = 20


# ============================================================
# CONFIGURATION VALIDATION
# ============================================================

def validate_configuration():

    total_ratio = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if abs(total_ratio - 1.0) > 0.000001:

        raise ValueError(
            "Train + validation + test ratios "
            "must equal 1.0."
        )


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    if not path.exists():

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )

    records = []

    print()
    print(f"Loading: {path}")

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
                    f"Invalid JSON on line "
                    f"{line_number}: {error}"
                ) from error

            if not isinstance(record, dict):

                raise ValueError(
                    f"Line {line_number} "
                    f"is not a JSON object."
                )

            records.append(record)

    print(
        f"Loaded records: {len(records):,}"
    )

    return records


# ============================================================
# SAVE JSONL
# ============================================================

def save_jsonl(
    records,
    path
):

    with path.open(
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

def normalize_question(text):

    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# CATEGORY
# ============================================================

def get_category(record):

    value = record.get(
        "category",
        "Other"
    )

    if value is None:
        return "Other"

    value = str(value).strip()

    if not value:
        return "Other"

    return value


# ============================================================
# LEVEL
# ============================================================

def get_level(record):

    value = record.get(
        "level",
        "Unknown"
    )

    if value is None:
        return "Unknown"

    value = str(value).strip()

    if not value:
        return "Unknown"

    return value


# ============================================================
# EXACT DUPLICATE GROUPS
# ============================================================

def find_exact_duplicate_groups(records):

    question_map = defaultdict(list)

    for index, record in enumerate(records):

        question = normalize_question(
            record.get(
                "question",
                ""
            )
        )

        if question:

            question_map[
                question
            ].append(index)

    groups = []

    for question, indices in question_map.items():

        if len(indices) > 1:

            groups.append(
                {
                    "type": "exact",
                    "indices": indices
                }
            )

    return groups


# ============================================================
# UNION FIND
# ============================================================

class UnionFind:

    def __init__(self, size):

        self.parent = list(
            range(size)
        )

        self.rank = [
            0
            for _ in range(size)
        ]

    def find(self, value):

        while self.parent[value] != value:

            self.parent[value] = (
                self.parent[
                    self.parent[value]
                ]
            )

            value = self.parent[value]

        return value

    def union(
        self,
        first,
        second
    ):

        root_first = self.find(
            first
        )

        root_second = self.find(
            second
        )

        if root_first == root_second:
            return

        if self.rank[root_first] < self.rank[root_second]:

            self.parent[root_first] = root_second

        elif self.rank[root_first] > self.rank[root_second]:

            self.parent[root_second] = root_first

        else:

            self.parent[root_second] = root_first

            self.rank[root_first] += 1


# ============================================================
# FIND NEAR DUPLICATE GROUPS
# ============================================================

def find_near_duplicate_groups(records):

    print()
    print("=" * 70)
    print(
        "SEARCHING FOR NEAR-DUPLICATE GROUPS"
    )
    print("=" * 70)

    questions = []

    valid_indices = []

    for index, record in enumerate(records):

        question = normalize_question(
            record.get(
                "question",
                ""
            )
        )

        questions.append(
            question
        )

        if len(question) >= MIN_QUESTION_LENGTH:

            valid_indices.append(
                index
            )

    print()
    print(
        f"Questions used for similarity analysis: "
        f"{len(valid_indices):,}"
    )

    if not valid_indices:

        return []

    valid_questions = [
        questions[index]
        for index in valid_indices
    ]

    # --------------------------------------------------------
    # TF-IDF
    # --------------------------------------------------------

    print(
        "Creating character TF-IDF matrix..."
    )

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(
            NGRAM_MIN,
            NGRAM_MAX
        ),
        lowercase=False,
        min_df=2
    )

    matrix = vectorizer.fit_transform(
        valid_questions
    )

    print(
        f"TF-IDF matrix: {matrix.shape}"
    )

    # --------------------------------------------------------
    # Nearest neighbour search
    # --------------------------------------------------------

    print(
        "Searching for similar questions..."
    )

    distance_threshold = (
        1.0 - SIMILARITY_THRESHOLD
    )

    neighbour_count = min(
        NEIGHBORS_PER_RECORD + 1,
        len(valid_indices)
    )

    model = NearestNeighbors(
        n_neighbors=neighbour_count,
        metric="cosine",
        algorithm="brute"
    )

    model.fit(matrix)

    distances, neighbours = (
        model.kneighbors(
            matrix
        )
    )

    union_find = UnionFind(
        len(records)
    )

    connections = 0

    # --------------------------------------------------------
    # Connect similar questions
    # --------------------------------------------------------

    for row in range(
        len(valid_indices)
    ):

        source_index = valid_indices[
            row
        ]

        for neighbour_position in range(
            len(neighbours[row])
        ):

            target_row = neighbours[
                row
            ][neighbour_position]

            target_index = valid_indices[
                target_row
            ]

            if source_index == target_index:
                continue

            distance = distances[
                row
            ][neighbour_position]

            similarity = (
                1.0 - distance
            )

            if similarity >= SIMILARITY_THRESHOLD:

                connections += 1

                union_find.union(
                    source_index,
                    target_index
                )

    print(
        f"Similarity connections: "
        f"{connections:,}"
    )

    # --------------------------------------------------------
    # Build connected components
    # --------------------------------------------------------

    components = defaultdict(list)

    for index in range(
        len(records)
    ):

        root = union_find.find(
            index
        )

        components[root].append(
            index
        )

    groups = []

    for indices in components.values():

        if len(indices) > 1:

            groups.append(
                indices
            )

    print(
        f"Near-duplicate groups: "
        f"{len(groups):,}"
    )

    return groups


# ============================================================
# MERGE OVERLAPPING GROUPS
# ============================================================

def merge_duplicate_groups(
    exact_groups,
    near_groups,
    record_count
):

    print()
    print(
        "Merging overlapping groups..."
    )

    union_find = UnionFind(
        record_count
    )

    for group in exact_groups:

        first = group["indices"][0]

        for index in group["indices"][1:]:

            union_find.union(
                first,
                index
            )

    for group in near_groups:

        first = group[0]

        for index in group[1:]:

            union_find.union(
                first,
                index
            )

    components = defaultdict(list)

    for index in range(
        record_count
    ):

        root = union_find.find(
            index
        )

        components[root].append(
            index
        )

    final_groups = []

    for indices in components.values():

        final_groups.append(
            sorted(indices)
        )

    final_groups.sort(
        key=lambda group: (
            -len(group),
            group[0]
        )
    )

    print(
        f"Total final groups: "
        f"{len(final_groups):,}"
    )

    duplicate_group_count = sum(
        1
        for group in final_groups
        if len(group) > 1
    )

    print(
        f"Groups containing multiple records: "
        f"{duplicate_group_count:,}"
    )

    return final_groups


# ============================================================
# CREATE GROUP INFORMATION
# ============================================================

def create_group_objects(
    records,
    groups
):

    group_objects = []

    for group_id, indices in enumerate(
        groups,
        start=1
    ):

        categories = Counter()

        levels = Counter()

        for index in indices:

            categories[
                get_category(
                    records[index]
                )
            ] += 1

            levels[
                get_level(
                    records[index]
                )
            ] += 1

        group_objects.append(
            {
                "group_id": group_id,
                "size": len(indices),
                "indices": indices,
                "categories": dict(
                    categories
                ),
                "levels": dict(
                    levels
                )
            }
        )

    return group_objects


# ============================================================
# TARGET COUNTS
# ============================================================

def calculate_targets(
    total
):

    train_target = round(
        total * TRAIN_RATIO
    )

    validation_target = round(
        total * VALIDATION_RATIO
    )

    test_target = (
        total
        - train_target
        - validation_target
    )

    return {
        "train": train_target,
        "validation": validation_target,
        "test": test_target
    }


# ============================================================
# GROUP STRATIFICATION KEY
# ============================================================

def group_category_level_key(
    group
):

    # Use the dominant category and level
    # as the group's approximate stratum.

    category = max(
        group["categories"],
        key=group["categories"].get
    )

    level = max(
        group["levels"],
        key=group["levels"].get
    )

    return (
        category,
        level
    )


# ============================================================
# ASSIGN GROUPS TO SPLITS
# ============================================================

def assign_groups(
    records,
    group_objects
):

    print()
    print("=" * 70)
    print(
        "ASSIGNING GROUPS TO TARGET SPLITS"
    )
    print("=" * 70)

    total = len(records)

    targets = calculate_targets(
        total
    )

    print()
    print(
        "Target counts:"
    )

    print(
        f"  Train      : "
        f"{targets['train']:,}"
    )

    print(
        f"  Validation : "
        f"{targets['validation']:,}"
    )

    print(
        f"  Test       : "
        f"{targets['test']:,}"
    )

    # --------------------------------------------------------
    # Random generator
    # --------------------------------------------------------

    rng = random.Random(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Group order
    #
    # Large groups first prevents a large group from being
    # forced into a tiny split near the end.
    # --------------------------------------------------------

    groups = group_objects.copy()

    rng.shuffle(
        groups
    )

    groups.sort(
        key=lambda group: group["size"],
        reverse=True
    )

    # --------------------------------------------------------
    # Split state
    # --------------------------------------------------------

    split_indices = {
        "train": [],
        "validation": [],
        "test": []
    }

    split_counts = {
        "train": 0,
        "validation": 0,
        "test": 0
    }

    # --------------------------------------------------------
    # Category counters
    # --------------------------------------------------------

    category_counts = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter()
    }

    # --------------------------------------------------------
    # Overall category targets
    # --------------------------------------------------------

    overall_categories = Counter(
        get_category(record)
        for record in records
    )

    category_targets = {}

    for category, count in (
        overall_categories.items()
    ):

        category_targets[
            category
        ] = {
            "train": count * TRAIN_RATIO,
            "validation": count * VALIDATION_RATIO,
            "test": count * TEST_RATIO
        }

    # --------------------------------------------------------
    # Assignment function
    # --------------------------------------------------------

    for group in groups:

        size = group["size"]

        dominant_category = max(
            group["categories"],
            key=group["categories"].get
        )

        candidate_scores = {}

        for split_name in (
            "train",
            "validation",
            "test"
        ):

            current_count = split_counts[
                split_name
            ]

            target_count = targets[
                split_name
            ]

            new_count = (
                current_count
                + size
            )

            # ------------------------------------------------
            # Main size error
            # ------------------------------------------------

            size_error = abs(
                new_count
                - target_count
            ) / target_count

            # ------------------------------------------------
            # Category error
            # ------------------------------------------------

            current_category_count = (
                category_counts[
                    split_name
                ][dominant_category]
            )

            category_target = (
                category_targets[
                    dominant_category
                ][split_name]
            )

            new_category_count = (
                current_category_count
                + size
            )

            if category_target > 0:

                category_error = abs(
                    new_category_count
                    - category_target
                ) / category_target

            else:

                category_error = 0.0

            # ------------------------------------------------
            # Prevent validation/test from becoming too large
            # ------------------------------------------------

            overflow_penalty = 0.0

            if split_name == "validation":

                if new_count > (
                    targets["validation"]
                    + size
                ):

                    overflow_penalty = 2.0

            if split_name == "test":

                if new_count > (
                    targets["test"]
                    + size
                ):

                    overflow_penalty = 2.0

            # ------------------------------------------------
            # Weighted score
            # ------------------------------------------------

            score = (
                size_error * 3.0
                + category_error * 0.5
                + overflow_penalty
            )

            candidate_scores[
                split_name
            ] = score

        # ----------------------------------------------------
        # Choose best split
        # ----------------------------------------------------

        chosen_split = min(
            candidate_scores,
            key=candidate_scores.get
        )

        # ----------------------------------------------------
        # Assign group
        # ----------------------------------------------------

        split_indices[
            chosen_split
        ].extend(
            group["indices"]
        )

        split_counts[
            chosen_split
        ] += size

        for category, count in (
            group["categories"].items()
        ):

            category_counts[
                chosen_split
            ][category] += count

    # --------------------------------------------------------
    # Final shuffle
    # --------------------------------------------------------

    for split_name in split_indices:

        rng.shuffle(
            split_indices[
                split_name
            ]
        )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print(
        "Actual group-aware split:"
    )

    for split_name in (
        "train",
        "validation",
        "test"
    ):

        count = len(
            split_indices[
                split_name
            ]
        )

        percentage = (
            count / total * 100
        )

        print(
            f"  {split_name.capitalize():<12}"
            f"{count:>8,} "
            f"({percentage:6.2f}%)"
        )

    return split_indices


# ============================================================
# BUILD RECORDS
# ============================================================

def build_split_records(
    records,
    split_indices
):

    return {
        split_name: [
            records[index]
            for index in indices
        ]

        for split_name, indices
        in split_indices.items()
    }


# ============================================================
# VERIFY COUNTS
# ============================================================

def verify_record_counts(
    records,
    split_records
):

    original_count = len(records)

    combined_count = sum(
        len(split_records[name])
        for name in (
            "train",
            "validation",
            "test"
        )
    )

    print()
    print(
        "RECORD COUNT VERIFICATION"
    )

    print("-" * 70)

    print(
        f"Original : {original_count:,}"
    )

    print(
        f"Train    : "
        f"{len(split_records['train']):,}"
    )

    print(
        f"Validation: "
        f"{len(split_records['validation']):,}"
    )

    print(
        f"Test     : "
        f"{len(split_records['test']):,}"
    )

    print(
        f"Combined : {combined_count:,}"
    )

    if original_count != combined_count:

        raise RuntimeError(
            "Record count mismatch."
        )

    print()
    print(
        "PASS: No records were lost."
    )


# ============================================================
# VERIFY UNIQUE ASSIGNMENT
# ============================================================

def verify_unique_assignment(
    split_indices
):

    train = set(
        split_indices["train"]
    )

    validation = set(
        split_indices["validation"]
    )

    test = set(
        split_indices["test"]
    )

    train_validation = (
        train & validation
    )

    train_test = (
        train & test
    )

    validation_test = (
        validation & test
    )

    print()
    print(
        "SPLIT ASSIGNMENT VERIFICATION"
    )

    print("-" * 70)

    print(
        f"Train <-> Validation: "
        f"{len(train_validation)}"
    )

    print(
        f"Train <-> Test: "
        f"{len(train_test)}"
    )

    print(
        f"Validation <-> Test: "
        f"{len(validation_test)}"
    )

    if (
        train_validation
        or train_test
        or validation_test
    ):

        raise RuntimeError(
            "A record appears in multiple splits."
        )

    print()
    print(
        "PASS: Every record belongs to exactly "
        "one split."
    )


# ============================================================
# VERIFY GROUP INTEGRITY
# ============================================================

def verify_group_integrity(
    groups,
    split_indices
):

    split_lookup = {}

    for split_name, indices in (
        split_indices.items()
    ):

        for index in indices:

            split_lookup[index] = (
                split_name
            )

    violations = []

    for group in groups:

        assigned_splits = {
            split_lookup[index]
            for index in group["indices"]
        }

        if len(assigned_splits) > 1:

            violations.append(
                {
                    "group_id": group["group_id"],
                    "splits": sorted(
                        assigned_splits
                    )
                }
            )

    print()
    print(
        "GROUP LEAKAGE VERIFICATION"
    )

    print("-" * 70)

    print(
        f"Groups checked: "
        f"{len(groups):,}"
    )

    print(
        f"Groups crossing splits: "
        f"{len(violations):,}"
    )

    if violations:

        raise RuntimeError(
            "Near-duplicate groups crossed "
            "dataset splits."
        )

    print()
    print(
        "PASS: All groups remain inside "
        "one split."
    )


# ============================================================
# CATEGORY DISTRIBUTION
# ============================================================

def print_distribution(
    records,
    field,
    split_name
):

    counter = Counter()

    for record in records:

        if field == "category":

            value = get_category(
                record
            )

        else:

            value = get_level(
                record
            )

        counter[value] += 1

    total = len(records)

    print()
    print(
        f"{split_name.upper()} - "
        f"{field.upper()}"
    )

    print("-" * 70)

    for value, count in (
        counter.most_common()
    ):

        percentage = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"{value:<25}"
            f"{count:>8,} "
            f"({percentage:6.2f}%)"
        )


# ============================================================
# SAVE GROUP REPORT
# ============================================================

def save_group_report(
    group_objects
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    duplicate_groups = [
        group
        for group in group_objects
        if group["size"] > 1
    ]

    report = {
        "similarity_threshold": (
            SIMILARITY_THRESHOLD
        ),

        "ngram_range": [
            NGRAM_MIN,
            NGRAM_MAX
        ],

        "near_duplicate_groups": len(
            duplicate_groups
        ),

        "total_groups": len(
            group_objects
        ),

        "groups": group_objects
    }

    with GROUP_REPORT.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        f"Group report saved:"
    )

    print(
        f"  {GROUP_REPORT}"
    )


# ============================================================
# SAVE SPLIT REPORT
# ============================================================

def save_split_report(
    records,
    split_records,
    group_objects
):

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "master_dataset": str(
            MASTER_DATASET
        ),

        "master_records": len(
            records
        ),

        "split_sizes": {
            "train": len(
                split_records["train"]
            ),

            "validation": len(
                split_records["validation"]
            ),

            "test": len(
                split_records["test"]
            )
        },

        "target_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO
        },

        "random_seed": RANDOM_SEED,

        "similarity_threshold": (
            SIMILARITY_THRESHOLD
        ),

        "near_duplicate_groups": len(
            [
                group
                for group in group_objects
                if group["size"] > 1
            ]
        ),

        "total_groups": len(
            group_objects
        ),

        "master_dataset_modified": False
    }

    with SPLIT_REPORT.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(
        "Split report saved:"
    )

    print(
        f"  {SPLIT_REPORT}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "GROUP-AWARE 90/5/5 DATASET SPLITTER"
    )
    print("=" * 70)

    print()
    print(
        "MASTER DATASET WILL NOT BE MODIFIED."
    )

    print(
        f"Input : {MASTER_DATASET}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    validate_configuration()

    # --------------------------------------------------------
    # Check master dataset
    # --------------------------------------------------------

    if not MASTER_DATASET.exists():

        raise FileNotFoundError(
            f"Master dataset not found:\n"
            f"{MASTER_DATASET}"
        )

    # --------------------------------------------------------
    # Prevent accidental overwrite
    # --------------------------------------------------------

    existing_files = [
        path
        for path in (
            TRAIN_FILE,
            VALIDATION_FILE,
            TEST_FILE
        )
        if path.exists()
    ]

    if existing_files:

        print()
        print(
            "WARNING: Output files already exist:"
        )

        for path in existing_files:

            print(
                f"  {path}"
            )

        print()
        print(
            "Nothing was overwritten."
        )

        return

    # --------------------------------------------------------
    # Load master dataset
    # --------------------------------------------------------

    records = load_jsonl(
        MASTER_DATASET
    )

    if not records:

        raise ValueError(
            "Master dataset is empty."
        )

    # --------------------------------------------------------
    # Exact duplicates
    # --------------------------------------------------------

    print()
    print(
        "Checking exact duplicate questions..."
    )

    exact_groups = (
        find_exact_duplicate_groups(
            records
        )
    )

    print(
        f"Exact duplicate groups: "
        f"{len(exact_groups):,}"
    )

    # --------------------------------------------------------
    # Near duplicates
    # --------------------------------------------------------

    near_groups = (
        find_near_duplicate_groups(
            records
        )
    )

    # --------------------------------------------------------
    # Merge all duplicate relationships
    # --------------------------------------------------------

    final_groups = merge_duplicate_groups(
        exact_groups,
        near_groups,
        len(records)
    )

    # --------------------------------------------------------
    # Create group objects
    # --------------------------------------------------------

    group_objects = create_group_objects(
        records,
        final_groups
    )

    print()
    print(
        f"Total splitting groups: "
        f"{len(group_objects):,}"
    )

    # --------------------------------------------------------
    # Save group report
    # --------------------------------------------------------

    save_group_report(
        group_objects
    )

    # --------------------------------------------------------
    # Assign groups
    # --------------------------------------------------------

    split_indices = assign_groups(
        records,
        group_objects
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    verify_unique_assignment(
        split_indices
    )

    verify_group_integrity(
        group_objects,
        split_indices
    )

    # --------------------------------------------------------
    # Build records
    # --------------------------------------------------------

    split_records = build_split_records(
        records,
        split_indices
    )

    # --------------------------------------------------------
    # Verify counts
    # --------------------------------------------------------

    verify_record_counts(
        records,
        split_records
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
        "Saving group-aware datasets..."
    )

    save_jsonl(
        split_records["train"],
        TRAIN_FILE
    )

    save_jsonl(
        split_records["validation"],
        VALIDATION_FILE
    )

    save_jsonl(
        split_records["test"],
        TEST_FILE
    )

    # --------------------------------------------------------
    # Print distributions
    # --------------------------------------------------------

    for split_name in (
        "train",
        "validation",
        "test"
    ):

        print_distribution(
            split_records[split_name],
            "category",
            split_name
        )

        print_distribution(
            split_records[split_name],
            "level",
            split_name
        )

    # --------------------------------------------------------
    # Save split report
    # --------------------------------------------------------

    save_split_report(
        records,
        split_records,
        group_objects
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "GROUP-AWARE SPLIT COMPLETED"
    )
    print("=" * 70)

    print()
    print(
        "Created:"
    )

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
        "Master dataset:"
    )

    print(
        f"  {MASTER_DATASET}"
    )

    print()
    print(
        "Master dataset was NOT modified."
    )

    print()
    print(
        "Next: run the leakage audit against "
        "the v2 grouped splits."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()