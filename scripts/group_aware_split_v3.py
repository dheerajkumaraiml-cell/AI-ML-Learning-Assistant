import json
import random
import re
import sys
from collections import Counter, defaultdict
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

MASTER_DATASET = Path(
    "dataset/merged_dataset.jsonl"
)

OUTPUT_DIR = Path(
    "dataset/splits_grouped_v3"
)

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALIDATION_FILE = OUTPUT_DIR / "validation.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"

REPORT_DIR = Path(
    "dataset/leakage_reports"
)

GROUP_REPORT = (
    REPORT_DIR / "duplicate_groups_v3.json"
)

SPLIT_REPORT = (
    REPORT_DIR / "group_aware_split_report_v3.json"
)


# ============================================================
# TARGET SPLITS
# ============================================================

TRAIN_RATIO = 0.90
VALIDATION_RATIO = 0.05
TEST_RATIO = 0.05

RANDOM_SEED = 42


# ============================================================
# SIMILARITY SETTINGS
# ============================================================

SIMILARITY_THRESHOLD = 0.90

NGRAM_MIN = 3
NGRAM_MAX = 5

NEIGHBORS_PER_RECORD = 5

MIN_QUESTION_LENGTH = 20


# ============================================================
# LOAD DATASET
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

    return value if value else "Other"


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

    return value if value else "Unknown"


# ============================================================
# EXACT DUPLICATES
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
# UNION-FIND
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
# FIND NEAR-DUPLICATE GROUPS
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

    valid_questions = [
        questions[index]
        for index in valid_indices
    ]

    if not valid_questions:

        return []

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
    # Nearest neighbours
    # --------------------------------------------------------

    print(
        "Searching for similar questions..."
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
    # Create similarity connections
    # --------------------------------------------------------

    for row in range(
        len(valid_indices)
    ):

        source_index = valid_indices[
            row
        ]

        for position in range(
            len(neighbours[row])
        ):

            target_row = neighbours[
                row
            ][position]

            target_index = valid_indices[
                target_row
            ]

            if source_index == target_index:
                continue

            distance = distances[
                row
            ][position]

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
    # Connected components
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
                sorted(indices)
            )

    print(
        f"Near-duplicate groups: "
        f"{len(groups):,}"
    )

    return groups


# ============================================================
# MERGE EXACT + NEAR DUPLICATE GROUPS
# ============================================================

def merge_groups(
    exact_groups,
    near_groups,
    record_count
):

    print()
    print(
        "Merging duplicate relationships..."
    )

    union_find = UnionFind(
        record_count
    )

    # Exact groups
    for group in exact_groups:

        indices = group["indices"]

        first = indices[0]

        for index in indices[1:]:

            union_find.union(
                first,
                index
            )

    # Near groups
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

    final_groups = [
        sorted(indices)
        for indices in components.values()
    ]

    final_groups.sort(
        key=lambda group: (
            -len(group),
            group[0]
        )
    )

    print(
        f"Final groups: "
        f"{len(final_groups):,}"
    )

    multi_record_groups = sum(
        1
        for group in final_groups
        if len(group) > 1
    )

    print(
        f"Groups containing multiple records: "
        f"{multi_record_groups:,}"
    )

    return final_groups


# ============================================================
# GROUP OBJECTS
# ============================================================

def create_group_objects(
    records,
    groups
):

    objects = []

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

        objects.append(
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

    return objects


# ============================================================
# TARGET COUNTS
# ============================================================

def calculate_targets(total):

    train = round(
        total * TRAIN_RATIO
    )

    validation = round(
        total * VALIDATION_RATIO
    )

    test = (
        total
        - train
        - validation
    )

    return {
        "train": train,
        "validation": validation,
        "test": test
    }


# ============================================================
# GROUP-AWARE ALLOCATION
#
# IMPORTANT:
#
# We allocate TRAIN first because it is the largest split.
#
# Validation and Test are then balanced against each other.
#
# This prevents the previous 85/7/7 drift.
# ============================================================

def assign_groups(
    group_objects,
    total_records
):

    print()
    print("=" * 70)
    print(
        "ALLOCATING GROUPS USING 90/5/5 TARGETS"
    )
    print("=" * 70)

    targets = calculate_targets(
        total_records
    )

    print()
    print(
        f"Train target      : {targets['train']:,}"
    )

    print(
        f"Validation target : {targets['validation']:,}"
    )

    print(
        f"Test target       : {targets['test']:,}"
    )

    rng = random.Random(
        RANDOM_SEED
    )

    groups = group_objects.copy()

    # --------------------------------------------------------
    # Shuffle first for reproducibility.
    # --------------------------------------------------------

    rng.shuffle(
        groups
    )

    # --------------------------------------------------------
    # Sort largest groups first.
    # --------------------------------------------------------

    groups.sort(
        key=lambda group: group["size"],
        reverse=True
    )

    # --------------------------------------------------------
    # Initial containers.
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
    # CATEGORY COUNTS
    # --------------------------------------------------------

    category_counts = {
        "train": Counter(),
        "validation": Counter(),
        "test": Counter()
    }

    overall_categories = Counter()

    for group in groups:

        for category, count in (
            group["categories"].items()
        ):

            overall_categories[
                category
            ] += count

    # --------------------------------------------------------
    # Category targets
    # --------------------------------------------------------

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

    # ========================================================
    # ALLOCATION
    # ========================================================

    for group in groups:

        size = group["size"]

        dominant_category = max(
            group["categories"],
            key=group["categories"].get
        )

        # ----------------------------------------------------
        # Determine which split should receive this group.
        #
        # The score represents how bad the allocation would be.
        # Lower is better.
        # ----------------------------------------------------

        scores = {}

        for split_name in (
            "train",
            "validation",
            "test"
        ):

            current = split_counts[
                split_name
            ]

            target = targets[
                split_name
            ]

            new_count = (
                current + size
            )

            # ------------------------------------------------
            # Size error
            # ------------------------------------------------

            size_error = abs(
                new_count - target
            ) / target

            # ------------------------------------------------
            # Category error
            # ------------------------------------------------

            current_category = (
                category_counts[
                    split_name
                ][dominant_category]
            )

            new_category = (
                current_category + size
            )

            category_target = (
                category_targets[
                    dominant_category
                ][split_name]
            )

            if category_target > 0:

                category_error = abs(
                    new_category
                    - category_target
                ) / category_target

            else:

                category_error = 0.0

            # ------------------------------------------------
            # OVERFLOW penalty
            #
            # Validation/Test are small.
            # We strongly discourage filling them too early.
            # ------------------------------------------------

            overflow = 0.0

            if split_name in (
                "validation",
                "test"
            ):

                if new_count > (
                    target + size
                ):

                    overflow = 10.0

            # ------------------------------------------------
            # Train preference
            #
            # When train is below 90%, favor it.
            # ------------------------------------------------

            train_bonus = 0.0

            if split_name == "train":

                if current < (
                    targets["train"]
                    - size
                ):

                    train_bonus = -0.25

            # ------------------------------------------------
            # Final score
            # ------------------------------------------------

            scores[
                split_name
            ] = (
                size_error * 5.0
                + category_error * 0.25
                + overflow
                + train_bonus
            )

        chosen_split = min(
            scores,
            key=scores.get
        )

        # ----------------------------------------------------
        # Assign
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
    # Shuffle records
    # --------------------------------------------------------

    for split_name in split_indices:

        rng.shuffle(
            split_indices[
                split_name
            ]
        )

    # --------------------------------------------------------
    # Print actual sizes
    # --------------------------------------------------------

    print()
    print(
        "FINAL GROUP-AWARE SPLIT"
    )

    print("-" * 70)

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
            count
            / total_records
            * 100
        )

        print(
            f"{split_name.capitalize():<12}"
            f"{count:>8,}"
            f"  ({percentage:6.2f}%)"
        )

    return split_indices


# ============================================================
# BUILD RECORDS
# ============================================================

def build_split_records(
    records,
    split_indices
):

    result = {}

    for split_name, indices in (
        split_indices.items()
    ):

        result[
            split_name
        ] = [
            records[index]
            for index in indices
        ]

    return result


# ============================================================
# VERIFY COUNTS
# ============================================================

def verify_counts(
    records,
    split_records
):

    original = len(records)

    combined = sum(
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
        f"Original : {original:,}"
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
        f"Combined : {combined:,}"
    )

    if original != combined:

        raise RuntimeError(
            "ERROR: Some records were lost."
        )

    print()
    print(
        "PASS: All records accounted for."
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

    tv = train & validation
    tt = train & test
    vt = validation & test

    print()
    print(
        "SPLIT ASSIGNMENT VERIFICATION"
    )

    print("-" * 70)

    print(
        f"Train <-> Validation: {len(tv)}"
    )

    print(
        f"Train <-> Test:       {len(tt)}"
    )

    print(
        f"Validation <-> Test:  {len(vt)}"
    )

    if tv or tt or vt:

        raise RuntimeError(
            "ERROR: A record appears in multiple splits."
        )

    print()
    print(
        "PASS: Every record belongs to exactly one split."
    )


# ============================================================
# VERIFY GROUP INTEGRITY
# ============================================================

def verify_group_integrity(
    groups,
    split_indices
):

    lookup = {}

    for split_name, indices in (
        split_indices.items()
    ):

        for index in indices:

            lookup[index] = split_name

    violations = []

    for group in groups:

        assigned_splits = {
            lookup[index]
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
            "ERROR: Near-duplicate groups "
            "crossed splits."
        )

    print()
    print(
        "PASS: No near-duplicate group "
        "crosses splits."
    )


# ============================================================
# DISTRIBUTION
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
        )

        print(
            f"{value:<25}"
            f"{count:>8,}"
            f" ({percentage:6.2f}%)"
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

    report = {
        "similarity_threshold":
            SIMILARITY_THRESHOLD,

        "ngram_range": [
            NGRAM_MIN,
            NGRAM_MAX
        ],

        "near_duplicate_groups":
            sum(
                1
                for group in group_objects
                if group["size"] > 1
            ),

        "total_groups":
            len(group_objects),

        "groups":
            group_objects
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

        "master_dataset":
            str(MASTER_DATASET),

        "master_records":
            len(records),

        "split_sizes": {

            "train":
                len(split_records["train"]),

            "validation":
                len(split_records["validation"]),

            "test":
                len(split_records["test"])
        },

        "target_ratios": {

            "train":
                TRAIN_RATIO,

            "validation":
                VALIDATION_RATIO,

            "test":
                TEST_RATIO
        },

        "random_seed":
            RANDOM_SEED,

        "similarity_threshold":
            SIMILARITY_THRESHOLD,

        "near_duplicate_groups":
            sum(
                1
                for group in group_objects
                if group["size"] > 1
            ),

        "total_groups":
            len(group_objects),

        "master_dataset_modified":
            False
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
        f"Split report saved:"
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
        "GROUP-AWARE 90/5/5 SPLITTER V3"
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
    # Check input
    # --------------------------------------------------------

    if not MASTER_DATASET.exists():

        raise FileNotFoundError(
            f"Master dataset not found:\n"
            f"{MASTER_DATASET}"
        )

    # --------------------------------------------------------
    # Protect existing output
    # --------------------------------------------------------

    existing = [
        path
        for path in (
            TRAIN_FILE,
            VALIDATION_FILE,
            TEST_FILE
        )
        if path.exists()
    ]

    if existing:

        print()
        print(
            "WARNING: v3 output files already exist:"
        )

        for path in existing:

            print(
                f"  {path}"
            )

        print()
        print(
            "Nothing was overwritten."
        )

        return

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    records = load_jsonl(
        MASTER_DATASET
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
    # Merge
    # --------------------------------------------------------

    final_groups = merge_groups(
        exact_groups,
        near_groups,
        len(records)
    )

    # --------------------------------------------------------
    # Group objects
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
    # Assign
    # --------------------------------------------------------

    split_indices = assign_groups(
        group_objects,
        len(records)
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
    # Count verification
    # --------------------------------------------------------

    verify_counts(
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
    # Save
    # --------------------------------------------------------

    print()
    print(
        "Saving v3 datasets..."
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
    # Distribution
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
    # Report
    # --------------------------------------------------------

    save_split_report(
        records,
        split_records,
        group_objects
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "V3 GROUP-AWARE SPLIT COMPLETED"
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
        "Master dataset remains unchanged:"
    )

    print(
        f"  {MASTER_DATASET}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "Run the leakage audit on v3 before "
        "freezing the split."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()