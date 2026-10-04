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

OUTPUT_DIR = Path(
    "dataset/splits_final_candidate"
)

TRAIN_FILE = OUTPUT_DIR / "train.jsonl"
VALIDATION_FILE = OUTPUT_DIR / "validation.jsonl"
TEST_FILE = OUTPUT_DIR / "test.jsonl"

REPORT_DIR = Path(
    "dataset/leakage_reports"
)

GROUP_REPORT = (
    REPORT_DIR / "stratified_groups.json"
)

SPLIT_REPORT = (
    REPORT_DIR / "group_stratified_split_report.json"
)


# ============================================================
# SPLIT RATIOS
# ============================================================

TRAIN_RATIO = 0.90
VALIDATION_RATIO = 0.05
TEST_RATIO = 0.05

RANDOM_SEED = 42


# ============================================================
# NEAR-DUPLICATE SETTINGS
# ============================================================

SIMILARITY_THRESHOLD = 0.90

NGRAM_MIN = 3
NGRAM_MAX = 5

NEIGHBORS_PER_RECORD = 5

MIN_QUESTION_LENGTH = 20


# ============================================================
# LOAD JSONL
# ============================================================

def load_jsonl(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    records = []

    print()
    print(f"Loading dataset:")
    print(f"  {path}")

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
# GET CATEGORY
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
# GET LEVEL
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
# FIND EXACT DUPLICATES
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
                indices
            )

    return groups


# ============================================================
# FIND NEAR DUPLICATES
# ============================================================

def find_near_duplicate_groups(records):

    print()
    print("=" * 70)
    print(
        "NEAR-DUPLICATE DETECTION"
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
        f"Questions analyzed: "
        f"{len(valid_indices):,}"
    )

    if not valid_indices:

        return []

    valid_questions = [
        questions[index]
        for index in valid_indices
    ]

    print(
        "Creating TF-IDF matrix..."
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

    print(
        "Finding nearest neighbours..."
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
        model.kneighbors(matrix)
    )

    union_find = UnionFind(
        len(records)
    )

    connections = 0

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
# MERGE GROUP RELATIONSHIPS
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

    for group in exact_groups:

        first = group[0]

        for index in group[1:]:

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

    multi_record_groups = sum(
        1
        for group in final_groups
        if len(group) > 1
    )

    print(
        f"Final groups: "
        f"{len(final_groups):,}"
    )

    print(
        f"Multi-record groups: "
        f"{multi_record_groups:,}"
    )

    return final_groups


# ============================================================
# CREATE GROUP OBJECTS
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

        category_counter = Counter()

        level_counter = Counter()

        for index in indices:

            category_counter[
                get_category(
                    records[index]
                )
            ] += 1

            level_counter[
                get_level(
                    records[index]
                )
            ] += 1

        dominant_category = (
            category_counter.most_common(1)[0][0]
        )

        dominant_level = (
            level_counter.most_common(1)[0][0]
        )

        objects.append(
            {
                "group_id": group_id,
                "indices": indices,
                "size": len(indices),
                "category": dominant_category,
                "level": dominant_level,
                "categories": dict(
                    category_counter
                ),
                "levels": dict(
                    level_counter
                )
            }
        )

    return objects


# ============================================================
# CREATE STRATA
#
# Stratum = category + level
# ============================================================

def create_strata(
    group_objects
):

    strata = defaultdict(list)

    for group in group_objects:

        key = (
            group["category"],
            group["level"]
        )

        strata[key].append(
            group
        )

    return strata


# ============================================================
# ALLOCATE GROUPS WITHIN ONE STRATUM
#
# This function uses a deterministic quota method.
#
# For each stratum:
#
#     ~90% -> train
#     ~5%  -> validation
#     ~5%  -> test
#
# Groups are never divided.
# ============================================================

def allocate_stratum(
    groups,
    rng
):

    # --------------------------------------------------------
    # Randomize order first.
    # --------------------------------------------------------

    groups = groups.copy()

    rng.shuffle(
        groups
    )

    # --------------------------------------------------------
    # Larger groups first.
    # --------------------------------------------------------

    groups.sort(
        key=lambda group: group["size"],
        reverse=True
    )

    total = sum(
        group["size"]
        for group in groups
    )

    if total == 0:

        return {
            "train": [],
            "validation": [],
            "test": []
        }

    # --------------------------------------------------------
    # Desired counts.
    # --------------------------------------------------------

    desired = {
        "train": total * TRAIN_RATIO,
        "validation": total * VALIDATION_RATIO,
        "test": total * TEST_RATIO
    }

    current = {
        "train": 0,
        "validation": 0,
        "test": 0
    }

    result = {
        "train": [],
        "validation": [],
        "test": []
    }

    # --------------------------------------------------------
    # Assign each group to the split whose current count
    # is furthest below its desired proportion.
    #
    # Importantly, this works independently inside each
    # category+level stratum.
    # --------------------------------------------------------

    for group in groups:

        size = group["size"]

        scores = {}

        for split_name in (
            "train",
            "validation",
            "test"
        ):

            desired_count = desired[
                split_name
            ]

            current_count = current[
                split_name
            ]

            # Positive means this split is under target.
            deficit = (
                desired_count
                - current_count
            )

            # Normalize so Train does not automatically
            # win just because it is larger.
            normalized_deficit = (
                deficit / desired_count
                if desired_count > 0
                else 0
            )

            scores[
                split_name
            ] = normalized_deficit

        chosen_split = max(
            scores,
            key=scores.get
        )

        result[
            chosen_split
        ].append(
            group
        )

        current[
            chosen_split
        ] += size

    return result


# ============================================================
# IMPROVE GLOBAL SPLIT SIZE
#
# After stratified allocation, move whole groups between
# splits if doing so makes the global 90/5/5 ratio better.
#
# We only move whole groups.
# ============================================================

def improve_global_allocation(
    split_groups,
    targets
):

    def total_size(groups):

        return sum(
            group["size"]
            for group in groups
        )

    def calculate_error(
        counts
    ):

        return sum(
            abs(
                counts[name]
                - targets[name]
            )
            for name in (
                "train",
                "validation",
                "test"
            )
        )

    # --------------------------------------------------------
    # Repeat several passes.
    # --------------------------------------------------------

    for _ in range(20):

        counts = {
            name: total_size(
                split_groups[name]
            )
            for name in (
                "train",
                "validation",
                "test"
            )
        }

        current_error = calculate_error(
            counts
        )

        best_move = None
        best_error = current_error

        # ----------------------------------------------------
        # Try moving one complete group from one split to
        # another.
        # ----------------------------------------------------

        for source in (
            "train",
            "validation",
            "test"
        ):

            for target in (
                "train",
                "validation",
                "test"
            ):

                if source == target:
                    continue

                for position, group in enumerate(
                    split_groups[source]
                ):

                    size = group["size"]

                    new_counts = counts.copy()

                    new_counts[source] -= size
                    new_counts[target] += size

                    new_error = calculate_error(
                        new_counts
                    )

                    if new_error < best_error:

                        best_error = new_error

                        best_move = (
                            source,
                            target,
                            position
                        )

        # ----------------------------------------------------
        # No improvement.
        # ----------------------------------------------------

        if best_move is None:
            break

        source, target, position = (
            best_move
        )

        group = split_groups[
            source
        ].pop(position)

        split_groups[
            target
        ].append(group)

    return split_groups


# ============================================================
# GLOBAL SPLIT
# ============================================================

def create_group_stratified_split(
    records,
    group_objects
):

    print()
    print("=" * 70)
    print(
        "GROUP-STRATIFIED ALLOCATION"
    )
    print("=" * 70)

    total_records = len(records)

    targets = {
        "train": round(
            total_records * TRAIN_RATIO
        ),

        "validation": round(
            total_records * VALIDATION_RATIO
        )
    }

    targets["test"] = (
        total_records
        - targets["train"]
        - targets["validation"]
    )

    print()
    print(
        "Target record counts:"
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

    rng = random.Random(
        RANDOM_SEED
    )

    strata = create_strata(
        group_objects
    )

    print()
    print(
        f"Strata: {len(strata)}"
    )

    # --------------------------------------------------------
    # Allocate every stratum.
    # --------------------------------------------------------

    split_groups = {
        "train": [],
        "validation": [],
        "test": []
    }

    for stratum_key, groups in strata.items():

        allocated = allocate_stratum(
            groups,
            rng
        )

        for split_name in (
            "train",
            "validation",
            "test"
        ):

            split_groups[
                split_name
            ].extend(
                allocated[split_name]
            )

    # --------------------------------------------------------
    # Global improvement.
    # --------------------------------------------------------

    split_groups = improve_global_allocation(
        split_groups,
        targets
    )

    # --------------------------------------------------------
    # Convert groups to record indices.
    # --------------------------------------------------------

    split_indices = {
        "train": [],
        "validation": [],
        "test": []
    }

    for split_name in (
        "train",
        "validation",
        "test"
    ):

        for group in split_groups[
            split_name
        ]:

            split_indices[
                split_name
            ].extend(
                group["indices"]
            )

    # --------------------------------------------------------
    # Shuffle.
    # --------------------------------------------------------

    for split_name in split_indices:

        rng.shuffle(
            split_indices[
                split_name
            ]
        )

    # --------------------------------------------------------
    # Print.
    # --------------------------------------------------------

    print()
    print(
        "FINAL SPLIT SIZES"
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
            f" ({percentage:6.2f}%)"
        )

    return split_indices


# ============================================================
# BUILD RECORDS
# ============================================================

def build_records(
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
            "Record count mismatch."
        )

    print()
    print(
        "PASS: All records are accounted for."
    )


# ============================================================
# VERIFY UNIQUE RECORD ASSIGNMENT
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
            "Record assignment overlap detected."
        )

    print()
    print(
        "PASS: Every record belongs to one split."
    )


# ============================================================
# VERIFY GROUP INTEGRITY
# ============================================================

def verify_group_integrity(
    group_objects,
    split_indices
):

    lookup = {}

    for split_name, indices in (
        split_indices.items()
    ):

        for index in indices:

            lookup[index] = split_name

    violations = []

    for group in group_objects:

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
        f"{len(group_objects):,}"
    )

    print(
        f"Groups crossing splits: "
        f"{len(violations):,}"
    )

    if violations:

        raise RuntimeError(
            "Near-duplicate groups crossed splits."
        )

    print()
    print(
        "PASS: No near-duplicate group "
        "crosses splits."
    )


# ============================================================
# DISTRIBUTION REPORT
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
            count
            / total
            * 100
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
        "master_dataset": str(
            MASTER_DATASET
        ),

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
        f"Group report:"
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
        f"Split report:"
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
        "GROUP-STRATIFIED LEAKAGE-SAFE SPLITTER"
    )
    print("=" * 70)

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "merged_dataset.jsonl will NOT be modified."
    )

    print(
        f"Input : {MASTER_DATASET}"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    # --------------------------------------------------------
    # Prevent accidental overwrite.
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
            "WARNING: Final candidate files already exist:"
        )

        for path in existing:

            print(
                f"  {path}"
            )

        print()
        print(
            "No files were overwritten."
        )

        return

    # --------------------------------------------------------
    # Load.
    # --------------------------------------------------------

    records = load_jsonl(
        MASTER_DATASET
    )

    if not records:

        raise ValueError(
            "Dataset is empty."
        )

    # --------------------------------------------------------
    # Exact duplicates.
    # --------------------------------------------------------

    print()
    print(
        "Checking exact duplicates..."
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
    # Near duplicates.
    # --------------------------------------------------------

    near_groups = (
        find_near_duplicate_groups(
            records
        )
    )

    # --------------------------------------------------------
    # Merge.
    # --------------------------------------------------------

    final_groups = merge_groups(
        exact_groups,
        near_groups,
        len(records)
    )

    # --------------------------------------------------------
    # Group objects.
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
    # Save group report.
    # --------------------------------------------------------

    save_group_report(
        group_objects
    )

    # --------------------------------------------------------
    # Stratified allocation.
    # --------------------------------------------------------

    split_indices = (
        create_group_stratified_split(
            records,
            group_objects
        )
    )

    # --------------------------------------------------------
    # Verify assignment.
    # --------------------------------------------------------

    verify_unique_assignment(
        split_indices
    )

    verify_group_integrity(
        group_objects,
        split_indices
    )

    # --------------------------------------------------------
    # Build records.
    # --------------------------------------------------------

    split_records = build_records(
        records,
        split_indices
    )

    # --------------------------------------------------------
    # Verify counts.
    # --------------------------------------------------------

    verify_counts(
        records,
        split_records
    )

    # --------------------------------------------------------
    # Create output.
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    print()
    print(
        "Saving final candidate splits..."
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
    # Print distributions.
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
    # Save report.
    # --------------------------------------------------------

    save_split_report(
        records,
        split_records,
        group_objects
    )

    # --------------------------------------------------------
    # Final summary.
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FINAL CANDIDATE SPLIT CREATED"
    )
    print("=" * 70)

    print()
    print(
        "Files:"
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
        "Master dataset was NOT modified:"
    )

    print(
        f"  {MASTER_DATASET}"
    )

    print()
    print(
        "NEXT STEP:"
    )

    print(
        "Run the deep leakage audit against "
        "splits_final_candidate."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()