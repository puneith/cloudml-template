TASK_TYPE = ''

HEADERS = []

HEADER_DEFAULTS = []

NUMERIC_FEATURE_NAMES = []

# categorical features with few values
CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {}

# categorical features with many values
CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET = {}

CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys()) \
                            + list(CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET.keys())

FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES


TARGET_NAME = ''

TARGET_LABELS = []

UNUSED_FEATURE_NAMES = []