# task type can be either 'classification' or 'regression', based on the target feature in the dataset
TASK_TYPE = ''  # classification | regression

# list of all the columns (header) of the input data file(s)
HEADER = []

# list of the default values of all the columns of the input data, to help decoding the data types of the columns
HEADER_DEFAULTS = []

# list of the feature names of type int or float
NUMERIC_FEATURE_NAMES = []

# a dictionary of feature names with int values, but to be treated as categorical features.
# In the dictionary, the key is the feature name, and the value is the num_buckets (count of distinct values)
CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY = {}

# a dictionary of categorical features with few nominal values (to be encoded as one-hot indicators)
#  In the dictionary, the key is the feature name, and the value is the list of feature vocabulary
CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {}

# a dictionary of categorical features with many values (sparse features)
# In the dictionary, the key is the feature name, and the value is the bucket size
CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET = {}

# list of all the categorical feature names
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY.keys()) \
                            + list(CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys()) \
                            + list(CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET.keys()) \

# list of all the input feature names to be used in the model
FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES

# target feature name (response or class variable)
TARGET_NAME = ''

# list of the class values (labels) in a classification dataset
TARGET_LABELS = []

# list column to be ignores (e.g. keys, constants, etc.)
UNUSED_FEATURE_NAMES = list(set(HEADER) - set(FEATURE_NAMES) - {TARGET_NAME})
