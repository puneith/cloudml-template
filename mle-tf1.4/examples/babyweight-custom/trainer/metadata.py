# ************************************************************************
# YOU NEED TO MODIFY THE META DATA TO ADAPT THE TRAINER TEMPLATE YOUR DATA
# ************************************************************************

# task type can be either 'classification' or 'regression', based on the target feature in the dataset
TASK_TYPE = 'custom'

# list of all the columns (header) of the input data file(s)
HEADER = ['weight_pounds', 'is_male', 'mother_age', 'mother_race', 'plurality', 'gestation_weeks',
          'mother_married', 'cigarette_use', 'alcohol_use', 'key']

# list of the default values of all the columns of the input data, to help decoding the data types of the columns
HEADER_DEFAULTS = [[0.0], ['null'], [0.0], ['null'], [0.0], [0.0], ['null'], ['null'], ['null'], ['nokey']]

# list of the feature names of type int or float
INPUT_NUMERIC_FEATURE_NAMES = ['mother_age', 'plurality', 'gestation_weeks']

# numeric features constructed, if any, in process_features function in input.py module,
# as part of reading data
CONSTRUCTED_NUMERIC_FEATURE_NAMES = []

# a dictionary of feature names with int values, but to be treated as categorical features.
# In the dictionary, the key is the feature name, and the value is the num_buckets (count of distinct values)
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY = {}

# categorical features with identity constructed, if any, in process_features function in input.py module,
# as part of reading data. Usually include constructed boolean flags
CONSTRUCTED_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY = {}

# a dictionary of categorical features with few nominal values (to be encoded as one-hot indicators)
#  In the dictionary, the key is the feature name, and the value is the list of feature vocabulary
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {
    'mother_race': ['White', 'Black', 'American Indian', 'Chinese', 'Japanese', 'Hawaiian',
                    'Filipino', 'Unknown', 'Asian Indian', 'Korean', 'Samaon', 'Vietnamese'],
    'is_male': ['True', 'False'],
    'mother_married': ['True', 'False'],
    'cigarette_use': ['True', 'False', 'None'],
    'alcohol_use': ['True', 'False', 'None']
}

# a dictionary of categorical features with many values (sparse features)
# In the dictionary, the key is the feature name, and the value is the bucket size
INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET = {}

# list of all the categorical feature names
INPUT_CATEGORICAL_FEATURE_NAMES = list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_IDENTITY.keys()) \
                                  + list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY.keys()) \
                                  + list(INPUT_CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET.keys())

# list of all the input feature names to be used in the model
INPUT_FEATURE_NAMES = INPUT_NUMERIC_FEATURE_NAMES + INPUT_CATEGORICAL_FEATURE_NAMES

# the column include the weight of each record
WEIGHT_COLUMN_NAME = None

# target feature name (response or class variable)
TARGET_NAME = 'weight_pounds'

# list of the class values (labels) in a classification dataset
TARGET_LABELS = []

# list of the columns expected during serving (which probably different than the header of the training data)
SERVING_COLUMNS = ['is_male', 'mother_age', 'mother_race', 'plurality', 'gestation_weeks',
                   'mother_married', 'cigarette_use', 'alcohol_use']

# list of the default values of all the columns of the serving data, to help decoding the data types of the columns
SERVING_DEFAULTS = [['null'], [0.0], ['null'], [0.0], [0.0], ['null'], ['null'], ['null']]
