import tensorflow as tf
import metadata
from tensorflow.python.feature_column import feature_column


def extend_feature_columns(feature_columns):
    """ Use to define additional feature columns, such as bucketized_column and crossed_column
    Default behaviour is to return the original feature_column list as is

    Args:
        feature_columns: [tf.feature_column] - list of base feature_columns to be extended
    Returns:
        [tf.feature_column]: extended feature_column list
    """

    # examples - given:
    # 'x' and 'y' are two numeric features:
    # 'alpha' and 'beta' are two categorical features

    # feature_columns['alpha_X_beta'] = tf.feature_column.crossed_column(
    #     [feature_columns['alpha'], feature_columns['beta']], 4)
    #
    # num_buckets = parameters.HYPER_PARAMS.num_buckets
    # buckets = np.linspace(-2, 2, num_buckets).tolist()
    #
    # feature_columns['x_bucketized'] = tf.feature_column.bucketized_column(
    #     feature_columns['x'], buckets)
    #
    # feature_columns['y_bucketized'] = tf.feature_column.bucketized_column(
    #     feature_columns['y'], buckets)
    #
    # feature_columns['x_bucketized_X_y_bucketized'] = tf.feature_column.crossed_column(
    #     [feature_columns['x_bucketized'], feature_columns['y_bucketized']], int(1e4))

    # note that, extensions can be applied on features constructed in process_features function

    return feature_columns


def create_feature_columns():
    """Creates tensorFlow feature_column definitions based on the metadata of the features.

    the tensorFlow feature_column objects are created based on the data types of the features
    defined in the metadata.py module. Extended featured (if any) are created, based on the base features,
    as the extend_feature_columns method is called.

    Returns:
      {string: tf.feature_column}: dictionary of name:feature_column .
    """

    # numeric features constructed, if any, in process_features function in input.py module,
    # as part of reading data
    CONSTRUCTED_NUMERIC_FEATURES_NAMES = []
    all_numeric_feature_names = metadata.NUMERIC_FEATURE_NAMES + CONSTRUCTED_NUMERIC_FEATURES_NAMES

    numeric_columns = {feature_name: tf.feature_column.numeric_column(feature_name)
                       for feature_name in all_numeric_feature_names}

    # categorical features with vocabulary constructed, if any, in process_features function in input.py module,
    # as part of reading data
    CONSTRUCTED_CATEGORICAL_FEATURES_NAMES_WITH_VOCABULARY = {}
    all_categorical_feature_names_with_vocabulary = \
        CONSTRUCTED_CATEGORICAL_FEATURES_NAMES_WITH_VOCABULARY.update(metadata.CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY)

    categorical_column_with_vocabulary = \
        {item[0]: tf.feature_column.categorical_column_with_vocabulary_list(item[0], item[1])
         for item in all_categorical_feature_names_with_vocabulary.items()}

    # categorical features with hash buckets constructed, if any, in process_features function in input.py module,
    # as part of reading data
    CONSTRUCTED_CATEGORICAL_FEATURES_NAMES_WITH_HASH_BUCKETS = {}
    all_categorical_feature_names_with_hash_buckets= \
        CONSTRUCTED_CATEGORICAL_FEATURES_NAMES_WITH_HASH_BUCKETS.update(metadata.CATEGORICAL_FEATURE_NAMES_WITH_HASH_BUCKET)

    categorical_column_with_hash_bucket = \
        {item[0]: tf.feature_column.categorical_column_with_hash_bucket(item[0], item[1])
         for item in all_categorical_feature_names_with_hash_buckets.items()}

    feature_columns = {}

    if numeric_columns is not None:
        feature_columns.update(numeric_columns)

    if categorical_column_with_vocabulary is not None:
        feature_columns.update(categorical_column_with_vocabulary)

    if categorical_column_with_hash_bucket is not None:
        feature_columns.update(categorical_column_with_hash_bucket)

    # add extended feature definitions before returning the feature_columns list
    return extend_feature_columns(feature_columns)


def get_deep_and_wide_columns(feature_columns, use_indicators=True):
    """Creates deep and wide feature column lists.

    given a list of feature_column, each feature_column is categorised as either:
    1) dense, if the column is tf.feature_column._NumericColumn or feature_column._EmbeddingColumn,
    2) categorical, if the column is tf.feature_column._VocabularyListCategoricalColumn or
    tf.feature_column._BucketizedColumn, or
    3) sparse, if the column is tf.feature_column._HashedCategoricalColumn or tf.feature_column._CrossedColumn.

    if use_indicators=True, then  categorical_columns are converted into indicator_column. and used as dense feature
    in the deep part of the model

    deep_columns = dense_columns + indicator_columns
    wide_columns = categorical_columns + sparse_columns

    Args:
        feature_columns: [tf.feature_column] - A list of tf.feature_column objects.
        use_indicators: bool - if True, then categorical_columns are converted into tf.feature_column.indicator_column

    Returns:
        [tf.feature_column],[tf.feature_column]: deep and wide feature_column lists.
    """
    dense_columns = list(
        filter(lambda column: isinstance(column, feature_column._NumericColumn) |
                              isinstance(column, feature_column._EmbeddingColumn),
               feature_columns
        )
    )

    categorical_columns = list(
        filter(lambda column: isinstance(column,feature_column._VocabularyListCategoricalColumn) |
                              isinstance(column, feature_column._IdentityCategoricalColumn) |
                              isinstance(column, feature_column._BucketizedColumn),
               feature_columns)
    )

    sparse_columns = list(
        filter(lambda column: isinstance(column,feature_column._HashedCategoricalColumn) |
                              isinstance(column, feature_column._CrossedColumn),
               feature_columns)
    )

    indicator_columns = []

    if use_indicators:

        indicator_columns = list(
            map(lambda column: tf.feature_column.indicator_column(column),
                categorical_columns)
        )

    deep_columns = dense_columns + indicator_columns
    wide_columns = categorical_columns + sparse_columns

    return deep_columns, wide_columns
