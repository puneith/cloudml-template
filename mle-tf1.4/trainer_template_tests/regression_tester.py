import tensorflow as tf

from trainer_template import metadata

print("TensorFlow version:{}".format(tf.__version__))


metadata.TASK_TYPE = "regression"
metadata.HEADERS = "key,x,y,alpha,beta,target".split(",")
metadata.HEADER_DEFAULTS = [[0.], [0.], [0.], [''], [''], [0.]]
metadata.NUMERIC_FEATURE_NAMES = ["x", "y"]
metadata.CATEGORICAL_FEATURE_NAMES_WITH_VOCABULARY = {'alpha': ['ax01', 'ax02'], 'beta': ['bx01', 'bx02']}
metadata.TARGET_NAME = 'target'


print(metadata.CATEGORICAL_FEATURE_NAMES)
print(metadata.UNUSED_FEATURE_NAMES)