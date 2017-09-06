import argparse
import os
import shutil

import tensorflow as tf
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.contrib.learn.python.learn.utils import (saved_model_export_utils)
from tensorflow.contrib.training.python.training import hparam

import parameters
import experiment
import serving


def main():

    args_parser = argparse.ArgumentParser()
    args = parameters.initialise_arguments(args_parser)
    parameters.HYPER_PARAMS = hparam.HParams(**args.__dict__)

    # Set python level verbosity
    tf.logging.set_verbosity(args.verbosity)
    # Set C++ Graph Execution level verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
        tf.logging.__dict__[args.verbosity] / 10)

    # Run the training job
    # learn_runner pulls configuration information from environment
    # variables using tf.learn.RunConfig and uses this configuration
    # to conditionally execute Experiment, or param server code

    if args.remove_model_dir == 'True':
        print("Removing model {}".format(args.job_dir))
        shutil.rmtree(args.job_dir, ignore_errors=True)
    else:
        print("Resume training model {}".format(args.job_dir))

    learn_runner.run(
        experiment.generate_experiment_fn(
            min_eval_frequency=args.min_eval_frequency,
            eval_delay_secs=args.eval_delay_secs,
            train_steps=args.train_steps,
            eval_steps=args.eval_steps,
            export_strategies=[saved_model_export_utils.make_export_strategy(
                serving.SERVING_FUNCTIONS[args.export_format],
                exports_to_keep=1,
                default_output_alternative_key=None,
            )]
        ),
        run_config=run_config.RunConfig(model_dir=args.job_dir),
        hparams=parameters.HYPER_PARAMS
    )


if __name__ == '__main__':
    main()