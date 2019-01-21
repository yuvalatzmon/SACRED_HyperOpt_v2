import sys

import mnist_keras
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
from pymongo import MongoClient

import warnings
# see https://stackoverflow.com/a/40846742
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

common_args = mnist_keras.args

# ex = Experiment('My_Experiment')
# curr_db_name = 'sacred_mnist_example'
ex = None
curr_db_name = None
config, sacred_main, SACRED_log_metrics  = None, None, None

def sacred_db_name(is_debug):
    str_debug = 'debug' if is_debug == 1 else ''
    return 'MNIST' + str_debug


def sacred_ex_name(cfg):
    return 'MNIST_CNN'

def setup_SACRED_experiments_framework(mongo_url='127.0.0.1:27017'):
    global ex, curr_db_name
    global config, sacred_main, SACRED_log_metrics, common_args

    curr_db_name = sacred_db_name(vars(common_args).get('debug', 0))
    ex_name = sacred_ex_name(vars(common_args))
    ex = Experiment(ex_name)
    ex.captured_out_filter = apply_backspaces_and_linefeeds

    if common_args.disable_logging == False:
        print(f'Connecting to MongoDB at {mongo_url}:{curr_db_name}')
        ex.observers.append(MongoObserver.create(url=mongo_url, db_name=curr_db_name))

    # noinspection PyUnusedLocal
    @ex.config
    def config():
        # Set SACRED configuration and the default values (it will override argparse
        # defaults). Note that we take only the hyper-params. Other arguments like
        # --verbose, or --gpu_memory_fraction are irrelevant for the model.
        seed = 0
        lr = 1e-3
        dropout_rate = 0.5
        fc_dim = 128
        epochs = 20
        batch_size = 32

    @ex.main
    def main(_run):
        cfg = _run.config
        print(cfg)

        # Override argparse arguments with sacred arguments
        command_line_args = mnist_keras.args # argparse command line arguments
        vars(command_line_args).update(cfg)

        client = MongoClient('localhost', 27017)
        print('db = ', curr_db_name)
        db = client[curr_db_name]
        duplicate_ex = hyperopt_utils.check_for_completed_experiment(db, _run.config)
        if duplicate_ex is not None:
            _run.info['model_metrics'] = duplicate_ex['info']['model_metrics']
            _run.info['duplicate_id'] = duplicate_ex['_id']
            print('Aborting due to a duplicate experiment')
            raise hyperopt_utils.DuplicateExperiment(duplicate_ex)
        else:

            # call main script
            val_acc, test_acc = mnist_keras.main(f_log_metrics=log_metrics)
            _run.info['model_metrics'] = dict(val_acc=val_acc, test_acc=test_acc)

            ex.add_artifact('mnist_model.h5')

        return val_acc

    @ex.capture
    def log_metrics(_run, logs):
        _run.log_scalar("loss", float(logs.get('loss')))
        _run.log_scalar("acc", float(logs.get('acc')))
        _run.log_scalar("val_loss", float(logs.get('val_loss')))
        _run.log_scalar("val_acc", float(logs.get('val_acc')))
        _run.result = float(logs.get('val_acc'))



##### HyperOpt support #####################
from hyperopt import STATUS_OK
# noinspection PyUnresolvedReferences
import hyperopt_utils
def hyperopt_objective(params):
    config = {}

    if type(params) == dict:
        params = params.items()

    for (key, value) in params:
        if key in ['fc_dim']:
            value = int(value)
        config[key] = value

    db_name = sacred_db_name(config.get('debug', 0))
    print('db_name=', db_name)

    setup_SACRED_experiments_framework()
    ex_name = sacred_ex_name(dict(vars(common_args), **config))
    try:
        run = ex.run(config_updates=config, options={'--name': ex_name})
        ex_res = vars(run)
    except hyperopt_utils.DuplicateExperiment as e:
        ex_res = e.ex

    err_rate = ex_res['result']
    metrics = {'loss': 1 - err_rate, 'status': STATUS_OK}

    return metrics

##### End HyperOpt support #################



if __name__ == '__main__':
    argv = sys.argv
    # Remove the argparse arguments, since they have already been processed by
    # argparse library, and they interfere with the SACRED command line parser
    sacred_argv = [arg for arg in argv if not arg.startswith('--')]

    setup_SACRED_experiments_framework()
    ex.run_commandline(sacred_argv)
