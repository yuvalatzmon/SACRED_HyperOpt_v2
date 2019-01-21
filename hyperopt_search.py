import math

import hyperopt
from hyperopt import fmin, hp
from hyperopt.mongoexp import MongoTrials
from pymongo import MongoClient

from sacred_wrapper import hyperopt_objective
from hopt_sacred import hyperopt_grid, to_exp_space

import warnings
# see https://stackoverflow.com/a/40846742
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

def delete_hopt_search_from_mongo_db(experiment_name, hyperopt_db_name, mongo_port):
    """Deletes old experiment-search from hyperopt db (if exists).
    Note: This does not delete the sacred experiments, associated with it. In
    fact, if a sacred experiment exists for some hyper-param, its result will
    be reused.
    """
    client = MongoClient('localhost', mongo_port)
    db_hopt = client[hyperopt_db_name]
    result = db_hopt.jobs.delete_many({'exp_key': {'$regex': experiment_name}})
    print(f'Deleted {result.deleted_count} record(s)')

if __name__ == '__main__':
    # algo = 'rand'
    algo = 'grid'
    mongo_port = 27017
    experiment_name = 'mnist_cnn' #f'{version}{model_obj.name}'
    hyperopt_db_name = 'hyperopt_mnist'

    # Delete old experiment-search from hyperopt db (if exists).
    # Note: This does not delete the sacred experiments, associated with it. In
    # fact, if a sacred experiment exists for some hyper-param, its result will
    # be reused.
    delete_hopt_search_from_mongo_db(experiment_name, hyperopt_db_name, mongo_port)

    trials = MongoTrials(f'mongo://localhost:{mongo_port}/{hyperopt_db_name}/jobs',
                         exp_key=experiment_name)


    if algo == 'grid':

        # Grid search

        # Define the search spac. Ranges are defined in log10 space
        # log_ranges = dict(lr=(-5.5, -0.5, 0.5), # 1e-5, 3e-5, 1e-4, 3e-4, ...
        #                   fc_dim=(1., 2.5, 0.25), # 10, 20, 30, 60, 100, 200, 300
        #                   dropout_rate=(-1, -0.1, 0.1)) # .1, .2, .3, .4, .5, .6, .8,
        log_ranges = dict(lr=(-5, -2, 3), # 1e-5, 3e-5, 1e-4, 3e-4, ...
                          fc_dim=(1., 1., 0.25), # 10, 20, 30, 60, 100, 200, 300
                          dropout_rate=(-1, -1, 0.1)) # .1, .2, .3, .4, .5, .6, .8,
        # A finer search space
        # log_ranges = dict(lr=(-2.5, -2.5, 0.5), # 3e-3
        #                   fc_dim=(1.5, 2.5, 0.25), # 30, 60, 100, 200, 300,
        #                   dropout_rate=(-0.4, -0.1, 0.1)) # .4, .5, .6, .8,


        grid = hyperopt_grid(log_ranges)
        print('Pending on workers to connect ..')
        argmin = fmin(fn=hyperopt_objective,
                     space=grid.space,
                    algo=grid.suggest,
                    # max_evals=grid.num_combinations,
                    max_evals=30,
                    trials=trials,
                    verbose=1)
        best_acc = 1-trials.best_trial['result']['loss']

        print('best val acc = ', best_acc, '\nparams = ', to_exp_space(argmin),
              '\nlog10(params) = ', argmin)

    elif algo == 'rand':
        # Define the search space
        space = {
            'lr': hp.qloguniform('lr', math.log(0.00001), math.log(0.1), 0.00001),
            'fc_dim': hp.qloguniform('fc_dim', math.log(32), math.log(256), 8),
            'dropout_rate': hp.quniform('dropout_rate', 0.1, 1, 0.05),
        }

        print('Pending on workers to connect ..')
        argmin = fmin(fn=hyperopt_objective,
                    space=space,
                    algo=hyperopt.rand.suggest,
                    max_evals=12,
                    trials=trials,
                    verbose=1)
        best_acc = 1-trials.best_trial['result']['loss']

        print('best val acc=', best_acc, 'params:', argmin)

