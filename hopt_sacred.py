import warnings

import numpy as np
import json
import collections
import re

from hyperopt import pyll, hp
from hyperopt.base import miscs_update_idxs_vals
from hyperopt.pyll import scope
from pymongo import MongoClient
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds


class objective_wrapper(object):
    def __init__(self, f_obj, main_config, arguments):
        self._f_obj = f_obj
        self._main_config = main_config
        self._arguments = arguments

        self.name = self._get_name()
    def _get_name(self):
        cfg = self._main_config.copy()
        model_name = cfg['model_name']
        cfg.pop('model_name')

        return model_name + '__' + build_string_from_dict(cfg, sep=r'%')

    def objective(self, config):
        print('main config before update', dict(config))
        config = dict(self._main_config, **dict(config))
        print('main config after update', config)
        return self._f_obj(config, self._arguments)



class DuplicateExperiment(Exception):
    def __init__(self, ex):
        self.ex = ex
        super(DuplicateExperiment, self).__init__()

def check_for_completed_experiment(db, cfg):
    ex_list = query_by_config(db, cfg, ignore_missing_keys=False)
    if ex_list:
        print(ex_list[0]['config'])
        return ex_list[0]
    else:
        return None


def query_by_config(db, cfg, ignore_missing_keys=False):
    ex_list = []
    for ex in db.runs.find({'status': 'COMPLETED'}):

        if len(cfg) != len(ex['config']):
            # this ex does not match
            continue

        for k, v in cfg.items():
            try:
                if v != ex['config'][k]:
                    # this ex does not match
                    break
            except KeyError:
                if ignore_missing_keys:
                    continue
                else:
                    # this ex does not match
                    break
        else:
            # match!
            ex_list.append(ex)
    return ex_list


def pow10_and_round(v):
    """
    This calculates the power of 10, and rounds by its decimal precision
    e.g.
    -0.5 --> 10**-0.5 --> 0.316  --> 0.3
    -1.5 --> 10**-1.5 --> 0.0316 --> 0.03
    0.25 --> 10**0.25 --> 1.77   --> 2
    1.25 --> 10**1.25 --> 17.78  --> 20
    """
    return np.round(10**v, -np.int32(np.floor(v)))

def dict_to_sorted_str(d):
    def default(o):
        # a workaround dealing with numpy.int64
        # see https://stackoverflow.com/a/50577730/2476373
        if isinstance(o, np.int64): return int(o)
        raise TypeError

    return json.dumps(d, sort_keys=True, default=default)

def to_exp_space(log_values):
    if type(log_values) == dict:
        return dict([(k, pow10_and_round(v)) for k, v in log_values.items()])
    elif type(log_values) in [tuple, list]:
        return tuple(pow10_and_round(v) for v in log_values)
    else:
        raise RuntimeError()

@scope.define
def f_log_grid(log_space):
    return [(v[0], pow10_and_round(v[1])) for v in log_space]

@scope.define
def f_grid(lin_space):
    return [(v[0], v[1]) for v in lin_space]

class hyperopt_grid():
    def __init__(self, grid_log_ranges={}, grid_lin_ranges={}, grid_categorical={}):
        self._grid_log_ranges = grid_log_ranges
        self._grid_lin_ranges = grid_lin_ranges
        self._grid_categorical = grid_categorical
        self.space = self._get_grid_space()

        # Used for debug during development
        # for _ in range(5):
        #     import hyperopt.pyll
        #     print(hyperopt.pyll.stochastic.sample(self.space))

        self.num_combinations = self._len_outer_product()
        self.executed_params = set()
        self._cnt = 0
        self._cnt_skip = 0
        # print('self.num_combinations ', self.num_combinations)

    def pprint(self):
        for key, _range in self._grid_log_ranges.items():
            if _range[0] != _range[1]:
                print(f'{key}: {pow10_and_round(_range[0])}..{pow10_and_round(_range[1])}  '
                      f'steps: ×{10**_range[2]}')
            else:
                print(f'{key}: {pow10_and_round(_range[0])}')

        for key, _range in self._grid_lin_ranges.items():
            if _range[0] != _range[1]:
                print(f'{key}: {_range[0]}..{_range[1]}  '
                      f'steps: {_range[2]}')
            else:
                print(f'{key}: {_range[0]}')

        for key, _range in self._grid_categorical.items():
            print(f'{key}: {_range}')

    def _get_grid_space(self):
        log_args = [(key, hp.quniform(key, *range_) ) for key, range_ in
                           self._grid_log_ranges.items()]
        lin_args = [(key, hp.quniform(key, *range_) ) for key, range_ in
                           self._grid_lin_ranges.items()]
        categorical_args = [(key, hp.choice(key, range_) ) for key, range_ in
                            self._grid_categorical.items()]
        grid_space = scope.f_log_grid(log_args) + scope.f_grid(lin_args) +\
                     scope.f_grid(categorical_args)

        return grid_space


    def _len_outer_product(self):
        # r[1]+r[2] because quniform range includes the last item
        ranges_elements = [np.unique(np.arange(r[0], r[1] + r[2], r[2])) for r in
                                self._grid_log_ranges.values()]
        ranges_elements += list(self._grid_categorical.values())
        ranges_elements += [np.unique(np.arange(r[0], r[1] + r[2], r[2])) for r in
                                self._grid_lin_ranges.values()]

        return np.prod([len(range_) for range_ in ranges_elements])

    @staticmethod
    def _convert_neg_zeros_to_zeros(trial_dict):
        """ Helps to avoid counting floating zero twice (as +0.0 and -0.0) """
        for k in trial_dict.keys():
            if trial_dict[k][0] == 0 and not isinstance(trial_dict[k][0], (int, np.integer)):
                trial_dict[k] = [+0.0]
        return trial_dict

    @staticmethod
    def _get_historical_params(trials):
        historical_params = []
        for k, trial in enumerate(trials.trials):
            if trials.statuses()[k] == 'ok':
                current_run_params = hyperopt_grid._convert_neg_zeros_to_zeros(
                    dict(trial['misc']['vals']))

                historical_params.append(
                    dict_to_sorted_str(current_run_params))
        return historical_params

    def suggest(self, new_ids, domain, trials, seed):
        rng = np.random.RandomState(seed)
        rval = []    # print('new_ids', new_ids)
        for ii, new_id in enumerate(new_ids):
            while self._cnt <= self.num_combinations:
                # -- sample new specs, idxs, vals
                idxs, vals = pyll.rec_eval(
                    domain.s_idxs_vals,
                    memo={
                        domain.s_new_ids: [new_id],
                        domain.s_rng: rng,
                    })
                new_result = domain.new_result()
                new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
                miscs_update_idxs_vals([new_misc], idxs, vals)
                new_trial = trials.new_trial_docs([new_id],
                            [None], [new_result], [new_misc])
                # Except the `while`, until here, code is copied from rand.suggest

                # new code from here
                self.executed_params = self.executed_params.union(
                    self._get_historical_params(trials))

                # avoid counting floating zero twice (as +0.0 and -0.0)
                this_run_params = hyperopt_grid._convert_neg_zeros_to_zeros(
                    dict(new_misc['vals']))
                # represent the params as a hashed string
                this_run_params_str = dict_to_sorted_str(this_run_params)

                # if these params are seen for the first time, then generate a new
                # trial for them
                if this_run_params_str not in self.executed_params:

                    # add the new trial to returned list
                    rval.extend(new_trial)

                    # log the new trial as executed, in order to avoid duplication
                    self._cnt += 1
                    self.executed_params = \
                        self.executed_params.union([this_run_params_str])
                    print(self._cnt, this_run_params)
                    break
                else:
                    # otherwise (params were seen), skip this trial
                    # update internal counter
                    self._cnt_skip += 1

                # Stopping condition (breaking the hyperopt loop)
                if len(self.executed_params) >= self.num_combinations:
                    # returning an empty list, breaks the hyperopt loop
                    return []


                # "Emergency" stopping condition, breaking the hyperopt loop when
                # loop runs for too long without submitted experiments
                if self._cnt_skip >= 100*self.num_combinations:
                    warnings.warn('Warning: Exited due to too many skips.'
                          ' This can happen if most of the param combinationa have '
                                  'been encountered, and drawing a new '
                                  'unseen combination, involves a very low probablity.')
                    # returning an empty list, breaks the hyperopt loop
                    return []

        return rval

def build_string_from_dict(d, sep='__'):
    """
     Builds a string from a dictionary.
     Mainly used for formatting hyper-params to file names.
     Key-Value(s) are sorted by the key, and dictionaries with
     nested structure are flattened.

    Args:
        d: dictionary

    Returns: string
    :param d: input dictionary
    :param sep:

    """
    fd = _flatten_dict(d)
    return sep.join(
        ['{}={}'.format(k, _value2str(fd[k])) for k in sorted(fd.keys())])

def _flatten_dict(d, parent_key='', sep='_'):
    # from http://stackoverflow.com/a/6027615/2476373
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _value2str(val):
    if isinstance(val, float):  # and not 1e-3<val<1e3:
        # %g means: "Floating point format.
        # Uses lowercase exponential format if exponent is less than -4 or not less than precision,
        # decimal format otherwise."
        val = '%g' % val
    else:
        val = '{}'.format(val)
    val = re.sub('\.', '_', val)
    return val


class hyperopt_sacred_wrapper():

    def sacred_db_name(self):
        """ This method should be override by child class
        returns the Mongo DB name for this set of experiments.
        """
        return 'SACRED_DB'

    def sacred_ex_name(self):
        """ This method should be override by child class
        returns the current experiment name.
        """
        return 'Experiment'


    def __init__(self, f_main, f_config, f_capture, cfg,
                 mongo_url='127.0.0.1:27017', disable_logging=False):

        curr_db_name = self.sacred_db_name()
        ex_name = self.sacred_ex_name()
        ex = Experiment(ex_name)
        ex.captured_out_filter = apply_backspaces_and_linefeeds

        if disable_logging == False:
            print(f'Connecting to MongoDB at {mongo_url}:{curr_db_name}')
            ex.observers.append(MongoObserver.create(url=mongo_url, db_name=curr_db_name))

        # init the experiment configuration (params)
        ex.config(f_config)

        # init the experiment logging (capture) method
        f_ex_capture = ex.capture(f_capture)

        # init the experiment main
        @ex.main
        def ex_main(_run):
            return main_wrapper(f_main, ex, f_ex_capture, self.sacred_db_name(), _run)

        self.ex = ex


def main_wrapper(f_main, ex, f_ex_capture, curr_db_name, _run):
    """
    f_main updates the main experiment function arguments, calls it and save the
    experiment results and artifacts.
    f_main should implement the following API:
        f_main(ex, _run, f_log_metrics)
    """
    client = MongoClient('localhost', 27017)
    print('db = ', curr_db_name)
    db = client[curr_db_name]
    duplicate_ex = check_for_completed_experiment(db, _run.config)
    if duplicate_ex is not None:
        _run.info['model_metrics'] = duplicate_ex['info']['model_metrics']
        _run.info['duplicate_id'] = duplicate_ex['_id']
        print('Aborting due to a duplicate experiment')
        raise DuplicateExperiment(duplicate_ex)
    else:
        return f_main(ex, _run, f_ex_capture)



