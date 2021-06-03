# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=unused-argument, no-self-use, invalid-name
"""Base class of tuner"""
import logging
import time
import numpy as np
import os
from ..measure import MeasureInput, create_measure_batch
from sklearn.externals import joblib
from ..env import GLOBAL_SCOPE

logger = logging.getLogger('autotvm')


class Tuner(object):
    """Base class for tuners

    Parameters
    ----------
    task: autotvm.task.Task
        Tuning Task
    """

    def __init__(self, task, **kwargs):
        self.param = kwargs
        self.recorder = None

        self.task = task
        self.FC = []
        # keep the current best
        self.best_config = None
        self.best_flops = 0
        self.best_measure_pair = None
        self.best_iter = 0

        # time to leave
        self.ttl = None
        self.n_trial = None
        self.early_stopping = None
        self.next_batch_count = 0
        self.lowest_cost = 1

    def has_next(self):
        """Whether has next untried config in the space

        Returns
        -------
        has_next: bool
        """
        raise NotImplementedError()

    def next_batch(self, batch_size):
        """get the next batch of configs to be measure on real hardware

        Parameters
        ----------
        batch_size: int
            The size of the batch

        Returns
        -------
        a batch of configs
        """
        raise NotImplementedError()

    def update(self, inputs, results):
        """Update parameters of the tuner according to measurement results

        Parameters
        ----------
        inputs: Array of autotvm.measure.MeasureInput
            The input for measurement
        results: Array of autotvm.measure.MeasureResult
            result for measurement
        """

    def tuneMCL(self, n_trial, measure_option, early_stopping=None, callbacks=(), initConfig=False, useFilter=False, useRecommend=False,
                sch="dnmm", dtype="float32", L2 = 256 * 1024, cacheline=64, Vl=128 / 8):
        '''
        tuneMCL:
           We introduce several optimization strategies, combining analytic ideal cache models with
           machine learning models trained with real hardware measures, and integrate them into a unified
            auto-tuning framework, called AutoMCL, to improve the performance of DL compilers on both the
             operation level and the end-to-end model inference.
        :param Vl:  int
                Vl be the length of vectorization(B)
        :param cacheline: int
                the cache line size(B)
        :param L2: int
                cache size(B)
        :param n_trial:  int
                Maximum number of configs to try (measure on real hardware)
        :param measure_option: dict
            The options for how to measure generated code.
            You should use the return value ot autotvm.measure_option for this argument.
        :param early_stopping: int, optional
            Early stop the tuning when not finding better configs in this number of trials
        :param callbacks: List of callable
            A list of callback functions. The signature of callback function is
            (Tuner, List of MeasureInput, List of MeasureResult)
            with no return value. These callback functions will be called on
            every measurement pair. See autotvm/tuner/callback.py for some examples.
        :param initConfig: boolean
            Select whether to use the initConfig knob
        :param useFilter: boolean
            Select whether to use the filter knob
        :param useRecommend: boolean
            Select whether to use the recommend knob
        :param sch: string
            support 'tmm', 'ttmm', 'dnmm', 'dnmm332', 'dpmm', 'lpmm', 'rpmm', 'rpmmv', 'conv2d','convim2colrpmm','convim2coldnmm' and 'autoschedule'
        :param dtype: string
            float32 or float64
        '''
        measure_batch = create_measure_batch(self.task, measure_option)
        n_parallel = getattr(measure_batch, 'n_parallel', 1)
        early_stopping = early_stopping or 1e9
        self.n_trial = n_trial
        self.early_stopping = early_stopping
        old_level = logger.level
        GLOBAL_SCOPE.in_tuning = True
        i = error_ct = 0
        Total_time = 0


        M,K,N=0,0,0
        if sch is "autoschedule":
            print("autoschedule:auto select schedule from template!")
            if self.task.name == "topi_nn_dense":
                print("task name is topi nn dense")
                full_path = os.path.dirname(__file__)
                clf = joblib.load(full_path+'/clf.model')
                M, N, K = self.task.args[0][1][0], self.task.args[1][1][0], self.task.args[0][1][1]
                sch = str(int(clf.predict([[M, K, N]])[0]))
                if sch == "2":
                    print("dnmm332")
                    sch = "dnmm332"
                if sch == "4":
                    print("rpmmv")
                    sch = "rpmmv"
                if sch == "6":
                    print("lpmm")
                    sch = "lpmm"
                if sch == "7":
                    print("tmm")
                    sch = "tmm"
            elif self.task.name == "topi_nn_conv2d":
                print("task name is topi_nn_conv2d")
                sch = "conv2d"
                # full_path = os.path.dirname(__file__)
                # clf = joblib.load(full_path+'/clf_conv.model')
                # B, IC, DH, DW = self.task.args[0][1]
                # OC, IC, KH, KW = self.task.args[1][1]
                # p1, p2 = self.task.args[2]
                # s1, s2 = self.task.args[3]
                # d1, d2 = self.task.args[4]
                #
                # sch = str(int(clf.predict([[B, IC,DH,OC,KH,s1,p1]])[0]))
                # if sch == "1":
                #     print("conv2d")
                #     sch = "conv2d"
                # if sch == "2":
                #     print("convim2colrpmm")
                #     sch = "convim2colrpmm"
                # if sch == "3":
                #     print("convim2coldnmm")
                #     sch = "convim2coldnmm"
            else:
                logger.error("\n autoschedule:not support now!")


        if sch not in ['tmm', 'ttmm', 'dnmm', 'dnmm332', 'dpmm', 'lpmm', 'rpmm', 'rpmmv', 'conv2d','convim2colrpmm','convim2coldnmm']:
            logger.error("\n Your selected schedule is not support!.")
        print("schedule template is:",sch)
        print(self.task.args)
        dataframe, configSpace = self.initConfigSpace(schedule=sch,n_parallel=n_parallel,isInitConfig=initConfig,task_args=self.task.args)

        if useFilter is True:
            configSpace = self.filterConfigSpace(schedule=sch, dtype=dtype, L2=L2, cacheline=cacheline, Vl=Vl, useFilter=useFilter, dataframe=dataframe,task_args=self.task.args)
            # iprint("filter configs indexs:",filter_configs)
            # self.FC = filter_configs
        #total_features = self.cost_model._get_feature(range(len(self.task.config_space)))
        #np.savetxt("total_features.txt",total_features)

        #print("n_parallel:", n_parallel)
        while i < n_trial:
            if not self.has_next():
                break
            '''get the next batch of configs to be measure on real hardware'''
            #if useFilter is True:
            if useRecommend is True and self.next_batch_count < self.plan_size // n_parallel and not self.transfer_flag:
                print(" | case: init_batch_by_Complexity")
                configs = self.init_batch_by_Complexity(M, K, N, min(n_parallel, n_trial - i), sch, configSpace, i)
                self.next_batch_count += 1
                #print(" next batch count:", self.next_batch_count)
                # if useRecommend is False and self.next_batch_count < len(filter_configs)//8 and not self.transfer_flag:
                # configs = self.next_batch_filter(min(n_parallel, n_trial - i), filter_configs)
                # self.next_batch_count +=1
            if useRecommend is False or self.next_batch_count >= self.plan_size // n_parallel - 1:
                    # self.next_batch_count += 1
                print(" | case next_batch_filter")
                useRecommend = False
                configs = self.next_batch_filter(min(n_parallel, n_trial - i), configSpace)
           # print(" next batch count:", self.next_batch_count)
            else:
                print(" | case next_batch")
                configs = self.next_batch(min(n_parallel, n_trial - i))

            if len(configs) == 0:
                print("\n Config is null--->break")
                break

            inputs = [MeasureInput(self.task.target, self.task, config) for config in configs]
            # print("tuner inputs:",inputs)
            tic = time.time()
            results = measure_batch(inputs)
            tie = time.time() - tic
            # print(" | measure batch cost: %.2f s"%tie)
            Total_time += tie
            costs = 0.0
            all_cost = 0.0
            ccount = 0
            batch_costs_set = set()
            for measureResult in results:
                if (isinstance(measureResult.costs[0], float)):
                    batch_costs_set.add(measureResult.costs[0])
                all_cost += measureResult.all_cost

            # self.lowest_cost = min(self.lowest_cost,min(batch_costs_set))
            # print(" | ",self.lowest_cost," s")
            # print(max(batch_costs_set))
            # iif self.lowest_cost*50 < max(batch_costs_set):
            # print("lowest cost break")
            # break

            for k, (inp, res) in enumerate(zip(inputs, results)):
                config = inp.config
                if res.error_no == 0:

                    flops = inp.task.flop / np.mean(res.costs)
                    error_ct = 0
                else:

                    flops = 0
                    error_ct += 1

                if flops > self.best_flops:

                    self.best_flops = flops
                    self.best_config = config
                    self.best_measure_pair = (inp, res)
                    self.best_iter = i + k

                logger.debug("No: %d\tGFLOPS: %.2f/%.2f\tresult: %s\t%s", i + k + 1, flops / 1e9, self.best_flops / 1e9,
                             res, self.best_config)

            i += len(results)
            self.ttl = min(early_stopping + self.best_iter, n_trial) - i
            self.update(inputs, results)
            for callback in callbacks:
                callback(self, inputs, results)

            if error_ct > 150:
                logging.basicConfig()
                logger.warning("Too many errors happen in the tuning. Now is in debug mode")
                logger.setLevel(logging.DEBUG)
            else:
                logger.setLevel(old_level)

            if i >= self.best_iter + early_stopping:
                print("early stopping!")
                logger.debug("Early stopped. Best iter: %d.", self.best_iter)
                break

        GLOBAL_SCOPE.in_tuning = False
        del measure_batch

    def tune(self, n_trial, measure_option, early_stopping=None, callbacks=()):
        """Begin tuning

        Parameters
        ----------
        n_trial: int
            Maximum number of configs to try (measure on real hardware)
        measure_option: dict
            The options for how to measure generated code.
            You should use the return value ot autotvm.measure_option for this argument.
        early_stopping: int, optional
            Early stop the tuning when not finding better configs in this number of trials
        callbacks: List of callable
            A list of callback functions. The signature of callback function is
            (Tuner, List of MeasureInput, List of MeasureResult)
            with no return value. These callback functions will be called on
            every measurement pair. See autotvm/tuner/callback.py for some examples.
        """
        measure_batch = create_measure_batch(self.task, measure_option)
        n_parallel = getattr(measure_batch, 'n_parallel', 1)
        early_stopping = early_stopping or 1e9
        self.n_trial = n_trial
        self.early_stopping = early_stopping

        old_level = logger.level

        GLOBAL_SCOPE.in_tuning = True
        i = error_ct = 0
        while i < n_trial:
            if not self.has_next():
                break
            print(" | Next Batch")
            configs = self.next_batch(min(n_parallel, n_trial - i))

            inputs = [MeasureInput(self.task.target, self.task, config) for config in configs]
            results = measure_batch(inputs)

            # keep best config
            for k, (inp, res) in enumerate(zip(inputs, results)):
                config = inp.config
                if res.error_no == 0:
                    flops = inp.task.flop / np.mean(res.costs)
                    error_ct = 0
                else:
                    flops = 0
                    error_ct += 1

                if flops > self.best_flops:
                    self.best_flops = flops
                    self.best_config = config
                    self.best_measure_pair = (inp, res)
                    self.best_iter = i + k

                logger.debug("No: %d\tGFLOPS: %.2f/%.2f\tresult: %s\t%s",
                             i + k + 1, flops / 1e9, self.best_flops / 1e9,
                             res, config)

            i += len(results)
            self.ttl = min(early_stopping + self.best_iter, n_trial) - i

            self.update(inputs, results)
            for callback in callbacks:
                callback(self, inputs, results)

            if i >= self.best_iter + early_stopping:
                logger.debug("Early stopped. Best iter: %d.", self.best_iter)
                break

            if error_ct > 150:
                logging.basicConfig()
                logger.warning("Too many errors happen in the tuning. Now is in debug mode")
                logger.setLevel(logging.DEBUG)
            else:
                logger.setLevel(old_level)

        GLOBAL_SCOPE.in_tuning = False
        del measure_batch

    def reset(self):
        """reset the status of tuner"""
        self.best_config = None
        self.best_flops = 0
        self.best_measure_pair = None

    def load_history(self, data_set):
        """load history data for transfer learning

        Parameters
        ----------
        data_set: Array of (MeasureInput, MeasureResult) pair
            Previous tuning records
        """
        raise NotImplementedError()
