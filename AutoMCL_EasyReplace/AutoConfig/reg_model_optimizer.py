# -*- coding:utf8 -*

import heapq
import logging
import time
import pandas as pd
import numpy as np
import random

from ..util import sample_ints
from .model_based_tuner import ModelOptimizer, knob2point, point2knob

logger = logging.getLogger('autotvm')


class RegOptimizer(ModelOptimizer):
    def __init__(self, task, n_iter=500, temp=(1, 0), persistent=True, parallel_size=128,
                 early_stop=50, log_interval=50):
        super(RegOptimizer, self).__init__()

        self.task = task
        self.dims = [len(x) for x in self.task.config_space.space_map.values()]

        self.n_iter = n_iter
        self.temp = temp
        self.persistent = persistent
        self.parallel_size = min(parallel_size, len(self.task.config_space))
        self.early_stop = early_stop or 1e9
        self.log_interval = log_interval
        self.points = None
        self.find_maximums_count = 0

    def find_maximums(self, model, num, exclusive):
        # 典型调用：maximums = self.model_optimizer.find_maximums(base_model, self.plan_size, self.visited)
        """Find maximum of a cost model
               Note we use cost model to predict GFLOPS, so we should find the maximum
               Parameters
               ----------
               model: CostModel
                   Cost model
               num: int  ----->常见值对应plan_size = 64
                   The number of returned maximum points
               exclusive: set, optional
                   The excluded set of this optimizer. Return results won't include any
                   elements in this set.
               """
        #print("regOptimizer find maximums!")

        points = np.array(np.arange(1, len(self.task.config_space), 1)).astype('int32')
        pointset = set(points)
        result = pointset - exclusive

        points = np.array(list(result))
        # <class 'numpy.ndarray'>
        if len(points) == 0:
            print("result is null!")
            return list(points)
        scores = model.predict(points)
        if scores.shape[0] > 64:
            # print("case1")
            new_points = self.top_num(points, scores, num, exclusive)  # 前64
        if scores.shape[0] <= 64:
            # print("case2")
            return list(points)
        return new_points

    def top_num(self, points, scores, num, exclusive):
        # tic = time.time()
        points = points.reshape((-1, 1))
        scores = scores.reshape((-1, 1))
        data = np.append(points, scores, axis=1)
        dataframe = pd.DataFrame(data, columns=['index', 'score'])
        sorteddf = dataframe.sort_values(by="score", ascending=False)
        res = []
        count = 0
        ex_count = 0
        config_space = len(self.task.config_space)
        for i, row in sorteddf.iterrows():
            # print(row['index'],'---->' ,row['score'])
            ex_count += 1
            if not exclusive.__contains__(row['index']) and count < num and row['score'] > 1e-9:
                res.append(int(row['index']))
                count += 1
            if count == num or ex_count >= config_space // 2 or len(exclusive) >= config_space // 2:
                # print("top num break")
                break
        # print("top num cost time：", time.time() - tic)
        return res

    def top_num_expand(self, points, scores, num, exclusive):
        # tic = time.time()
        points = points.reshape((-1, 1))
        scores = scores.reshape((-1, 1))
        data = np.append(points, scores, axis=1)
        dataframe = pd.DataFrame(data, columns=['index', 'score'])
        sorteddf = dataframe.sort_values(by="score", ascending=False)
        res = []
        count = 0
        expand_factors = 2.0
        for i, row in sorteddf.iterrows():
            # print(row['index'], '---->',row['score'])
            if not exclusive.__contains__(row['index']) and count < num * expand_factors:
                res.append(row['index'])
                count += 1
            if count == num * expand_factors:
                break
        res_expand_temp = random.sample(res, num)
        res_expand = []
        for r in res_expand_temp:
            res_expand.append(int(r))
        # print("top num expand cost time：", time.time() - tic)
        return res_expand


