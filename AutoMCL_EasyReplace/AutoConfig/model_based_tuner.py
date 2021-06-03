import gc
from .tuner import Tuner
from ..env import GLOBAL_SCOPE
import time
import random
import math
import re
import numpy as np
import pandas as pd


def get_pow2s(n):
    return [2 ** x for x in range(math.ceil(math.log2(n)) + 1)]


def get_div2s(n):
    res = [n, 1]
    while n % 2 == 0:
        n = n / 2
        res.append(int(n))
    return res


def getFactors(n):
    res = []
    Factor(n, res, 0)
    res = set(res)
    res.add(n)
    res = sorted(res)
    # print(res)
    return res


def Merge1(TM_dataframe, TN_dataframe):
    newDf = pd.DataFrame()
    for _, A_row in TM_dataframe.iterrows():
        for _, B_row in TN_dataframe.iterrows():
            AData = A_row['Tm']
            BData = B_row['Tn']
            row = pd.DataFrame([dict(Tm=AData, Tn=BData)])
            newDf = newDf.append(row, ignore_index=True)
    return newDf


def Merge2(newDf, Tk_dataframe):
    newDf1 = pd.DataFrame()
    for _, A_row in newDf.iterrows():
        for _, B_row in Tk_dataframe.iterrows():
            AData = A_row
            BData = B_row['Tk']
            row = pd.DataFrame([dict(AData, Tk=BData)])
            newDf1 = newDf1.append(row, ignore_index=True)
    return newDf1


def initConfigSpace_Mini(M, N, K, pThread):
    DF = pd.DataFrame()
    P = pThread
    P1_set = getFactors(P)
    for p1 in P1_set:
        p2 = math.ceil(P / p1)
        # candidateTM = getFactors(math.ceil(M / p1))+get_pow2s(math.ceil(M / p1))
        # candidateTN = getFactors(math.ceil(N / p2))+get_pow2s(math.ceil(N / p2))
        candidateTM = getFactors(math.ceil(M / p1))
        candidateTN = getFactors(math.ceil(N / p2))
        candidateTM = [x for x in list(set(candidateTM)) if x <= M]
        candidateTN = [x for x in list(set(candidateTN)) if x <= N]
        Tm_dataframe = pd.DataFrame(list(set(candidateTM)), columns=['Tm'])
        Tn_dataframe = pd.DataFrame(list(set(candidateTN)), columns=['Tn'])
        newDf = Merge1(Tm_dataframe, Tn_dataframe)
        DF = DF.append(newDf, ignore_index=True)
    #DF.drop_duplicates(['Tm', 'Tn'])

    # candidateTM_appendx_p = [x for x in list(set(get_pow2s(M))) if x <= M]
    # candidateTN_appendx_p = [x for x in list(set(get_pow2s(N))) if x <= N]
    # Tm_dataframe = pd.DataFrame(list(set(candidateTM_appendx_p)), columns=['Tm'])
    # Tn_dataframe = pd.DataFrame(list(set(candidateTN_appendx_p)), columns=['Tn'])
    # newDf = Merge1(Tm_dataframe, Tn_dataframe)
    # DF = DF.append(newDf, ignore_index=True)
    #candidateTK = getFactors(K) + get_pow2s(K)
    candidateTK = getFactors(K)
    candidateTK = [x for x in list(set(candidateTK)) if x <= K]
    Tk_dataframe = pd.DataFrame(list(candidateTK), columns=['Tk'])
    DF = Merge2(DF, Tk_dataframe)
    DF = DF.drop_duplicates(['Tm', 'Tn', 'Tk'])
    newDF = DF[['Tm', 'Tn', 'Tk']]
    print('fm:', sorted(set(newDF['Tm'].tolist())))
    print('fn:', sorted(set(newDF['Tn'].tolist())))
    print('fk:', sorted(set(newDF['Tk'].tolist())))
    print('len:',len(newDF))
    return newDF

def initConfigSpace_PRO(M, N, K, pThread):
    DF = pd.DataFrame()
    G = list()
    TM_dataframe = pd.DataFrame()
    TN_dataframe = pd.DataFrame()
    P = pThread
    P1_set = getFactors(P)
    for p1 in P1_set:
        candidateTM = set()
        candidateTN = set()
        candidateTK = set()
        p2 = math.ceil(P / p1)
        for i in range(math.ceil(M / p1)):
            by = math.ceil(M / (p1 * (i + 1)))
            candidateTM.add(by)
            # if M % (p1 *(i+1)) == 0:
            #    by = int(M / (p1 * (i+1)))
            #    candidateTM.add(by)
            if by > 1:
                candidateTM.add(by - 1)
            if by < M:
                candidateTM.add(by + 1)

        for i in range(math.ceil(N / p2)):
            bx = math.ceil(N / (p2 * (i + 1)))
            candidateTN.add(bx)
            # if N % (p2 *(i+1)) == 0:
            #    bx = int(N / (p2 * (i+1)))
            #    candidateTN.add(bx)
            if bx > 1:
                candidateTN.add(bx - 1)
            if bx < N:
                candidateTN.add(bx + 1)

        Tm_dataframe = pd.DataFrame(list(candidateTM), columns=['Tm'])
        TM_dataframe = pd.concat([TM_dataframe, Tm_dataframe])
        Tn_dataframe = pd.DataFrame(list(candidateTN), columns=['Tn'])
        TN_dataframe = pd.concat([TN_dataframe, Tn_dataframe])
        newDf = Merge1(Tm_dataframe, Tn_dataframe)
        DF = DF.append(newDf, ignore_index=True)


    DF.drop_duplicates(['Tm', 'Tn'])
    newDF = DF[['Tm', 'Tn']]
    #np.savetxt("init_config_space_pro.txt", newDF.values)
    return newDF

def Factor(n, res, count):
    if n == 1:
        count += 1
    else:
        for i in range(2, int(n + 1)):
            if n % i == 0:
                res.append(int(n / i))
                Factor(n / i, res, count)


def tmm(M, N, K, Mt, Nt, Kt, Cw):
    return (M * N * K) / Nt * (1 / Cw + 1 / Kt) \
           + M * N * K / Mt * (1 / Cw + 1 / Kt) \
           + M * N / Cw + M * N / Nt


def ttmm(M, N, K, Mt, Nt, Kt, Cw):
    return (M * N * K) / Nt * (1 / Cw + 1 / Kt) \
           + M * N * K / Mt * (1 / Cw + 1 / Nt) \
           + M * N * (1 / Cw + 1 / Nt) \
           + K * N * (2 / Cw + 1 / Nt + 1 / Kt)


def dnmm(M, N, K, Mt, Nt, Kt, Cw):
    return (M * N * K) / Nt * (1 / Cw + 1 / Kt) \
           + M * N * K / Mt * (1 / Cw + 1 / Kt) \
           + M * N * (Kt + 1) / Cw + 2 * M * N / Nt


def dpmm(M, N, K, Mt, Nt, Kt, Cw):
    return 2 * N * K / Cw + 2 * M * N / Cw \
           + N + M * N / Mt / Nt + M * N / Nt + N / Nt \
           + M * N * K / Nt * (1 / Cw + 1 / Kt) \
           + M * N * K / Mt * (1 / Cw + 1 / Nt / Kt)


def get_system_info(cacheline, L2, Vl, dtype):
    Dl = 4
    if dtype is "float32":
        Dl = 4
    if dtype is "float64":
        Dl = 8
    Cw = cacheline / Dl
    Z = L2
    Zw = Z / Dl
    Vw = Vl / Dl
    return Cw, Zw, Vw


def complexityRecommend(M, K, N, parallel_size, schedule, i):
    parallel_size = 8
    data = np.loadtxt("filter_config_space.txt")
    res = dict()
    Cw = 16
    for d in data:
        Mt, Nt, Kt, index = d[0], d[1], d[2], d[3]
       # print("Mt, Kt, Nt = ", Mt, Kt, Nt)
        if schedule is "tmm":
            res[str(index)] = tmm(M, N, K, Mt, Nt, Kt, Cw)
        elif schedule is "ttmm":
            res[str(index)] = ttmm(M, N, K, Mt, Nt, Kt, Cw)
        elif schedule is "dnmm":
            res[str(index)] = dnmm(M, N, K, Mt, Nt, Kt, Cw)
        elif schedule is "dpmm":
            res[str(index)] = dpmm(M, N, K, Mt, Nt, Kt, Cw)

    ress = sorted(res.items(), key=lambda d: d[1], reverse=False)
    keylist = dict(ress[i * parallel_size: parallel_size * (i + 1)]).keys()
    keylist = list(map(float, keylist))
    keylist = list(map(int, keylist))
    # print(keylist)
    return keylist


class FeatureCache(object):
    """Feature cache manager for cache sharing between different cost models"""

    def __init__(self):
        self.feature_cache = {}

    def get(self, key):
        """ Get feature cache dictionary for a key

        Parameters
        ----------
        key: str
            The key of a feature type

        Returns
        -------
        fea_cache: dict
            cache dictionary
        """
        if key not in self.feature_cache:
            self.feature_cache[key] = {}

        return self.feature_cache[key]

    def size(self, key):
        """" Get the size of a feature cache dictionary

        Parameters
        ----------
        key: str
            The key of a feature type

        Returns
        -------
        n: int
        """
        return len(self.feature_cache.get(key, tuple()))

    def clear(self, key):
        """Clear feature cache for a key

        Parameters
        ----------
        key: str
            The key of a feature type
        """
        del self.feature_cache[key]
        self.feature_cache[key] = {}
        gc.collect()


class CostModel(object):
    """Cost model to predict the speed of a config"""

    def __init__(self):
        pass

    def fit(self, xs, ys, plan_size):
        """Fit to training data

        Parameters
        ----------
        xs: Array of int
            indexes of configs in the config space
        ys: Array of float
            The speed (flop, float number operations per second)
        plan_size: int
            The plan size of tuner
        """
        raise NotImplementedError()

    def fit_log(self, records, plan_size):
        """Fit training data from log.

        Parameters
        ----------
        records: Array of Tuple(MeasureInput, MeasureResult)!!!!!!!!!
            The tuning records
        plan_size: int
            The plan size of tuner
        """
        raise NotImplementedError()

    def predict(self, xs, output_margin=False):
        """Predict the speed of configs

        Parameters
        ----------
        xs: Array of int
            The indexes of configs to predict
        output_margin: bool, optional
            Whether output the untransformed margin.
            When a model is used as base model, it should output untransformed margin

        Returns
        -------
        preds: Array of float
            The prediction
        """
        raise NotImplementedError()

    def load_basemodel(self, base_model):
        """Load base model for transfer learning

        Parameters
        ----------
        base_model: CostModel
                base model
        """
        raise NotImplementedError()

    def spawn_base_model(self):
        """Clone a base model with the same parameters.
        The base model is used to fit history data in transfer learning.

        Returns
        -------
        model: CostModel
            A model with the same hyperparameter (argument)
        """
        raise NotImplementedError()


class ModelOptimizer(object):
    """Optimizer used to find optimal points of cost model"""

    def __init__(self):
        pass

    def find_maximums(self, model, num, exclusive):
        """Find maximum of a cost model

        Note we use cost model to predict GFLOPS, so we should find the maximum

        Parameters
        ----------
        model: CostModel
            Cost model
        num: int
            The number of returned maximum points
        exclusive: set, optional
            The excluded set of this optimizer. Return results won't include any
            elements in this set.
        """
        raise NotImplementedError()


class ModelBasedTuner(Tuner):
    """Base class for model based tuner
    This type of tuner will fit a cost model and use an optimizer to
    find the maximums of the cost model as next trials

    Parameters
    ----------
    task: autotvm.task.Task
        The tuning task
    cost_model: CostModel
        The cost model that predicts the speed of a config (IR)
    model_optimizer:
        The optimizer to find local optimum points of cost model in tuning search space
    plan_size: int
        Tuner will re-fit model per `plan_size` new measure samples
    diversity_filter_ratio: int or float, optional
        If is not None, the tuner will first select
        top-(plan_size * diversity_filter_ratio) candidates according to the cost model
        and then pick plan_size of them according to the diversity metric.
    """

    def __init__(self, task, cost_model, model_optimizer, plan_size, diversity_filter_ratio=None):
        super(ModelBasedTuner, self).__init__(task)

        # space
        self.task = task
        self.target = task.target
        self.plan_size = plan_size
        self.space = task.config_space
        self.space_len = len(task.config_space)
        self.dims = [len(x) for x in self.space.space_map.values()]

        self.cost_model = cost_model
        self.model_optimizer = model_optimizer
        self.diversity_filter_ratio = diversity_filter_ratio

        if self.diversity_filter_ratio:
            assert self.diversity_filter_ratio >= 1, "Diversity filter ratio " \
                                                     "must be larger than one"

        # trial plan
        self.trials = []
        self.trial_pt = 0
        self.visited = set()

        # observed samples

        self.xs = []
        self.ys = []
        self.flops_max = 0.0
        self.train_ct = 0
        self.rets = []
        self.transfer_flag = False

    def next_batch(self, batch_size):
        # tic = time.time()
        ret = []
        counter = 0
        while counter < batch_size:
            if len(self.visited) >= len(self.space) * 1.0:
                break

            while self.trial_pt < len(self.trials):
                index = self.trials[self.trial_pt]
                if index not in self.visited:
                    break
                self.trial_pt += 1


            if self.trial_pt >= len(self.trials) - int(0.05 * self.plan_size):
                index = np.random.randint(len(self.space))
                while index in self.visited:
                    index = np.random.randint(len(self.space))

            ret.append(self.space.get(index))
            self.visited.add(index)

            counter += 1

        self.rets.append(ret)

        # print("next batch cost time:",time.time()-tic)
        return ret
    def initConfigSpace(self, schedule, n_parallel,isInitConfig,task_args):
        M,N,K=0,0,0
        if schedule in ['tmm', 'ttmm', 'dnmm332', 'dpmm', 'lpmm', 'rpmm', 'rpmmv']:
            M, N, K = task_args[0][1][0], task_args[1][1][0], task_args[0][1][1]
            config_strategy = "332"
        if schedule == "dnmm":
            M, N, K = task_args[0][1][0], task_args[1][1][0], task_args[0][1][1]
            config_strategy = "222"
        if schedule == "conv2d":
            #ic oc ow == K, N, M
            config_strategy = "222"
        if schedule in ["convim2colrpmm","convim2coldnmm"]:
            config_strategy = "332"

        M,N,K = int(M),int(N),int(K)
        print("space len:",self.space_len)
        ress = []
        pattern1 = re.compile("\d+", re.S)
        t1 = time.time()
        print("parse space start")
        for j in range(len(self.space)):
            data1 = pattern1.findall(str(self.space.get(j + 1)))
            res = []
            for i in data1:
                res.append(int(i))
            ress.append(res)
        ress = np.array(ress)
        ress_cols = ress.shape[1]
        t2 = time.time()
        #print(ress)
        #print(ress.shape)
        #np.savetxt("ress.txt",ress)
        print("parse space end,parse cost time is:",t2-t1)

        if config_strategy is "222":
            # for gemm: M N K = fx fy fk
            # for conv2d: ic oc ow = fx fy fk
            dataframe1 = pd.DataFrame(ress[:, :6], columns=['x', 'fx', 'y', 'fy', 'k', 'fk'])
            dataframe2 = pd.DataFrame(ress[:, ress_cols - 1:], columns=['index'])
            dataframe = pd.concat([dataframe1, dataframe2], axis=1)
        # M N K = fx fy fk
        if config_strategy is "332":
            dataframe1 = pd.DataFrame(ress[:, :8], columns=['xt', 'xo', 'fx', 'yt', 'yo', 'fy', 'k', 'fk'])
            dataframe2 = pd.DataFrame(ress[:, ress_cols - 1:], columns=['index'])
            dataframe = pd.concat([dataframe1, dataframe2], axis=1)

        print("configs space before filter:", len(dataframe['index'].tolist()))
        if schedule in ["conv2d","convim2colrpmm","convim2coldnmm"]:
             d_max = dataframe.max()
             print(d_max)
             # ic oc ow = fk fx fy
             M = d_max['fx']
             N = d_max['fy']
             K = d_max['fk']
        index = dataframe['index'].tolist()
        if isInitConfig is False or schedule in ["conv2d"]:
            return dataframe,index
        # init config space select:
        print("M N K",M,N,K)
        datamn_df = initConfigSpace_Mini(M, N, K, pThread=n_parallel)
        datamn_df.rename(columns={"Tm": "fx", "Tn": "fy","Tk":"fk"}, inplace=True)
        if schedule is "dpmm":
            print("N and M must divides")
            tn_candidate = getFactors(N)
            dataframe = dataframe[dataframe['fy'].isin(tn_candidate)]
            tm_candidate = getFactors(M)
            dataframe = dataframe[dataframe['fx'].isin(tm_candidate)]
        if schedule is "lpmm":
            print("M must divides")
            tm_candidate = getFactors(M)
            dataframe = dataframe[dataframe['fx'].isin(tm_candidate)]
        if schedule in ["rpmm","rpmmv","convim2colrpmm"]:
            print("N must divides")
            tn_candidate = getFactors(N)
            dataframe = dataframe[dataframe['fy'].isin(tn_candidate)]


        if config_strategy is "222":
            intersected_df = pd.merge(datamn_df, dataframe, on=['fx','fy','fk'], how='inner')
        if config_strategy is "332":
            new_dataframe = pd.DataFrame(columns=['fx', 'fy', 'fk','index'])
            # new_dataframe['x'] = dataframe['xo']
            # new_dataframe['fx'] = dataframe['fx']
            # new_dataframe['y'] = dataframe['yo']
            # new_dataframe['fy'] = dataframe['fy']
            # new_dataframe['k'] = dataframe['k']
            new_dataframe['fx'] = dataframe['xo']*dataframe['fx']
            new_dataframe['fy'] = dataframe['yo']*dataframe['fy']
            new_dataframe['fk'] = dataframe['fk']
            new_dataframe['index'] = dataframe['index'].tolist()
            intersected_df = pd.merge(datamn_df, new_dataframe, on=['fx', 'fy','fk'], how='inner')


        #intersected_df = intersected_df.drop_duplicates()
        index = intersected_df['index'].tolist()
        print('fm:', sorted(set(intersected_df['fx'].tolist())))
        print('fn:', sorted(set(intersected_df['fy'].tolist())))
        print('fk:', sorted(set(intersected_df['fk'].tolist())))
        index = list(set(index))
        #np.savetxt("intersected_df_values.txt",intersected_df['index'].values)
        print("Step1.initial configs space is: ", len(index))
        return intersected_df,index

    def filterConfigSpace(self, schedule, dtype, L2, cacheline, Vl, useFilter, dataframe, task_args):
        print("\n***********************filterConfigSpace start...***********************\n")
        M, N, K = 0,0,0
        if schedule in ["conv2d","convim2colrpmm","convim2coldnmm"] :
            B, IC, DH, DW = task_args[0][1]
            OC, IC, KH, KW = task_args[1][1]
            p1, p2 = task_args[2]
            s1, s2 = task_args[3]
            d1, d2 = task_args[4]
            #print(((B,IC,DH,DW),(OC,IC,KH,KW),(p1,p2,p1,p2),(s1,s2),(d1,d2)))
        else:
            M, N, K = task_args[0][1][0], task_args[1][1][0], task_args[0][1][1]
            

        # system infomation
        Cw, Zw, Vw = get_system_info(cacheline=cacheline, L2=L2, Vl=Vl, dtype=dtype)
        if useFilter is True:
            if schedule in ["dnmm","dnmm332","convim2coldnmm"]:
            # cache limitation
                dataframe = dataframe[(dataframe['fx'] * (
                    (dataframe['fy'] * dataframe['fk'] // Cw + (dataframe['fy'] * dataframe['fk'] % Cw) != 0) + 1)
                                   + dataframe['fx'] * (dataframe['fy'] // Cw + (dataframe['fy'] % Cw != 0) + 1))
                                  <= Zw / Cw]
                dataframe = dataframe[(dataframe['fx'] * (
                    (dataframe['fy'] * dataframe['fk'] // Cw + (dataframe['fy'] * dataframe['fk'] % Cw) != 0) + 1)
                                   + (dataframe['fx'] + dataframe['fy']) * (
                                           dataframe['fk'] // Cw + (dataframe['fk'] % Cw != 0) + 1))
                                  <= Zw / Cw]
            # vec limitation
                print((Vw if K > Vw else K),type(Vw if K > Vw else K))
                dataframe = dataframe[dataframe['fk'] >= (Vw if K > Vw else K)]
            if schedule is "tmm":
            # cache limitation
                dataframe = dataframe[(dataframe['fx'] * ((dataframe['fk'] // Cw + (dataframe['fk'] % Cw) != 0) + 1))
                                  + (dataframe['fk'] * ((dataframe['fy'] // Cw + (dataframe['fy'] % Cw) != 0) + 1))
                                  + (dataframe['fx'] * (
                    (dataframe['fy'] // Cw + (dataframe['fy'] % Cw) != 0) + 1)) <= Zw / Cw]
            # vec limitation
                dataframe = dataframe[dataframe['fy'] >= (Vw if N > Vw else N)]
            if schedule is "ttmm":
                dataframe = dataframe[(dataframe['fk'] * ((dataframe['fy'] // Cw + dataframe['fy'] % Cw != 0) + 1)
                                   + dataframe['fy'] * (
                                           (dataframe['fk'] // Cw + dataframe['fk'] % Cw != 0) + 1)) <= Zw / Cw]
                dataframe = dataframe[(dataframe['fk'] * ((dataframe['fy'] // Cw + (dataframe['fk'] % Cw) != 0) + 1)
                                   + dataframe['fx'] * (((dataframe['fk'] // Cw + dataframe['fk'] % Cw != 0) + 1) + (
                            dataframe['fy'] // Cw + dataframe['fy'] % Cw != 0) + 1))
                                  <= Zw / Cw]
                dataframe = dataframe[dataframe['fy'] >= (Vw if N > Vw else N)]
            if schedule is "dpmm":
                dataframe = dataframe[(dataframe['fx']*dataframe['fy'] // Cw + (dataframe['fx']*dataframe['fy']% Cw != 0) + 1) +
                                    dataframe['fx']*((dataframe['fy'] // Cw)+(dataframe['fy'] % Cw != 0) + 1)
                                  <= Zw / Cw]
                dataframe = dataframe[(dataframe['fx']*dataframe['fy'] // Cw + (dataframe['fx']*dataframe['fy']% Cw != 0) + 1) +
                                    ((dataframe['fx'] * dataframe['fk'] // Cw)+(dataframe['fx'] * dataframe['fk'] % Cw != 0) + 1)
                                  <= Zw / Cw]
                dataframe = dataframe[(dataframe['fy'] // Cw + (dataframe['fy']% Cw != 0) + 1) + dataframe['fy']
                                  <= Zw / Cw]
                dataframe = dataframe[(dataframe['fx'] // Cw + (dataframe['fx'] % Cw != 0) + 1) + dataframe['fx']
                                      <= Zw / Cw]
                dataframe = dataframe[dataframe['fy'] >= (Vw if N > Vw else N)]
            if schedule is "lpmm":
                dataframe = dataframe[
                    (dataframe['fx'] * dataframe['fy'] // Cw + (dataframe['fx'] * dataframe['fy'] % Cw != 0) + 1) +
                    dataframe['fx'] * ((dataframe['fy'] // Cw) + (dataframe['fy'] % Cw != 0) + 1)
                    <= Zw / Cw]
                dataframe = dataframe[
                    (dataframe['fx'] * dataframe['fy'] // Cw + (dataframe['fx'] * dataframe['fy'] % Cw != 0) + 1) +
                    dataframe['fy'] * ((dataframe['fk'] // Cw) + (dataframe['fk'] % Cw != 0) + 1) +
                    (dataframe['fx'] * dataframe['fk'] // Cw + (dataframe['fx'] * dataframe['fk'] % Cw != 0) + 1)
                    <= Zw / Cw]
                dataframe = dataframe[
                    (dataframe['fx'] // Cw + (dataframe['fx'] % Cw != 0) + 1) + dataframe['fx']
                    <= Zw / Cw]
                dataframe = dataframe[dataframe['fy'] >= (Vw if N > Vw else N)]
            if schedule in ["rpmm","rpmmv","convim2colrpmm"]:
                dataframe = dataframe[
                    (dataframe['fx'] * dataframe['fy'] // Cw + (dataframe['fx'] * dataframe['fy'] % Cw != 0) + 1) +
                    dataframe['fx'] * ((dataframe['fy'] // Cw) + (dataframe['fy'] % Cw != 0) + 1)
                    <= Zw / Cw]
                dataframe = dataframe[
                    (dataframe['fx'] * dataframe['fy'] // Cw + (dataframe['fx'] * dataframe['fy'] % Cw != 0) + 1) +
                    dataframe['fx'] * ((dataframe['fk'] // Cw) + (dataframe['fk'] % Cw != 0) + 1) +
                    (dataframe['fy'] * dataframe['fk'] // Cw + (dataframe['fy'] * dataframe['fk'] % Cw != 0) + 1)
                    <= Zw / Cw]
                dataframe = dataframe[
                    (dataframe['fy'] // Cw + (dataframe['fy'] % Cw != 0) + 1) + dataframe['fy']
                    <= Zw / Cw]
                dataframe = dataframe[dataframe['fy'] >= (Vw if N > Vw else N)]
            if schedule is "conv2d":
                print("conv2d filer ...")
                # for conv2d: ic oc ow = fx fy fk
                #ic oc ow = fx,fy,fk
                # fx - fk

                dataframe = dataframe[
                    dataframe['fy'] * (KW*KH*dataframe['fk'] // Cw + (KW*KH*dataframe['fk'] % Cw != 0) + 1) +
                    KW * KH * dataframe['fk'] * ((dataframe['fy'] // Cw) + (dataframe['fy'] % Cw != 0) + 1)
                    <= Zw / Cw]

                #modified at 20210321
                dataframe = dataframe[
                    (dataframe['fx'] * dataframe['fy'] // Cw + (dataframe['fx'] * dataframe['fy'] % Cw != 0) + 1) +
                    KH * dataframe['fk'] * (((s2 * (dataframe['fx']-1)+(KW-1)*d2+1)// Cw) + ((s2 * (dataframe['fx']-1)+(KW-1)*d2+1) % Cw != 0) + 1) +
                    KH * KW * dataframe['fk'] * ((dataframe['fy'] // Cw) + (dataframe['fy'] % Cw != 0) + 1)
                    <= Zw / Cw]

                dataframe = dataframe[
                    (dataframe['fx'] *dataframe['fy'] // Cw + (dataframe['fx'] *dataframe['fy'] % Cw != 0) + 1) +
                    dataframe['fx'] * ((dataframe['fy'] // Cw) + (dataframe['fy'] % Cw != 0) + 1)
                    <= Zw / Cw]

                dataframe = dataframe[
                    dataframe['fx'] * (dataframe['fy'] // Cw + (dataframe['fy'] % Cw != 0) + 1) +
                    dataframe['fy'] * ((dataframe['fx'] // Cw) + (dataframe['fx'] % Cw != 0) + 1)
                    <= Zw / Cw]


            if schedule in ["convim2colrpmm","convim2coldnmm"]:
                print("convim2col filer ...")
                # for im2col ow oc ic= fx fy fk
                # P = KH*KW*dataframe['fk']

                dataframe = dataframe[
                    dataframe['fx'] * (KH*KW*dataframe['fk'] // Cw + (KH*KW*dataframe['fk'] % Cw != 0) + 1) +
                    (dataframe['fk'] + (dataframe['fx'] % Cw != 0) + 1)* KH* (((dataframe['fx']-1)*s2+(KW-1)*d2+1)// Cw + (((dataframe['fx']-1)*s2+(KW-1)*d2+1) % Cw != 0) + 1)
                    <= Zw / Cw]

                dataframe = dataframe[2 * dataframe['fy'] * (KH*KW*dataframe['fk'] // Cw + (KH*KW*dataframe['fk'] % Cw != 0) + 1)
                    <= Zw / Cw]

                dataframe = dataframe[
                    dataframe['fy'] * (dataframe['fx'] // Cw + (dataframe['fx'] % Cw != 0) + 1) +
                    dataframe['fx'] * (dataframe['fy'] // Cw + (dataframe['fy'] % Cw != 0) + 1)
                    <= Zw / Cw]



        index = dataframe['index'].tolist()
        #print("Step2.filter by cache and vec limitation configs space is: ", len(index))
        print("if gemm: Tm,Tn,Tk = fx,fy,fk;if conv2d: ow_bn,oc_bn,ic_bn = fx,fy,fk")
        print('fx:', sorted(set(dataframe['fx'].tolist())))
        print('fy:', sorted(set(dataframe['fy'].tolist())))
        print('fk:', sorted(set(dataframe['fk'].tolist())))
        #new_dataframe = pd.concat([dataframe['fx'],dataframe['fy'], dataframe['fk'],dataframe['index']], axis=1)
        #np.savetxt("filter_config_space.txt", new_dataframe.values)
        print("***********************filterConfigSpace end.***************************")
        index = list(set(index))
        print("Step2.filter by cache and vec limitation configs space is: ", len(index))
        return index

    def next_batch_filter(self, batch_size, filter_configs):
        '''
        :param self:
        :param batch_size: batch_size  一般为64或128
        :param filter_configs: list    用于接收config_filter的返回index列表
        :return:
        '''
        filter_configs_set = set(filter_configs)
        index = filter_configs_set - self.visited
        index = list(index)
        if len(index) > batch_size:
            slice = random.sample(index, batch_size)
        else:
            slice = index
        # print("next_batch_filter slice:",slice)
        ret = []
        for i in slice:
            ret.append(self.space.get(i))
            self.visited.add(i)
        self.rets.append(ret)
        return ret

    def init_batch_by_Complexity(self, M, K, N, batch_size, sch, filter_configs, i):
        '''
        :param self:
        :param batch_size: batch_size一般为64或128
        :param filter_configs: list    用于接收config_filter的返回index列表
        :return:
        '''
        filter_configs_set = set(filter_configs)
        index = filter_configs_set - self.visited
        index = list(index)
        if len(index) > batch_size:
            slice = complexityRecommend(M, K, N, batch_size, sch, i)
            # slice = random.sample(index, batch_size)
        else:
            slice = index
        # print("next_batch_filter slice:",slice)
        ret = []
        for i in slice:
            ret.append(self.space.get(i))
            self.visited.add(i)
        self.rets.append(ret)
        return ret

    def update(self, inputs, results):
        for inp, res in zip(inputs, results):
            index = inp.config.index

            if res.error_no == 0:
                self.xs.append(index)
                flops = inp.task.flop / np.mean(res.costs)
                self.flops_max = max(self.flops_max, flops)
                self.ys.append(flops)
            else:
                self.xs.append(index)
                self.ys.append(0.0)

        # if we have enough new training samples

        if len(self.xs) >= self.plan_size * (self.train_ct + 1) and self.flops_max > 1e-6:
            self.cost_model.fit(self.xs, self.ys, self.plan_size)
            if self.diversity_filter_ratio:
                candidate = self.model_optimizer.find_maximums(self.cost_model,
                                                               self.plan_size * self.diversity_filter_ratio,
                                                               self.visited)

                scores = self.cost_model.predict(candidate)
                knobs = [point2knob(x, self.dims) for x in candidate]

                pick_index = submodular_pick(0.2 * scores, knobs, self.plan_size, knob_weight=1)  # 原本这0.2是0
                maximums = np.array(candidate)[pick_index]
            else:
                maximums = self.model_optimizer.find_maximums(self.cost_model, self.plan_size, self.visited)


            self.trials = maximums
            self.trial_pt = 0
            self.train_ct += 1

    def load_history(self, data_set):
        # set in_tuning as True to make the feature extraction consistent
        GLOBAL_SCOPE.in_tuning = True

        # fit base model
        base_model = self.cost_model.spawn_base_model()
        # fit_log
        success = base_model.fit_log(data_set, self.plan_size)
        self.transfer_flag = success
        if not success:
            GLOBAL_SCOPE.in_tuning = False
            return

        # use base model to select initial points
        if not self.trials:
            # no plan yet, use base model to select initial trials
            maximums = self.model_optimizer.find_maximums(base_model, self.plan_size, self.visited)
            self.trials = maximums
            self.trial_pt = 0

        self.cost_model.load_basemodel(base_model)
        GLOBAL_SCOPE.in_tuning = False

    def has_next(self):
        return len(self.visited) < len(self.space)


def point2knob(p, dims):
    """convert point form (single integer) to knob form (vector)"""
    knob = []
    for dim in dims:
        knob.append(p % dim)
        p //= dim
    return knob


def knob2point(knob, dims):
    """convert knob form (vector) to point form (single integer)"""
    p = 0
    for j, k in enumerate(knob):
        p += int(np.prod(dims[:j])) * k
    return p


def submodular_pick(scores, knobs, n_pick, knob_weight=1.0):
    """Run greedy optimization to pick points with regard to both score and diversity.

    DiversityScore = knob_weight * number of unique knobs in the selected set
    Obj = sum(scores[i] for i in pick) + DiversityScore
    Note that this objective function is a monotone submodular function.

    Parameters
    ----------
    scores: Array of float
        score of every points
    knobs: Array of Array of int
        feature vector (tunable knobs) of every points
    n_pick: int
        number of points to pick
    knob_weight: float
        weight of an unique knob feature
    """
    n = len(scores)
    assert n == len(knobs)
    n_knobs = len(knobs[0])

    knobs_set = [set() for _ in range(n_knobs)]

    ret = []
    remain = list(range(len(scores)))

    for _ in range(n_pick):
        max_x = -1
        max_delta = -1e9

        for x in remain:
            tmp_delta = scores[x]
            for i in range(n_knobs):
                if knobs[x][i] not in knobs_set[i]:
                    tmp_delta += knob_weight

            if tmp_delta > max_delta:
                max_delta, max_x = tmp_delta, x

        ret.append(max_x)
        remain.remove(max_x)
        for i in range(n_knobs):
            knobs_set[i].add(knobs[max_x][i])

    return ret
