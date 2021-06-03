# -*- coding:utf8 -*
import logging
import numpy as np
import tvm
import sys
import math
import timeit
from tvm import autotvm


def numpyBaseline(M, K, N):
    np_repeat = 100
    np_runing_time = timeit.timeit(setup='import numpy\n'
                                         'M = ' + str(M) + '\n'
                                                           'K = ' + str(K) + '\n'
                                                                             'N = ' + str(N) + '\n'
                                                                                               'dtype = "float32"\n'
                                                                                               'a = numpy.random.rand(M, K).astype(dtype)\n'
                                                                                               'b = numpy.random.rand(K, N).astype(dtype)\n',
                                   stmt='answer = numpy.dot(a, b)',
                                   number=np_repeat)
    print("Numpy running time: %f" % (np_runing_time / np_repeat))


def mytuner(n_trial, M, K, N, gemm_impl_schedule, early_stopping):
    target = 'llvm'
    dtype = 'float32'
    ctx = tvm.context(target, 0)
    src = str(M) + "x" + str(K) + "x" + str(N)
    recordFileName = gemm_impl_schedule.__name__+'_XGBTuner_matmul_' + src + 'tuner00.log'
    print(src)
    a_np = np.random.rand(M, K).astype(dtype)
    b_np = np.random.rand(K, N).astype(dtype)
    c_np = a_np.dot(b_np)
    a = tvm.nd.array(a_np, ctx)
    b = tvm.nd.array(b_np, ctx)
    c = tvm.nd.array(c_np, ctx)
    bt = tvm.nd.array(b_np.T, ctx)
    # numpyBaseline(M,K,N)
    tsk = autotvm.task.create(gemm_impl_schedule, args=(M, K, N, dtype), target=target)
    n_trial = min(n_trial, len(tsk.config_space))
    # print(tsk.config_space)
    measure_option = autotvm.measure_option(builder='local', runner=autotvm.LocalRunner(number=10))
    # print("XGBoost:")
    XGBtuner = autotvm.tuner.XGBTuner(tsk)
    res = []
    for j in range(len(tsk.config_space)):
        res.append(XGBtuner.space.get(j + 1))
    with open("res.txt", "w", encoding='utf-8') as f:
        for line in res:
            f.write(str(line) + '\n')
        f.close()
    # XGBtuner.tune(n_trial=n_trial,
    #               early_stopping=early_stopping,
    #               measure_option=measure_option,
    #               callbacks=[autotvm.callback.progress_bar(n_trial), autotvm.callback.log_to_file(recordFileName)])
    XGBtuner.tuneMCL(n_trial=n_trial,
                  early_stopping=early_stopping,
                  measure_option=measure_option,
                  callbacks=[autotvm.callback.progress_bar(n_trial), autotvm.callback.log_to_file(recordFileName)],
                 useFilter=True, useRecommend=False, sch=str(gemm_impl_schedule.__name__))

    autotvm.record.pick_best(recordFileName, recordFileName + ".best")

def tmm(M, K, N, dtype):
    data = tvm.placeholder((M, K), name='data', dtype=dtype)
    weight = tvm.placeholder((N, K), name='weight', dtype=dtype)
    z = tvm.reduce_axis((0, K), name='z')
    C = tvm.compute((M, N), lambda i, j: tvm.sum(data[i, z] * weight[j, z], axis=z), name='C')
    s = tvm.create_schedule(C.op)
    z = s[C].op.reduce_axis[0]
    x, y = s[C].op.axis
    cfg = autotvm.get_config()
    cfg.define_split("tile_x", x, num_outputs=3)
    cfg.define_split("tile_y", y, num_outputs=3)
    cfg.define_split("tile_z", z, num_outputs=2)
    xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    zo, zi = cfg["tile_z"].apply(s, C, z)
    s[C].reorder(xt, yt, xo, yo, zo, xi, zi, yi)
    xyt = s[C].fuse(xt, yt)
    xyo = s[C].fuse(xo, yo)
    s[C].vectorize(yi)
    # s[C].unroll(zi)
    s[C].parallel(xyt)
    # print(tvm.lower(s, [data, weight, C], simple_mode=True))
    return s, [data, weight, C]
if __name__ == '__main__':
    n_trial = 2213700
    M = int(sys.argv[1])
    K = int(sys.argv[2])
    N = int(sys.argv[3])
    mytuner(n_trial, M, K, N, tmm, 400)
'''

'''
