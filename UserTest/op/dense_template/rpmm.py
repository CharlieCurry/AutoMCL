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


def rpmm(M, K, N, dtype):
    data = tvm.placeholder((M, K), name='data', dtype=dtype)
    weight = tvm.placeholder((N, K), name='weight', dtype=dtype)
    # create tuning space
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", M, num_outputs=3, policy="factors")
    cfg.define_split("tile_x", N, num_outputs=3, policy="factors")
    cfg.define_split("tile_k", K, num_outputs=2, policy="factors")

    packw_bn = cfg["tile_x"].size[-1]
    packw_shape = (N // packw_bn, K, packw_bn)
    packw = tvm.compute(packw_shape,
                        lambda z, y, x: weight[z * packw_bn + x, y], name="packed_weight")

    k = tvm.reduce_axis((0, K), name="k")
    C = tvm.compute((M, N),
                    lambda y, x: tvm.sum(
                        data[y, k].astype(dtype) *
                        packw[tvm.indexdiv(x, packw_bn), k, tvm.indexmod(x, packw_bn)].astype(dtype),
                        axis=k))

    s = tvm.create_schedule(C.op)
    A, packedB = s[C].op.input_tensors
    CC = s.cache_write(C, "global")
    y, x = s[C].op.axis
    k, = s[CC].op.reduce_axis
    yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(yt, xt, yo, xo, yi, xi)
    xyt = s[C].fuse(yt, xt)
    s[C].parallel(xyt)
    xyo = s[C].fuse(yo, xo)
    s[C].unroll(yi)
    s[C].vectorize(xi)

    s[CC].compute_at(s[C], xyo)
    y, x = s[CC].op.axis
    ko, ki = cfg["tile_k"].apply(s, CC, k)
    s[CC].reorder(ko, ki, y, x)
    s[CC].vectorize(x)
    s[CC].unroll(y)
    s[CC].unroll(ki)

    z, y, x = s[packedB].op.axis
    s[packedB].reorder(z, x, y)
    s[packedB].parallel(z)
    return s, [data, weight, C]

if __name__ == '__main__':
    n_trial = 2213700
    M = int(sys.argv[1])
    K = int(sys.argv[2])
    N = int(sys.argv[3])
    mytuner(n_trial, M, K, N, rpmm, 400)
'''

'''
