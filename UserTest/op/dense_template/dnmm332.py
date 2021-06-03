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


def dnmm332(M, K, N, dtype):
    data = tvm.placeholder((M, K), name='data', dtype=dtype)
    weight = tvm.placeholder((N, K), name='weight', dtype=dtype)
    # create tuning space
    cfg = autotvm.get_config()
    cfg.define_split("tile_y", M, num_outputs=3, policy='factors')
    cfg.define_split("tile_x", N, num_outputs=3, policy='factors')
    cfg.define_split("tile_k", K, num_outputs=2, policy='factors')
    vec = cfg["tile_k"].size[-1]
    k = tvm.reduce_axis((0, K // vec), "k")
    CC = tvm.compute((M, N, vec),
                     lambda z, y, x: tvm.sum(data[z, k * vec + x].astype(dtype) * weight[y, k * vec + x].astype(dtype),
                                             axis=k))

    kk = tvm.reduce_axis((0, vec), "kk")
    C = tvm.compute((M, N), lambda y, x: tvm.sum(CC[y, x, kk], axis=kk))
    s = tvm.create_schedule(C.op)

    y, x = s[C].op.axis
    kk, = s[C].op.reduce_axis
    yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    # yo, xo, yi, xi = s[C].tile(C.op.axis[0], C.op.axis[1], 4, 4)
    s[C].reorder(yt, xt, yo, xo, yi, xi)
    xyt = s[C].fuse(yt, xt)
    xyo = s[C].fuse(yo, xo)
    s[C].parallel(xyt)
    s[C].unroll(kk)

    CC, = s[C].op.input_tensors
    s[CC].compute_at(s[C], xyo)
    z, y, x = s[CC].op.axis
    k, = s[CC].op.reduce_axis
    yz = s[CC].fuse(z, y)
    s[CC].reorder(k, yz, x)
    s[CC].unroll(yz)
    s[CC].vectorize(x)
    # print(tvm.lower(s, [data, weight, C], simple_mode=True))
    return s, [data, weight, C]

if __name__ == '__main__':
    n_trial = 2213700
    M = int(sys.argv[1])
    K = int(sys.argv[2])
    N = int(sys.argv[3])
    mytuner(n_trial, M, K, N, dnmm332, 400)
'''

'''
