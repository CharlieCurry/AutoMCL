# -*- coding:utf8 -*
import logging
import numpy as np
import tvm
import sys
import time
from tvm import relay
import math
import timeit
import topi
from tvm import autotvm
def im2col(img, ksize, stride=1):
    N, C, H, W = img.shape
    #print(img)
    out_h = (H - ksize) // stride + 1
    out_w = (W - ksize) // stride + 1
    col = np.empty((N * out_h * out_w, ksize * ksize * C))
    outsize = out_w * out_h
    for y in range(out_h):
        y_min = y * stride
        y_max = y_min + ksize
        y_start = y * out_w
        for x in range(out_w):
            x_min = x * stride
            x_max = x_min + ksize
            col[y_start+x::outsize, :] = img[:, :, y_min:y_max, x_min:x_max].reshape(N, -1)
    return col

def conv(X, W, stride, padding, dilation):
    FN, C, ksize, ksize= W.shape
    s1 = time.time()
    if padding >= 0:
        p = padding
        X = np.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), 'constant')
    e1 = time.time()
    N, C, H, _ = X.shape

    ksize = (ksize - 1)*dilation + 1
    col = im2col(X, ksize, stride)
    WT = W.reshape(W.shape[0], -1).transpose()
    z = np.dot(col, WT).astype("float32")
    z = z.reshape(N, int(z.shape[0] / N), -1)
    out_h = (H - ksize) // stride + 1
    out = z.reshape(N,  FN, out_h, -1)
    return out

def infer_pad(data, data_pad):
    if data_pad is None:
        return 0, 0
    _, _, IH, IW = data.shape
    _, _, TH, TW = data_pad.shape
    hpad = (TH - IH) // 2
    wpad = (TW - IW) // 2
    return int(hpad), int(wpad)

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




def conv2d0(data_shape, kernel_shape, padding, strides, dilation, dtype):
    B, IC, DH, DW  = data_shape
    OC, IC, KH, KW = kernel_shape
    data = tvm.placeholder(data_shape, name='data', dtype=dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=dtype)
    idxmod = tvm.indexmod
    idxdiv = tvm.indexdiv
    cfg = autotvm.get_config()
    out_dtype = dtype
    pt, pl, pb, pr = padding
    sh, sw = strides
    dilation_h, dilation_w = dilation

    OH = (DH - KH + pt + pb) // sh + 1
    OW = (DW - KW + pl + pr) // sw + 1

    HSTR, WSTR = strides

    pad_h = pt + pb
    pad_w = pl + pr

    pad_height = DH + pad_h
    pad_width = DW + pad_w
    is_kernel_1x1 = KH == 1 and KW == 1
    dilated_kernel_h = (KH - 1) * dilation_h + 1
    dilated_kernel_w = (KW - 1) * dilation_w + 1
    out_height = (DH + pad_h - dilated_kernel_h) // HSTR + 1
    out_width = (DW + pad_w - dilated_kernel_w) // WSTR + 1
    #print(out_height,out_width)
    # pack data
    DOPAD = (pad_h != 0 or pad_w != 0)
    if DOPAD:
        data_pad = topi.nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")
    else:
        data_pad = data

    cfg.define_split("tile_ic", IC, num_outputs=2)
    cfg.define_split("tile_oc", OC, num_outputs=2)
    cfg.define_split("tile_ow", OW, num_outputs=2, filter=lambda y: y.size[-1] <= 64)
    unroll_kw = False
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if OH > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])
        unroll_kw = cfg["unroll_kw"].val


    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]

    shape = (B, IC // ic_bn, pad_height, ic_bn, pad_width)
    # c from 0 to ic_bn-1
    # transposed

    data_vec = tvm.compute(shape,
                           lambda n, C, h, c, w: data_pad[n, C * ic_bn + c, h, w],
                           name='data_vec')

    # pack kernel
    shape = (OC // oc_bn, IC // ic_bn, KH, KW, ic_bn, oc_bn)

    # kernel shape to "shape"

    kernel_vec = tvm.compute(shape,
                             lambda CO, CI, h, w, ci, co:
                             kernel[CO * oc_bn + co, CI * ic_bn + ci, h, w],
                             name='kernel_vec')

    # convolution
    ic = tvm.reduce_axis((0, IC), name='ic')
    kh = tvm.reduce_axis((0, KH), name='kh')
    kw = tvm.reduce_axis((0, KW), name='kw')

    oshape = (B, OC // oc_bn, out_height, out_width, oc_bn)
    # defination of convolution, but with oc and ic tiled

    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
    tvm.sum(data_vec[n, idxdiv(ic, ic_bn), oh * HSTR + kh * dilation_h,
                     idxmod(ic, ic_bn), ow * WSTR + kw * dilation_w].astype(out_dtype) *
            kernel_vec[oc_chunk, idxdiv(ic, ic_bn), kh, kw, idxmod(ic, ic_bn), oc_block].astype(out_dtype),
            axis=[ic, kh, kw]), name='conv')


    unpack_shape = (B, OC, out_height, out_width)
    unpack = tvm.compute(unpack_shape,
                         lambda n, c, h, w: conv[n, idxdiv(c, oc_bn), h, w, idxmod(c, oc_bn)]
                         .astype(out_dtype),
                         name='output_unpack',
                         tag='conv2d_nchw')

    # print("common:_schedule_conv!!!")
    ic_bn, oc_bn, reg_n = (cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1],
                                      cfg["tile_ow"].size[-1])

    s = tvm.create_schedule(unpack.op)
    # no stride and padding info here
    padding = infer_pad(data, data_pad)
    HPAD, WPAD = padding
    DOPAD = (HPAD != 0 or WPAD != 0)

    A, W = data, kernel_vec
    A0, A1 = data_pad, data_vec


    # schedule data
    if DOPAD:
        s[A0].compute_inline()

    batch, ic_chunk, ih, ic_block, iw = s[A1].op.axis
    parallel_axis = s[A1].fuse(batch, ic_chunk, ih)
    s[A1].parallel(parallel_axis)


    # schedule kernel pack
    oc_chunk, ic_chunk, oh, ow, ic_block, oc_block = s[W].op.axis
    s[W].reorder(oc_chunk, oh, ic_chunk, ow, ic_block, oc_block)
    if oc_bn > 1:
        s[W].vectorize(oc_block)
    parallel_axis = s[W].fuse(oc_chunk, oh)
    s[W].parallel(parallel_axis)

    # schedule conv
    # conv_out： input for current convolution layer
    # output :   output for current convolution layer
    # last:      output of network
    C, O0, O = conv, unpack, unpack

    CC = s.cache_write(C, 'global')
    _, oc_chunk, oh, ow, oc_block = s[C].op.axis
    # print(s[C].op.axis)
    ow_chunk, ow_block = s[C].split(ow, factor=reg_n)
    s[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    s[C].fuse(oc_chunk, oh)
    s[C].vectorize(oc_block)

    s[CC].compute_at(s[C], ow_chunk)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    ic, kh, kw = s[CC].op.reduce_axis

    ow_chunk, ow_block = s[CC].split(ow, factor=reg_n)
    ic_chunk, ic_block = s[CC].split(ic, factor=ic_bn)

    if unroll_kw:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, ic_block, kw, ow_block, oc_block)
        s[CC].unroll(kw)
    else:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, kw, ic_block, ow_block, oc_block)

    s[CC].fuse(oc_chunk, oh)
    s[CC].vectorize(oc_block)
    s[CC].unroll(ow_block)

    if O0 != O:
        s[O0].compute_inline()

    batch, oc, oh, ow = s[O].op.axis
    ow_chunk, ow_block = s[O].split(ow, factor=reg_n)
    oc_chunk, oc_block = s[O].split(oc, factor=oc_bn)
    s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[O].fuse(batch, oc_chunk, oh)
    s[C].compute_at(s[O], parallel_axis)
    s[O].vectorize(oc_block)

    s[O].parallel(parallel_axis)
    #print(tvm.lower(s, [data, kernel, O], simple_mode=True))
    return s, [data, kernel, O]

def conv2d0_1x1(data_shape, kernel_shape, padding, strides, dilation, dtype):
    B, IC, DH, DW  = data_shape
    OC, IC, KH, KW = kernel_shape
    data = tvm.placeholder(data_shape, name='data', dtype=dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=dtype)
    idxmod = tvm.indexmod
    idxdiv = tvm.indexdiv
    cfg = autotvm.get_config()
    out_dtype = dtype
    pt, pl, pb, pr = padding
    sh, sw = strides
    dilation_h, dilation_w = dilation

    OH = (DH - KH + pt + pb) // sh + 1
    OW = (DW - KW + pl + pr) // sw + 1

    HSTR, WSTR = strides

    pad_h = pt + pb
    pad_w = pl + pr

    pad_height = DH + pad_h
    pad_width = DW + pad_w
    is_kernel_1x1 = KH == 1 and KW == 1
    dilated_kernel_h = (KH - 1) * dilation_h + 1
    dilated_kernel_w = (KW - 1) * dilation_w + 1
    out_height = (DH + pad_h - dilated_kernel_h) // HSTR + 1
    out_width = (DW + pad_w - dilated_kernel_w) // WSTR + 1
    #print(out_height,out_width)
    # pack data
    DOPAD = (pad_h != 0 or pad_w != 0)
    if DOPAD:
        data_pad = topi.nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")
    else:
        data_pad = data

    cfg.define_split("tile_ic", IC, num_outputs=2)
    cfg.define_split("tile_oc", OC, num_outputs=2)
    cfg.define_split("tile_ow", OW, num_outputs=2, filter=lambda y: y.size[-1] <= 64)
    cfg.define_split("tile_oh", OH, num_outputs=2, filter=lambda y: y.size[-1] <= 64)
    unroll_kw = False
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if OH > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])
        unroll_kw = cfg["unroll_kw"].val


    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]

    shape = (B, IC // ic_bn, pad_height, ic_bn, pad_width)
    # c from 0 to ic_bn-1
    # transposed

    data_vec = tvm.compute(shape,
                           lambda n, C, h, c, w: data_pad[n, C * ic_bn + c, h, w],
                           name='data_vec')

    # pack kernel
    shape = (OC // oc_bn, IC // ic_bn, KH, KW, ic_bn, oc_bn)

    # kernel shape to "shape"

    kernel_vec = tvm.compute(shape,
                             lambda CO, CI, h, w, ci, co:
                             kernel[CO * oc_bn + co, CI * ic_bn + ci, h, w],
                             name='kernel_vec')

    # convolution
    ic = tvm.reduce_axis((0, IC), name='ic')
    kh = tvm.reduce_axis((0, KH), name='kh')
    kw = tvm.reduce_axis((0, KW), name='kw')

    oshape = (B, OC // oc_bn, out_height, out_width, oc_bn)
    # defination of convolution, but with oc and ic tiled

    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
    tvm.sum(data_vec[n, idxdiv(ic, ic_bn), oh * HSTR + kh * dilation_h,
                     idxmod(ic, ic_bn), ow * WSTR + kw * dilation_w].astype(out_dtype) *
            kernel_vec[oc_chunk, idxdiv(ic, ic_bn), kh, kw, idxmod(ic, ic_bn), oc_block].astype(out_dtype),
            axis=[ic, kh, kw]), name='conv')


    unpack_shape = (B, OC, out_height, out_width)
    unpack = tvm.compute(unpack_shape,
                         lambda n, c, h, w: conv[n, idxdiv(c, oc_bn), h, w, idxmod(c, oc_bn)]
                         .astype(out_dtype),
                         name='output_unpack',
                         tag='conv2d_nchw')

    # print("common:_schedule_conv!!!")
    ic_bn, oc_bn, oh_factor, ow_factor = (cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1],
                                          cfg["tile_oh"].size[-1], cfg["tile_ow"].size[-1])
    s = tvm.create_schedule(unpack.op)
    # no stride and padding info here
    padding = infer_pad(data, data_pad)
    HPAD, WPAD = padding
    DOPAD = (HPAD != 0 or WPAD != 0)

    A, W = data, kernel_vec
    A0, A1 = data_pad, data_vec
    # schedule data
    if DOPAD:
        s[A0].compute_inline()
    batch, ic_chunk, ih, ic_block, iw = s[A1].op.axis
    parallel_axis = s[A1].fuse(batch, ic_chunk, ih)
    s[A1].parallel(parallel_axis)

    # schedule kernel pack
    oc_chunk, ic_chunk, oh, ow, ic_block, oc_block = s[W].op.axis
    s[W].reorder(oc_chunk, oh, ic_chunk, ow, ic_block, oc_block)
    if oc_bn > 1:
        s[W].vectorize(oc_block)
    parallel_axis = s[W].fuse(oc_chunk, oh)
    s[W].parallel(parallel_axis)

    C, O0, O = conv, unpack, unpack
    CC = s.cache_write(C, 'global')

    batch, oc_chunk, oh, ow, oc_block = s[C].op.axis
    oh_outer, oh_inner = s[C].split(oh, factor=oh_factor)
    s[C].vectorize(oc_block)

    s[CC].compute_at(s[C], oh_outer)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    ic, _, _ = s[CC].op.reduce_axis

    ic_chunk, ic_block = s[CC].split(ic, factor=ic_bn)

    oh_outer, oh_inner = s[CC].split(oh, factor=oh_factor)
    ow_outer, ow_inner = s[CC].split(ow, factor=ow_factor)

    s[CC].reorder(oc_chunk, oh_outer, ow_outer, ic_chunk, ic_block, oh_inner, ow_inner, oc_block)
    s[CC].vectorize(oc_block)

    s[CC].unroll(ow_inner)
    s[CC].unroll(oh_inner)

    if O0 != O:
        s[O0].compute_inline()
    batch, oc, oh, ow = s[O].op.axis

    oc_chunk, oc_block = s[O].split(oc, factor=oc_bn)
    oh_outer, oh_inner = s[O].split(oh, factor=oh_factor)
    ow_outer, ow_inner = s[O].split(ow, factor=ow_factor)
    s[O].reorder(oc_chunk, oh_outer, ow_outer, oh_inner, ow_inner, oc_block)

    parallel_axis = s[O].fuse(batch, oc_chunk, oh_outer)
    s[C].compute_at(s[O], parallel_axis)
    s[O].vectorize(oc_block)

    s[O].parallel(parallel_axis)
    return s, [data, kernel, O]


def conv2d_opt(data_shape, kernel_shape, padding, strides, dilation, dtype):
    B, IC, DH, DW  = data_shape
    OC, IC, KH, KW = kernel_shape
    data = tvm.placeholder(data_shape, name='data', dtype=dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=dtype)
    idxmod = tvm.indexmod
    idxdiv = tvm.indexdiv
    cfg = autotvm.get_config()
    out_dtype = dtype
    pt, pl, pb, pr = padding
    sh, sw = strides
    dilation_h, dilation_w = dilation

    OH = (DH - KH + pt + pb) // sh + 1
    OW = (DW - KW + pl + pr) // sw + 1

    HSTR, WSTR = strides

    pad_h = pt + pb
    pad_w = pl + pr

    pad_height = DH + pad_h
    pad_width = DW + pad_w
    is_kernel_1x1 = KH == 1 and KW == 1
    dilated_kernel_h = (KH - 1) * dilation_h + 1
    dilated_kernel_w = (KW - 1) * dilation_w + 1
    out_height = (DH + pad_h - dilated_kernel_h) // HSTR + 1
    out_width = (DW + pad_w - dilated_kernel_w) // WSTR + 1
    #print(out_height,out_width)
    # pack data
    DOPAD = (pad_h != 0 or pad_w != 0)
    if DOPAD:
        data_pad = topi.nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")
    else:
        data_pad = data

    cfg.define_split("tile_ic", IC, num_outputs=2)
    cfg.define_split("tile_oc", OC, num_outputs=2)
    cfg.define_split("tile_ow", OW, num_outputs=2, filter=lambda y: y.size[-1] <= 64)
    unroll_kw = False
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if OH > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])
        unroll_kw = cfg["unroll_kw"].val


    ic_bn, oc_bn = cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1]

    shape = (B, IC // ic_bn, pad_height, ic_bn, pad_width)
    # c from 0 to ic_bn-1
    # transposed
    t1 = time.time()
    data_vec = tvm.compute(shape,
                           lambda n, C, h, c, w: data_pad[n, C * ic_bn + c, h, w],
                           name='data_vec')
    #print("data -> data_vec",time.time()-t1)
    # pack kernel
    shape = (OC // oc_bn, IC // ic_bn, KH, KW, ic_bn, oc_bn)

    # kernel shape to "shape"
    t2 = time.time()
    kernel_vec = tvm.compute(shape,
                             lambda CO, CI, h, w, ci, co:
                             kernel[CO * oc_bn + co, CI * ic_bn + ci, h, w],
                             name='kernel_vec')
    #print("kernel -> kernel_vec:",time.time()-t2)
    # convolution
    ic = tvm.reduce_axis((0, IC), name='ic')
    kh = tvm.reduce_axis((0, KH), name='kh')
    kw = tvm.reduce_axis((0, KW), name='kw')

    oshape = (B, OC // oc_bn, out_height, out_width, oc_bn)
    # defination of convolution, but with oc and ic tiled
    t3 = time.time()
    conv = tvm.compute(oshape, lambda n, oc_chunk, oh, ow, oc_block:
    tvm.sum(data_vec[n, idxdiv(ic, ic_bn), oh * HSTR + kh * dilation_h,
                     idxmod(ic, ic_bn), ow * WSTR + kw * dilation_w].astype(out_dtype) *
            kernel_vec[oc_chunk, idxdiv(ic, ic_bn), kh, kw, idxmod(ic, ic_bn), oc_block].astype(out_dtype),
            axis=[ic, kh, kw]), name='conv')
    #print("data_vec * kernel_vec -> conv:",time.time()-t3)
    t4 = time.time()
    unpack_shape = (B, OC, out_height, out_width)
    unpack = tvm.compute(unpack_shape,
                         lambda n, c, h, w: conv[n, idxdiv(c, oc_bn), h, w, idxmod(c, oc_bn)]
                         .astype(out_dtype),
                         name='output_unpack',
                         tag='conv2d_nchw')
    #print("conv -> unpack:",time.time()-t4)
    # print("common:_schedule_conv!!!")
    ic_bn, oc_bn, reg_n = (cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1],
                                      cfg["tile_ow"].size[-1])

    s = tvm.create_schedule(unpack.op)
    # no stride and padding info here
    padding = infer_pad(data, data_pad)
    HPAD, WPAD = padding
    DOPAD = (HPAD != 0 or WPAD != 0)

    A, W = data, kernel_vec
    A0, A1 = data_pad, data_vec


    # schedule data
    if DOPAD:
        s[A0].compute_inline()

    batch, ic_chunk, ih, ic_block, iw = s[A1].op.axis
    parallel_axis = s[A1].fuse(batch, ic_chunk, ih)
    s[A1].parallel(parallel_axis)


    # schedule kernel pack
    oc_chunk, ic_chunk, oh, ow, ic_block, oc_block = s[W].op.axis
    s[W].reorder(oc_chunk, ic_chunk, oh, ow, ic_block, oc_block)
    if oc_bn > 1:
        s[W].vectorize(oc_block)
    parallel_axis = s[W].fuse(oc_chunk, ic_chunk)
    s[W].parallel(parallel_axis)

    # schedule conv
    # conv_out： input for current convolution layer
    # output :   output for current convolution layer
    # last:      output of network
    C, O0, O = conv, unpack, unpack

    CC = s.cache_write(C, 'global')
    _, oc_chunk, oh, ow, oc_block = s[C].op.axis
    # print(s[C].op.axis)
    ow_chunk, ow_block = s[C].split(ow, factor=reg_n)
    s[C].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    s[C].fuse(oc_chunk, oh)
    s[C].vectorize(oc_block)

    s[CC].compute_at(s[C], ow_chunk)
    _, oc_chunk, oh, ow, oc_block = s[CC].op.axis
    ic, kh, kw = s[CC].op.reduce_axis

    ow_chunk, ow_block = s[CC].split(ow, factor=reg_n)
    ic_chunk, ic_block = s[CC].split(ic, factor=ic_bn)

    if unroll_kw:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, ic_block, kw, ow_block, oc_block)
        s[CC].unroll(kw)
    else:
        s[CC].reorder(oc_chunk, oh, ow_chunk, ic_chunk, kh, kw, ic_block, ow_block, oc_block)

    s[CC].fuse(oc_chunk, oh)
    s[CC].vectorize(oc_block)
    s[CC].unroll(ow_block)

    if O0 != O:
        s[O0].compute_inline()

    batch, oc, oh, ow = s[O].op.axis
    ow_chunk, ow_block = s[O].split(ow, factor=reg_n)
    oc_chunk, oc_block = s[O].split(oc, factor=oc_bn)
    s[O].reorder(oc_chunk, oh, ow_chunk, ow_block, oc_block)
    parallel_axis = s[O].fuse(batch, oc_chunk, oh)
    s[C].compute_at(s[O], parallel_axis)
    s[O].vectorize(oc_block)

    s[O].parallel(parallel_axis)
    #print(tvm.lower(s, [data, kernel, O], simple_mode=True))
    return s, [data, kernel, O]

def dnmm332_opreator(data,data_shape,weight,weight_shape, dtype,cfg):
    M , K = data_shape
    N , _ = weight_shape
    # create tuning space
    vec = cfg["tile_k"].size[-1]
    k = tvm.reduce_axis((0, K // vec), "k")
    CC = tvm.compute((M, N, vec),
                     lambda z, y, x: tvm.sum(data[z, k * vec + x].astype(dtype) * weight[y, k * vec + x].astype(dtype),
                                             axis=k))
    kk = tvm.reduce_axis((0, vec), "kk")
    C = tvm.compute((M, N), lambda y, x: tvm.sum(CC[y, x, kk], axis=kk))
    return C
def dnmm332_schedule(C,cfg,s):
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
    return s
def dnmm_opreator(data,data_shape,weight,weight_shape, dtype,cfg):
    M , K = data_shape
    N , _ = weight_shape
    # create tuning space
    vec = cfg["tile_k"].size[-1]
    k = tvm.reduce_axis((0, K // vec), "k")
    CC = tvm.compute((M, N, vec),
                     lambda z, y, x: tvm.sum(data[z, k * vec + x].astype(dtype) * weight[y, k * vec + x].astype(dtype),
                                             axis=k))
    kk = tvm.reduce_axis((0, vec), "kk")
    C = tvm.compute((M, N), lambda y, x: tvm.sum(CC[y, x, kk], axis=kk))
    return C
def dnmm_schedule(C,cfg,s):
    x, y = s[C].op.axis
    kk, = s[C].op.reduce_axis
    # ic_bn, oc_bn, ow_bn, unroll_kw = (cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1],
    #                                   cfg["tile_ow"].size[-1], cfg["unroll_kw"].val)
    # # xo, xi = s[C].split(x, factor=oc_bn)
    # # yo, yi = s[C].split(y, factor=KH*KW*ic_bn)
    xo, xi = cfg["tile_x"].apply(s, C, x)
    yo, yi = cfg["tile_y"].apply(s, C, y)
    # yo, xo, yi, xi = s[C].tile(C.op.axis[0], C.op.axis[1], 4, 4)
    s[C].reorder(xo,yo,xi,yi)
    xyo = s[C].fuse(xo, yo)
    s[C].parallel(xyo)
    s[C].unroll(kk)

    CC, = s[C].op.input_tensors
    s[CC].compute_at(s[C], xyo)
    z, y, x = s[CC].op.axis
    k, = s[CC].op.reduce_axis
    yz = s[CC].fuse(z, y)
    s[CC].reorder(k, yz, x)
    s[CC].unroll(yz)
    s[CC].vectorize(x)
    return s
def infer_datapad_im2col_shape(inputSize, filterSize, stride, dilation):
    N, IC, H, W = inputSize
    FN, OC, ksize, ksize = filterSize
    out_h = (H  - (ksize-1)*dilation - 1) // stride + 1
    out_w = (W  - (ksize-1)*dilation - 1) // stride + 1
    M = N * out_h * out_w
    K = ksize * ksize * IC
    N = FN*OC*ksize*ksize // K
    return M,K,N
def convTestSchedule1(data_shape, kernel_shape, padding, strides, dilation, dtype):
    B, IC, DH, DW  = data_shape
    OC, IC, KH, KW = kernel_shape
    data = tvm.placeholder(data_shape, name='data', dtype=dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=dtype)
    cfg = autotvm.get_config()
    out_dtype = dtype
    pt, pl, pb, pr = padding
    sh, sw = strides
    dilation_h, dilation_w = dilation

    OH = (DH - KH + pt + pb) // sh + 1
    OW = (DW - KW + pl + pr) // sw + 1

    HSTR, WSTR = strides

    pad_h = pt + pb
    pad_w = pl + pr

    pad_height = dh + pad_h
    pad_width = dw + pad_w
    is_kernel_1x1 = KH == 1 and KW == 1
    dilated_kernel_h = (kh - 1) * dilation_h + 1
    dilated_kernel_w = (kw - 1) * dilation_w + 1
    out_height = (dh + pad_h - dilated_kernel_h) // HSTR + 1
    out_width = (dw + pad_w - dilated_kernel_w) // WSTR + 1

    outsize = out_height * out_width
    # pack data
    DOPAD = (pad_h != 0 or pad_w != 0)
    if DOPAD:
        data_pad = topi.nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")
    else:
        data_pad = data


    M = int(B * out_height * out_width)
    K = int(dilated_kernel_h * dilated_kernel_h * IC)
    N = int(OC)


    idxdiv = tvm.indexdiv
    idxmod = tvm.indexmod
    #print(M,K,N)
    # Create schedule config
    cfg.define_split("tile_x", int(M), num_outputs=3, policy="factors")
    cfg.define_split("tile_y", int(N), num_outputs=3, policy="factors")
    cfg.define_split("tile_k", int(K), num_outputs=2, policy="factors")
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if OH > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])

    col_shape = (M, K)
    t1 = time.time()
    data_im2col = tvm.compute(col_shape,
                              lambda s, t: data_pad[idxdiv(s, outsize),
                                                    idxdiv(t, kh * kw),
                                                    idxdiv(idxmod(s, outsize), out_width) * strides[0] + (
                                                        idxdiv(idxmod(t, kh * kw), kw) * dilation_h),
                                                    idxmod(idxmod(s, outsize), out_width) * strides[1] + (
                                                        idxmod(idxmod(t, kh * kw), kw) * dilation_w)],
                              name='data_im2col')

    #print("data -> data_im2col",time.time()-t1)
    #t2 = time.time()
    kernel_shape = (N, K)
    kernel_im2col = tvm.compute(kernel_shape,
                                lambda o, t: kernel[o,
                                                    idxdiv(t, kh * kw),
                                                    idxdiv(idxmod(t, kw * kh), kw),
                                                    idxmod(idxmod(t, kw * kh), kw)],
                                name='kernel_im2col')


    #print("kernel -> kernel_im2col",time.time() - t2)
    #t3 = time.time()
    C_shape = (M, N)
    k = tvm.reduce_axis((0, K), name='k')
    C = tvm.compute(C_shape, lambda i, j: tvm.sum(data_im2col[i, k].astype(out_dtype)
                                                  * kernel_im2col[j, k].astype(out_dtype), axis=k), name='C')
    #print("data_im2col * kernel_im2col -> C",time.time()-t3)

    #t4 = time.time()
    out_shape = (B, OC, out_height, out_width)
    Out = tvm.compute(out_shape, lambda b, oc, oh, ow:
                C[b*out_width*out_height + oh*out_width + ow, oc].astype(out_dtype), name='Out')
    #print("C -> Out",time.time()-t4)
    cfg = autotvm.get_config()
    s = tvm.create_schedule(Out.op)


    x, z = s[data_im2col].op.axis
    xt, xo, xi = cfg["tile_x"].apply(s, data_im2col, x)
    ko, ki = cfg["tile_k"].apply(s, data_im2col, z)
    s[data_im2col].reorder(xt,xo,ko,xi,ki)
    s[data_im2col].parallel(xt)
    s[data_im2col].vectorize(ki)


    x, y = s[C].op.axis
    xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    ko, ki = cfg["tile_k"].apply(s, C, k)
    s[C].reorder(xt, yt, xo, yo, ko, xi, ki, yi)
    xyt = s[C].fuse(xt, yt)
    xyo = s[C].fuse(xo, yo)
    s[C].vectorize(yi)
    # s[C].unroll(zi)
    s[C].parallel(xyt)
    # s[C].parallel(xyo)
    #print(tvm.lower(s,[data,kernel,C,Out],simple_mode=True))
    return s, [data, kernel, Out]
def convTestSchedule2(data_shape, kernel_shape, padding, strides, dilation, dtype):
    B, IC, DH, DW = data_shape
    OC, IC, KH, KW = kernel_shape
    data = tvm.placeholder(data_shape, name='data', dtype=dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=dtype)
    cfg = autotvm.get_config()
    out_dtype = dtype
    pt, pl, pb, pr = padding
    sh, sw = strides
    dilation_h, dilation_w = dilation

    OH = (DH - KH + pt + pb) // sh + 1
    OW = (DW - KW + pl + pr) // sw + 1

    HSTR, WSTR = strides

    pad_h = pt + pb
    pad_w = pl + pr

    pad_height = dh + pad_h
    pad_width = dw + pad_w
    is_kernel_1x1 = KH == 1 and KW == 1
    dilated_kernel_h = (kh - 1) * dilation_h + 1
    dilated_kernel_w = (kw - 1) * dilation_w + 1
    out_height = (dh + pad_h - dilated_kernel_h) // HSTR + 1
    out_width = (dw + pad_w - dilated_kernel_w) // WSTR + 1

    outsize = out_height * out_width
    # pack data
    DOPAD = (pad_h != 0 or pad_w != 0)
    if DOPAD:
        data_pad = topi.nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")
    else:
        data_pad = data

    M = int(B * out_height * out_width)
    K = int(dilated_kernel_h * dilated_kernel_h * IC)
    N = int(OC)

    idxdiv = tvm.indexdiv
    idxmod = tvm.indexmod
    # print(M,K,N)
    # Create schedule config
    cfg.define_split("tile_x", int(M), num_outputs=3, policy="factors")
    cfg.define_split("tile_y", int(N), num_outputs=3, policy="factors")
    cfg.define_split("tile_k", int(K), num_outputs=2, policy="factors")
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if OH > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])

    col_shape = (M, K)
    t1 = time.time()
    data_im2col = tvm.compute(col_shape,
                              lambda s, t: data_pad[idxdiv(s, outsize),
                                                    idxdiv(t, kh * kw),
                                                    idxdiv(idxmod(s, outsize), out_width) * strides[0] + (
                                                            idxdiv(idxmod(t, kh * kw), kw) * dilation_h),
                                                    idxmod(idxmod(s, outsize), out_width) * strides[1] + (
                                                            idxmod(idxmod(t, kh * kw), kw) * dilation_w)],
                              name='data_im2col')

    # print("data -> data_im2col",time.time()-t1)
    # t2 = time.time()
    kernel_shape = (N, K)
    kernel_im2col = tvm.compute(kernel_shape,
                                lambda o, t: kernel[o,
                                                    idxdiv(t, kh * kw),
                                                    idxdiv(idxmod(t, kw * kh), kw),
                                                    idxmod(idxmod(t, kw * kh), kw)],
                                name='kernel_im2col')

    # print("kernel -> kernel_im2col",time.time() - t2)
    # t3 = time.time()
    vec = cfg["tile_k"].size[-1]
    k = tvm.reduce_axis((0, K // vec), "k")
    CC = tvm.compute((M, N, vec),
                     lambda z, y, x: tvm.sum(data_im2col[z, k * vec + x].astype(dtype) * kernel_im2col[y, k * vec + x].astype(dtype),
                                             axis=k))
    kk = tvm.reduce_axis((0, vec), "kk")
    C = tvm.compute((M, N), lambda y, x: tvm.sum(CC[y, x, kk], axis=kk))


    # print("data_im2col * kernel_im2col -> C",time.time()-t3)

    # t4 = time.time()
    out_shape = (B, OC, out_height, out_width)
    Out = tvm.compute(out_shape, lambda b, oc, oh, ow:
    C[b * out_width * out_height + oh * out_width + ow, oc].astype(out_dtype), name='Out')
    # print("C -> Out",time.time()-t4)
    cfg = autotvm.get_config()
    s = tvm.create_schedule(Out.op)


    x, z = s[data_im2col].op.axis
    xt, xo, xi = cfg["tile_x"].apply(s, data_im2col, x)
    ko, ki = cfg["tile_k"].apply(s, data_im2col, z)
    s[data_im2col].reorder(xt, xo, ko, xi, ki)
    s[data_im2col].parallel(xt)
    s[data_im2col].vectorize(ki)


    #same with dnmm332:  C  and CC
    y, x = s[C].op.axis
    kk, = s[C].op.reduce_axis
    yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    xt, xo, xi = cfg["tile_x"].apply(s, C, x)
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
    #print(tvm.lower(s,[data,kernel,C,Out],simple_mode=True))
    return s, [data, kernel, Out]
def convTestSchedule_dnmm332(data_shape, kernel_shape, padding, strides, dilation, dtype):
    B, IC, DH, DW = data_shape
    OC, IC, KH, KW = kernel_shape
    data = tvm.placeholder(data_shape, name='data', dtype=dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=dtype)
    cfg = autotvm.get_config()
    out_dtype = dtype
    pt, pl, pb, pr = padding
    sh, sw = strides
    dilation_h, dilation_w = dilation

    OH = (DH - KH + pt + pb) // sh + 1
    OW = (DW - KW + pl + pr) // sw + 1

    HSTR, WSTR = strides

    pad_h = pt + pb
    pad_w = pl + pr

    pad_height = dh + pad_h
    pad_width = dw + pad_w
    is_kernel_1x1 = KH == 1 and KW == 1
    dilated_kernel_h = (kh - 1) * dilation_h + 1
    dilated_kernel_w = (kw - 1) * dilation_w + 1
    out_height = (dh + pad_h - dilated_kernel_h) // HSTR + 1
    out_width = (dw + pad_w - dilated_kernel_w) // WSTR + 1

    outsize = out_height * out_width
    # pack data
    DOPAD = (pad_h != 0 or pad_w != 0)
    if DOPAD:
        data_pad = topi.nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")
    else:
        data_pad = data

    M = int(B * out_height * out_width)
    K = int(IC)
    N = int(OC)
    # M = int(B * out_height * out_width)
    # K = int(dilated_kernel_h * dilated_kernel_h * IC)
    # N = int(OC)



    idxdiv = tvm.indexdiv
    idxmod = tvm.indexmod
    # print(M,K,N)
    # Create schedule config
    cfg.define_split("tile_x", M, num_outputs=3, policy="factors")
    cfg.define_split("tile_y", N, num_outputs=3, policy="factors")
    cfg.define_split("tile_k", K, num_outputs=2, policy="factors")
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if OH > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])

    col_shape = (M, K)
    t1 = time.time()
    data_im2col = tvm.compute(col_shape,
                              lambda s, t: data_pad[idxdiv(s, outsize),
                                                    idxdiv(t, kh * kw),
                                                    idxdiv(idxmod(s, outsize), out_width) * strides[0] + (
                                                            idxdiv(idxmod(t, kh * kw), kw) * dilation_h),
                                                    idxmod(idxmod(s, outsize), out_width) * strides[1] + (
                                                            idxmod(idxmod(t, kh * kw), kw) * dilation_w)],
                              name='data_im2col')

    # print("data -> data_im2col",time.time()-t1)
    # t2 = time.time()
    kernel_shape = (N, K)
    kernel_im2col = tvm.compute(kernel_shape,
                                lambda o, t: kernel[o,
                                                    idxdiv(t, kh * kw),
                                                    idxdiv(idxmod(t, kw * kh), kw),
                                                    idxmod(idxmod(t, kw * kh), kw)],
                                name='kernel_im2col')

    # print("kernel -> kernel_im2col",time.time() - t2)
    # t3 = time.time()
    C = dnmm332_opreator(data_im2col,col_shape,kernel_im2col,kernel_shape,dtype,cfg)
    # print("data_im2col * kernel_im2col -> C",time.time()-t3)

    # t4 = time.time()
    out_shape = (B, OC, out_height, out_width)
    Out = tvm.compute(out_shape, lambda b, oc, oh, ow:
    C[b * out_width * out_height + oh * out_width + ow, oc].astype(out_dtype), name='Out')
    # print("C -> Out",time.time()-t4)
    s = tvm.create_schedule(Out.op)

    #data_im2col
    x, z = s[data_im2col].op.axis
    xt, xo, xi = cfg["tile_x"].apply(s, data_im2col, x)
    ko, ki = cfg["tile_k"].apply(s, data_im2col, z)
    s[data_im2col].reorder(xt, xo, ko, xi, ki)
    s[data_im2col].parallel(xt)
    s[data_im2col].vectorize(ki)

    #kernel_im2col
    y, z = s[kernel_im2col].op.axis
    yt, yo, yi = cfg["tile_y"].apply(s, kernel_im2col, y)
    zo, zi = cfg["tile_k"].apply(s, kernel_im2col, z)
    s[kernel_im2col].reorder(yt, yo, zo, yi, zi)
    s[kernel_im2col].parallel(yt)
    s[kernel_im2col].vectorize(zi)




    #same with dnmm332:  C  and CC
    s = dnmm332_schedule(C,cfg,s)

    b, oc, oh, ow = s[Out].op.axis
    yt, yo, yi = cfg["tile_y"].apply(s, Out, oc)
    xt, xo, xi = cfg["tile_x"].apply(s, Out, ow)
    #xyt = s[Out].fuse(yt,xt)
    s[Out].parallel(b)


    print(tvm.lower(s,[data,kernel,C,Out],simple_mode=True))
    return s, [data, kernel, Out]
def convTestSchedule_dnmm(data_shape, kernel_shape, padding, strides, dilation, dtype):
    B, IC, DH, DW = data_shape
    OC, IC, KH, KW = kernel_shape
    data = tvm.placeholder(data_shape, name='data', dtype=dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=dtype)
    cfg = autotvm.get_config()
    out_dtype = dtype
    pt, pl, pb, pr = padding
    sh, sw = strides
    dilation_h, dilation_w = dilation

    OH = (DH - KH + pt + pb) // sh + 1
    OW = (DW - KW + pl + pr) // sw + 1

    HSTR, WSTR = strides

    pad_h = pt + pb
    pad_w = pl + pr

    pad_height = dh + pad_h
    pad_width = dw + pad_w
    is_kernel_1x1 = KH == 1 and KW == 1
    dilated_kernel_h = (kh - 1) * dilation_h + 1
    dilated_kernel_w = (kw - 1) * dilation_w + 1
    out_height = (dh + pad_h - dilated_kernel_h) // HSTR + 1
    out_width = (dw + pad_w - dilated_kernel_w) // WSTR + 1

    outsize = out_height * out_width
    # pack data
    DOPAD = (pad_h != 0 or pad_w != 0)
    if DOPAD:
        data_pad = topi.nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")
    else:
        data_pad = data

    M = int(B * out_height * out_width)
    K = int(dilated_kernel_h * dilated_kernel_h * IC)
    N = int(OC)

    idxdiv = tvm.indexdiv
    idxmod = tvm.indexmod
    # print(M,K,N)
    # Create schedule config
    cfg.define_split("tile_x", M, num_outputs=2, policy="factors")
    cfg.define_split("tile_y", N, num_outputs=2, policy="factors")
    cfg.define_split("tile_k", K, num_outputs=2, policy="factors")
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if OH > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])

    col_shape = (M, K)
    t1 = time.time()
    data_im2col = tvm.compute(col_shape,
                              lambda s, t: data_pad[idxdiv(s, outsize),
                                                    idxdiv(t, kh * kw),
                                                    idxdiv(idxmod(s, outsize), out_width) * strides[0] + (
                                                            idxdiv(idxmod(t, kh * kw), kw) * dilation_h),
                                                    idxmod(idxmod(s, outsize), out_width) * strides[1] + (
                                                            idxmod(idxmod(t, kh * kw), kw) * dilation_w)],
                              name='data_im2col')

    # print("data -> data_im2col",time.time()-t1)
    # t2 = time.time()
    kernel_shape = (N, K)
    kernel_im2col = tvm.compute(kernel_shape,
                                lambda o, t: kernel[o,
                                                    idxdiv(t, kh * kw),
                                                    idxdiv(idxmod(t, kw * kh), kw),
                                                    idxmod(idxmod(t, kw * kh), kw)],
                                name='kernel_im2col')

    # print("kernel -> kernel_im2col",time.time() - t2)
    # t3 = time.time()
    C = dnmm332_opreator(data_im2col,col_shape,kernel_im2col,kernel_shape,dtype,cfg)
    # print("data_im2col * kernel_im2col -> C",time.time()-t3)

    # t4 = time.time()
    out_shape = (B, OC, out_height, out_width)
    Out = tvm.compute(out_shape, lambda b, oc, oh, ow:
    C[b * out_width * out_height + oh * out_width + ow, oc].astype(out_dtype), name='Out')
    # print("C -> Out",time.time()-t4)
    s = tvm.create_schedule(Out.op)

    #data_im2col
    x, z = s[data_im2col].op.axis
    xo, xi = cfg["tile_x"].apply(s, data_im2col, x)
    ko, ki = cfg["tile_k"].apply(s, data_im2col, z)
    s[data_im2col].reorder(xo, ko, xi, ki)
    s[data_im2col].parallel(xo)
    s[data_im2col].vectorize(ki)


    #same with dnmm332:  C  and CC
    s = dnmm_schedule(C,cfg,s)
    #print(tvm.lower(s,[data,kernel,C,Out],simple_mode=True))
    return s, [data, kernel, Out]
def convTestSchedule_opt_dnmm(data_shape, kernel_shape, padding, strides, dilation, dtype):
    B, IC, DH, DW = data_shape
    OC, IC, KH, KW = kernel_shape
    data = tvm.placeholder(data_shape, name='data', dtype=dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=dtype)
    cfg = autotvm.get_config()
    out_dtype = dtype
    pt, pl, pb, pr = padding
    sh, sw = strides
    dilation_h, dilation_w = dilation

    OH = (DH - KH + pt + pb) // sh + 1
    OW = (DW - KW + pl + pr) // sw + 1

    HSTR, WSTR = strides

    pad_h = pt + pb
    pad_w = pl + pr

    pad_height = dh + pad_h
    pad_width = dw + pad_w
    is_kernel_1x1 = KH == 1 and KW == 1
    dilated_kernel_h = (kh - 1) * dilation_h + 1
    dilated_kernel_w = (kw - 1) * dilation_w + 1
    out_height = (dh + pad_h - dilated_kernel_h) // HSTR + 1
    out_width = (dw + pad_w - dilated_kernel_w) // WSTR + 1

    outsize = out_height * out_width
    # pack data
    DOPAD = (pad_h != 0 or pad_w != 0)
    if DOPAD:
        data_pad = topi.nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")
    else:
        data_pad = data

    M = int(B * out_height * out_width)
    K = int(kh*kw*IC)
    N = int(OC)
    # M = int(B * out_height * out_width)
    # K = int(dilated_kernel_h * dilated_kernel_h * IC)
    # N = int(OC)

    idxdiv = tvm.indexdiv
    idxmod = tvm.indexmod
    # print(M,K,N)
    # Create schedule config
    cfg.define_split("tile_x", M, num_outputs=2, policy="factors")
    cfg.define_split("tile_y", N, num_outputs=2, policy="factors")
    cfg.define_split("tile_k", K, num_outputs=2, policy="factors")
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if OH > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])
    cfg.define_split("tile_oc", OC, num_outputs=2, policy="factors")
    cfg.define_split("tile_ow", OW, num_outputs=2, policy="factors")


    col_shape = (M, K)
    t1 = time.time()
    data_im2col = tvm.compute(col_shape,
                              lambda s, t: data_pad[idxdiv(s, outsize),
                                                    idxdiv(t, kh * kw),
                                                    idxdiv(idxmod(s, outsize), out_width) * strides[0] + (
                                                            idxdiv(idxmod(t, kh * kw), kw) * dilation_h),
                                                    idxmod(idxmod(s, outsize), out_width) * strides[1] + (
                                                            idxmod(idxmod(t, kh * kw), kw) * dilation_w)],
                              name='data_im2col')

    # print("data -> data_im2col",time.time()-t1)
    # t2 = time.time()
    kernel_shape = (N, K)
    kernel_im2col = tvm.compute(kernel_shape,
                                lambda o, t: kernel[o,
                                                    idxdiv(t, kh * kw),
                                                    idxdiv(idxmod(t, kw * kh), kw),
                                                    idxmod(idxmod(t, kw * kh), kw)],
                                name='kernel_im2col')

    # print("kernel -> kernel_im2col",time.time() - t2)
    # t3 = time.time()
    C = dnmm_opreator(data_im2col,col_shape,kernel_im2col,kernel_shape,dtype,cfg)
    # print("data_im2col * kernel_im2col -> C",time.time()-t3)

    # t4 = time.time()
    out_shape = (B, OC, out_height, out_width)
    Out = tvm.compute(out_shape, lambda b, oc, oh, ow:
    C[b * out_width * out_height + oh * out_width + ow, oc].astype(out_dtype), name='Out')
    # print("C -> Out",time.time()-t4)
    s = tvm.create_schedule(Out.op)

    #data_im2col
    x, z = s[data_im2col].op.axis
    xo, xi = cfg["tile_x"].apply(s, data_im2col, x)
    ko, ki = cfg["tile_k"].apply(s, data_im2col, z)
    s[data_im2col].reorder(xo, ko, xi, ki)
    xko = s[data_im2col].fuse(xo,ko)
    s[data_im2col].parallel(xko)
    s[data_im2col].vectorize(ki)

    #kernel_im2col
    y, z = s[kernel_im2col].op.axis
    yo, yi = cfg["tile_y"].apply(s, kernel_im2col, y)
    zo, zi = cfg["tile_k"].apply(s, kernel_im2col, z)
    s[kernel_im2col].reorder(yo, zo, yi, zi)
    yko = s[kernel_im2col].fuse(yo, zo)
    s[kernel_im2col].parallel(yko)
    s[kernel_im2col].vectorize(zi)
    #same with dnmm332:  C  and CC
    s = dnmm_schedule(C,cfg,s)
    b, oc, oh, ow = s[Out].op.axis
    oco, oci = cfg["tile_oc"].apply(s, Out, oc)
    owo, owi = cfg["tile_ow"].apply(s, Out, ow)
    s[Out].reorder(b, oco, oh, owo, oci, owi)
    b_oco_oh = s[Out].fuse(b,oco,oh)
    s[Out].parallel(b_oco_oh)
    s[Out].vectorize(owi)
    #print(tvm.lower(s,[data,kernel,C,Out],simple_mode=True))
    return s, [data, kernel, Out]
def convTestSchedule_opt_dnmm_0308(data_shape, kernel_shape, padding, strides, dilation, dtype):
    B, IC, DH, DW = data_shape
    OC, IC, KH, KW = kernel_shape
    data = tvm.placeholder(data_shape, name='data', dtype=dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=dtype)
    cfg = autotvm.get_config()
    out_dtype = dtype
    pt, pl, pb, pr = padding
    sh, sw = strides
    dilation_h, dilation_w = dilation

    OH = (DH - KH + pt + pb) // sh + 1
    OW = (DW - KW + pl + pr) // sw + 1

    HSTR, WSTR = strides

    pad_h = pt + pb
    pad_w = pl + pr

    pad_height = dh + pad_h
    pad_width = dw + pad_w
    is_kernel_1x1 = KH == 1 and KW == 1
    dilated_kernel_h = (kh - 1) * dilation_h + 1
    dilated_kernel_w = (kw - 1) * dilation_w + 1
    out_height = (dh + pad_h - dilated_kernel_h) // HSTR + 1
    out_width = (dw + pad_w - dilated_kernel_w) // WSTR + 1

    outsize = out_height * out_width
    # pack data
    DOPAD = (pad_h != 0 or pad_w != 0)
    if DOPAD:
        data_pad = topi.nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")
    else:
        data_pad = data

    M = int(B * out_height * out_width)
    K = int(kh*kw*IC)
    N = int(OC)
    # M = int(B * out_height * out_width)
    # K = int(dilated_kernel_h * dilated_kernel_h * IC)
    # N = int(OC)

    idxdiv = tvm.indexdiv
    idxmod = tvm.indexmod
    # print(M,K,N)
    # Create schedule config
    cfg.define_split("tile_x", M, num_outputs=2, policy="factors")
    cfg.define_split("tile_y", N, num_outputs=2, policy="factors")
    cfg.define_split("tile_k", K, num_outputs=2, policy="factors")
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if OH > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])


    col_shape = (M, K)
    t1 = time.time()
    data_im2col = tvm.compute(col_shape,
                              lambda s, t: data_pad[idxdiv(s, outsize),
                                                    idxdiv(t, kh * kw),
                                                    idxdiv(idxmod(s, outsize), out_width) * strides[0] + (
                                                            idxdiv(idxmod(t, kh * kw), kw) * dilation_h),
                                                    idxmod(idxmod(s, outsize), out_width) * strides[1] + (
                                                            idxmod(idxmod(t, kh * kw), kw) * dilation_w)],
                              name='data_im2col')

    # print("data -> data_im2col",time.time()-t1)
    # t2 = time.time()
    kernel_shape = (N, K)
    kernel_im2col = tvm.compute(kernel_shape,
                                lambda o, t: kernel[o,
                                                    idxdiv(t, kh * kw),
                                                    idxdiv(idxmod(t, kw * kh), kw),
                                                    idxmod(idxmod(t, kw * kh), kw)],
                                name='kernel_im2col')

    # print("kernel -> kernel_im2col",time.time() - t2)
    # t3 = time.time()
    C = dnmm_opreator(data_im2col,col_shape,kernel_im2col,kernel_shape,dtype,cfg)
    # print("data_im2col * kernel_im2col -> C",time.time()-t3)

    # t4 = time.time()
    out_shape = (B, OC, out_height, out_width)
    Out = tvm.compute(out_shape, lambda b, oc, oh, ow:
    C[b * out_width * out_height + oh * out_width + ow, oc].astype(out_dtype), name='Out')
    # print("C -> Out",time.time()-t4)
    s = tvm.create_schedule(Out.op)

    #data_im2col
    x, z = s[data_im2col].op.axis
    xo, xi = cfg["tile_x"].apply(s, data_im2col, x)
    ko, ki = cfg["tile_k"].apply(s, data_im2col, z)
    s[data_im2col].reorder(xo, ko, xi, ki)
    xko = s[data_im2col].fuse(xo,ko)
    s[data_im2col].parallel(xko)
    s[data_im2col].vectorize(ki)

    #kernel_im2col
    y, z = s[kernel_im2col].op.axis
    yo, yi = cfg["tile_y"].apply(s, kernel_im2col, y)
    zo, zi = cfg["tile_k"].apply(s, kernel_im2col, z)
    s[kernel_im2col].reorder(yo, zo, yi, zi)
    yko = s[kernel_im2col].fuse(yo, zo)
    s[kernel_im2col].parallel(yko)
    s[kernel_im2col].vectorize(zi)
    #same with dnmm332:  C  and CC
    s = dnmm_schedule(C,cfg,s)
    b, oc, oh, ow = s[Out].op.axis
    oco, oci = cfg["tile_y"].apply(s, Out, oc)
    owo, owi = cfg["tile_x"].apply(s, Out, ow)
    s[Out].reorder(b, oco, oh, owo, oci, owi)
    b_oco_oh = s[Out].fuse(b,oco,oh)
    s[Out].parallel(b_oco_oh)
    s[Out].vectorize(owi)
    #print(tvm.lower(s,[data,kernel,C,Out],simple_mode=True))
    return s, [data, kernel, Out]


def convTestSchedule_opt_dnmm222_0225(data_shape, kernel_shape, padding, strides, dilation, dtype):
    B, IC, DH, DW = data_shape
    OC, IC, KH, KW = kernel_shape
    data = tvm.placeholder(data_shape, name='data', dtype=dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=dtype)
    cfg = autotvm.get_config()
    out_dtype = dtype
    pt, pl, pb, pr = padding
    sh, sw = strides
    dilation_h, dilation_w = dilation

    OH = (DH - KH + pt + pb) // sh + 1
    OW = (DW - KW + pl + pr) // sw + 1

    HSTR, WSTR = strides

    pad_h = pt + pb
    pad_w = pl + pr

    pad_height = dh + pad_h
    pad_width = dw + pad_w
    is_kernel_1x1 = KH == 1 and KW == 1
    dilated_kernel_h = (kh - 1) * dilation_h + 1
    dilated_kernel_w = (kw - 1) * dilation_w + 1
    out_height = (dh + pad_h - dilated_kernel_h) // HSTR + 1
    out_width = (dw + pad_w - dilated_kernel_w) // WSTR + 1

    outsize = out_height * out_width
    k_size = kh * kw
    # pack data
    DOPAD = (pad_h != 0 or pad_w != 0)
    if DOPAD:
        data_pad = topi.nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")
    else:
        data_pad = data

    M = int(B * out_height * out_width)
    K = int(kh*kw*IC)
    N = int(OC)
    # M = int(B * out_height * out_width)
    # K = int(dilated_kernel_h * dilated_kernel_h * IC)
    # N = int(OC)

    idxdiv = tvm.indexdiv
    idxmod = tvm.indexmod
    # print(M,K,N)
    # Create schedule config

    cfg.define_split("tile_ic", K, num_outputs=2)
    cfg.define_split("tile_oc", OC, num_outputs=2)
    cfg.define_split("tile_ow", OW, num_outputs=2, filter=lambda y: y.size[-1] <= 64)
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if OH > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])
        unroll_kw = cfg["unroll_kw"].val

    ic_bn, oc_bn, ow_bn = (cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1],cfg["tile_ow"].size[-1])


    col_shape = (M, K)
    t1 = time.time()
    data_im2col = tvm.compute(col_shape,
                              lambda s, t: data_pad[idxdiv(s, outsize),
                                                    idxdiv(t, k_size),
                                                    idxdiv(idxmod(s, outsize), out_width) * strides[0] + (
                                                            idxdiv(idxmod(t, k_size), kw) * dilation_h),
                                                    idxmod(idxmod(s, outsize), out_width) * strides[1] + (
                                                            idxmod(idxmod(t, k_size), kw) * dilation_w)],
                              name='data_im2col')
    print("#################################################################")
    print("ic_bn, oc_bn, ow_bn",ic_bn, oc_bn, ow_bn)
    print("data -> data_im2col",time.time()-t1)
    t2 = time.time()
    kernel_shape = (N, K)
    kernel_im2col = tvm.compute(kernel_shape,
                                lambda o, t: kernel[o,
                                                    idxdiv(t, k_size),
                                                    idxdiv(idxmod(t, k_size), kw),
                                                    idxmod(idxmod(t, k_size), kw)],
                                name='kernel_im2col')

    print("kernel -> kernel_im2col",time.time() - t2)
    t3 = time.time()
    # create tuning space
    vec = ic_bn
    k = tvm.reduce_axis((0, K // vec), "k")
    CC = tvm.compute((M, N, vec),
                     lambda z, y, x: tvm.sum(data_im2col[z, k * vec + x].astype(dtype) * kernel_im2col[y, k * vec + x].astype(dtype),
                                             axis=k))
    kk = tvm.reduce_axis((0, vec), "kk")
    C = tvm.compute((M, N), lambda y, x: tvm.sum(CC[y, x, kk], axis=kk))
    print("data_im2col * kernel_im2col -> C",time.time()-t3)

    t4 = time.time()
    out_shape = (B, OC, out_height, out_width)
    Out = tvm.compute(out_shape, lambda b, oc, oh, ow:
    C[b * out_width * out_height + oh * out_width + ow, oc].astype(out_dtype), name='Out')
    print("C -> Out",time.time()-t4)
    s = tvm.create_schedule(Out.op)

    #data_pad

    # b,ic,h,w = s[data_pad].op.axis
    # b_ic = s[data_pad].fuse(b,ic)
    # s[data_pad].parallel(b_ic)


    #data_im2col
    x, z = s[data_im2col].op.axis
    xo, xi = s[data_im2col].split(x, factor=ow_bn)
    zo, zi = s[data_im2col].split(z, factor=ic_bn)
    s[data_im2col].reorder(xo, zo, xi, zi)
    xzo = s[data_im2col].fuse(xo,zo)
    s[data_im2col].parallel(xzo)
    s[data_im2col].vectorize(zi)


    #kernel_im2col
    y, z = s[kernel_im2col].op.axis
    yo, yi = s[kernel_im2col].split(y, factor=oc_bn)
    zo, zi = s[kernel_im2col].split(z, factor=ic_bn)
    s[kernel_im2col].reorder(yo, zo, yi, zi)
    yzo = s[kernel_im2col].fuse(yo, zo)
    s[kernel_im2col].parallel(yzo)
    s[kernel_im2col].vectorize(zi)

    #same with dnmm:  C  and CC
    x, y = s[C].op.axis
    kk, = s[C].op.reduce_axis

    xo, xi = s[C].split(x, factor=ow_bn)
    yo, yi = s[C].split(y, factor=oc_bn)
    # xo, xi = cfg["tile_x"].apply(s, C, x)
    # yo, yi = cfg["tile_y"].apply(s, C, y)
    # yo, xo, yi, xi = s[C].tile(C.op.axis[0], C.op.axis[1], 4, 4)
    s[C].reorder(xo,yo,xi,yi)
    xyo = s[C].fuse(xo, yo)
    s[C].parallel(xyo)
    s[C].unroll(kk)

    CC, = s[C].op.input_tensors
    s[CC].compute_at(s[C], xyo)
    z, y, x = s[CC].op.axis
    k, = s[CC].op.reduce_axis
    yz = s[CC].fuse(z, y)
    s[CC].reorder(k, yz, x)
    s[CC].unroll(yz)
    s[CC].vectorize(x)


    b, oc, oh, ow = s[Out].op.axis
    oco, oci = s[Out].split(oc, factor=oc_bn)
    owo, owi = s[Out].split(ow, factor=ow_bn)
    s[Out].reorder(b, oco, oh, owo, oci, owi)
    b_oco_oh = s[Out].fuse(b,oco,oh)
    s[Out].parallel(b_oco_oh)
    s[Out].vectorize(owi)
    #print(tvm.lower(s,[data,kernel,C,Out],simple_mode=True))
    return s, [data, kernel, Out]

def convTestSchedule_opt_lpmm222_0304(data_shape, kernel_shape, padding, strides, dilation, dtype):
    B, IC, DH, DW = data_shape
    OC, IC, KH, KW = kernel_shape
    data = tvm.placeholder(data_shape, name='data', dtype=dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=dtype)
    cfg = autotvm.get_config()
    out_dtype = dtype
    pt, pl, pb, pr = padding
    sh, sw = strides
    dilation_h, dilation_w = dilation

    OH = (DH - KH + pt + pb) // sh + 1
    OW = (DW - KW + pl + pr) // sw + 1

    HSTR, WSTR = strides

    pad_h = pt + pb
    pad_w = pl + pr

    pad_height = dh + pad_h
    pad_width = dw + pad_w
    is_kernel_1x1 = KH == 1 and KW == 1
    dilated_kernel_h = (kh - 1) * dilation_h + 1
    dilated_kernel_w = (kw - 1) * dilation_w + 1
    out_height = (dh + pad_h - dilated_kernel_h) // HSTR + 1
    out_width = (dw + pad_w - dilated_kernel_w) // WSTR + 1

    outsize = out_height * out_width
    k_size = kh * kw
    # pack data
    DOPAD = (pad_h != 0 or pad_w != 0)
    if DOPAD:
        data_pad = topi.nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")
    else:
        data_pad = data

    M = int(B * out_height * out_width)
    K = int(kh*kw*IC)
    N = int(OC)
    # M = int(B * out_height * out_width)
    # K = int(dilated_kernel_h * dilated_kernel_h * IC)
    # N = int(OC)

    idxdiv = tvm.indexdiv
    idxmod = tvm.indexmod
    # print(M,K,N)
    # Create schedule config

    cfg.define_split("tile_ic", K, num_outputs=2)
    cfg.define_split("tile_oc", OC, num_outputs=2)
    cfg.define_split("tile_ow", OW, num_outputs=2, filter=lambda y: y.size[-1] <= 64)
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if OH > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])
        unroll_kw = cfg["unroll_kw"].val

    ic_bn, oc_bn, ow_bn = (cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1],cfg["tile_ow"].size[-1])


    col_shape = (M, K)
    t1 = time.time()
    data_im2col = tvm.compute(col_shape,
                              lambda s, t: data_pad[idxdiv(s, outsize),
                                                    idxdiv(t, k_size),
                                                    idxdiv(idxmod(s, outsize), out_width) * strides[0] + (
                                                            idxdiv(idxmod(t, k_size), kw) * dilation_h),
                                                    idxmod(idxmod(s, outsize), out_width) * strides[1] + (
                                                            idxmod(idxmod(t, k_size), kw) * dilation_w)],
                              name='data_im2col')
    print("#################################################################")
    print("ic_bn, oc_bn, ow_bn",ic_bn, oc_bn, ow_bn)
    print("data -> data_im2col",time.time()-t1)
    t2 = time.time()
    kernel_shape = (N, K)
    kernel_im2col = tvm.compute(kernel_shape,
                                lambda o, t: kernel[o,
                                                    idxdiv(t, k_size),
                                                    idxdiv(idxmod(t, k_size), kw),
                                                    idxmod(idxmod(t, k_size), kw)],
                                name='kernel_im2col')

    print("kernel -> kernel_im2col",time.time() - t2)
    t3 = time.time()
    ########################
    packD_bn = ow_bn
    packD_shape = (M // packD_bn, K, packD_bn)
    packD = tvm.compute(packD_shape,
                        lambda z, y, x: data_im2col[z * packD_bn + x, y], name="packed_data")

    k = tvm.reduce_axis((0, K), name="k")
    C = tvm.compute((M, N),
                    lambda y, x: tvm.sum(
                        packD[tvm.indexdiv(y, packD_bn), k, tvm.indexmod(y, packD_bn)].astype(dtype) * kernel_im2col[x,k],
                        axis=k))
    #########################
    # # create tuning space
    # vec = ic_bn
    # k = tvm.reduce_axis((0, K // vec), "k")
    # CC = tvm.compute((M, N, vec),
    #                  lambda z, y, x: tvm.sum(data_im2col[z, k * vec + x].astype(dtype) * kernel_im2col[y, k * vec + x].astype(dtype),
    #                                          axis=k))
    # kk = tvm.reduce_axis((0, vec), "kk")
    # C = tvm.compute((M, N), lambda y, x: tvm.sum(CC[y, x, kk], axis=kk))
    #########################
    print("data_im2col * kernel_im2col -> C",time.time()-t3)

    t4 = time.time()
    out_shape = (B, OC, out_height, out_width)
    Out = tvm.compute(out_shape, lambda b, oc, oh, ow:
    C[b * out_width * out_height + oh * out_width + ow, oc].astype(out_dtype), name='Out')
    print("C -> Out",time.time()-t4)
    s = tvm.create_schedule(Out.op)

    #data_pad
    if DOPAD:
        s[data_pad].compute_inline()
    # b,ic,h,w = s[data_pad].op.axis
    # b_ic = s[data_pad].fuse(b,ic)
    # s[data_pad].parallel(b_ic)


    #data_im2col
    x, z = s[data_im2col].op.axis
    xo, xi = s[data_im2col].split(x, factor=ow_bn)
    zo, zi = s[data_im2col].split(z, factor=ic_bn)
    s[data_im2col].reorder(xo, zo, xi, zi)
    xzo = s[data_im2col].fuse(xo,zo)
    s[data_im2col].parallel(xzo)
    s[data_im2col].vectorize(zi)


    #kernel_im2col
    y, z = s[kernel_im2col].op.axis
    yo, yi = s[kernel_im2col].split(y, factor=oc_bn)
    zo, zi = s[kernel_im2col].split(z, factor=ic_bn)
    s[kernel_im2col].reorder(yo, zo, yi, zi)
    yzo = s[kernel_im2col].fuse(yo, zo)
    s[kernel_im2col].parallel(yzo)
    s[kernel_im2col].vectorize(zi)
    #####################################
    #same with lpmmv:  C  and CC
    CC = s.cache_write(C, "global")
    y, x = s[C].op.axis
    k, = s[CC].op.reduce_axis
    yo, yi = s[C].split(y, factor=ow_bn)
    xo, xi = s[C].split(x, factor=oc_bn)
    # yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    # xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    s[C].reorder(yo, xo, yi, xi)
    xyo = s[C].fuse(yo, xo)
    s[C].parallel(xyo)
    s[C].unroll(yi)
    s[C].vectorize(xi)

    s[CC].compute_at(s[C], xyo)
    y, x = s[CC].op.axis
    ko, ki = s[CC].split(k, factor=ic_bn)
    s[CC].reorder(ko, ki, y, x)
    s[CC].vectorize(x)
    s[CC].unroll(y)
    s[CC].unroll(ki)

    z, y, x = s[packD].op.axis
    s[packD].reorder(z, x, y)
    s[packD].parallel(z)
    # 20201104 fixed add vec for y
    s[packD].vectorize(y)
    #####################################

    b, oc, oh, ow = s[Out].op.axis
    oco, oci = s[Out].split(oc, factor=oc_bn)
    owo, owi = s[Out].split(ow, factor=ow_bn)
    s[Out].reorder(b, oco, oh, owo, oci, owi)
    b_oco_oh = s[Out].fuse(b,oco,oh)
    s[Out].parallel(b_oco_oh)
    s[Out].vectorize(owi)
    #print(tvm.lower(s,[data,kernel,C,Out],simple_mode=True))
    return s, [data, kernel, Out]

def convTestSchedule_opt_rpmm222_0304(data_shape, kernel_shape, padding, strides, dilation, dtype):
    B, IC, DH, DW = data_shape
    OC, IC, KH, KW = kernel_shape
    data = tvm.placeholder(data_shape, name='data', dtype=dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=dtype)
    cfg = autotvm.get_config()
    out_dtype = dtype
    pt, pl, pb, pr = padding
    sh, sw = strides
    dilation_h, dilation_w = dilation

    OH = (DH - KH + pt + pb) // sh + 1
    OW = (DW - KW + pl + pr) // sw + 1

    HSTR, WSTR = strides

    pad_h = pt + pb
    pad_w = pl + pr

    pad_height = dh + pad_h
    pad_width = dw + pad_w
    is_kernel_1x1 = KH == 1 and KW == 1
    dilated_kernel_h = (kh - 1) * dilation_h + 1
    dilated_kernel_w = (kw - 1) * dilation_w + 1
    out_height = (dh + pad_h - dilated_kernel_h) // HSTR + 1
    out_width = (dw + pad_w - dilated_kernel_w) // WSTR + 1

    outsize = out_height * out_width
    k_size = kh * kw
    # pack data
    DOPAD = (pad_h != 0 or pad_w != 0)
    if DOPAD:
        data_pad = topi.nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")
    else:
        data_pad = data

    M = int(B * out_height * out_width)
    K = int(kh*kw*IC)
    N = int(OC)
    # M = int(B * out_height * out_width)
    # K = int(dilated_kernel_h * dilated_kernel_h * IC)
    # N = int(OC)

    idxdiv = tvm.indexdiv
    idxmod = tvm.indexmod
    # print(M,K,N)
    # Create schedule config

    cfg.define_split("tile_ic", K, num_outputs=2)
    cfg.define_split("tile_oc", OC, num_outputs=2)
    cfg.define_split("tile_ow", OW, num_outputs=2, filter=lambda y: y.size[-1] <= 64)
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if OH > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])
        unroll_kw = cfg["unroll_kw"].val

    ic_bn, oc_bn, ow_bn = (cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1],cfg["tile_ow"].size[-1])


    col_shape = (M, K)
    t1 = time.time()
    data_im2col = tvm.compute(col_shape,
                              lambda s, t: data_pad[idxdiv(s, outsize),
                                                    idxdiv(t, k_size),
                                                    idxdiv(idxmod(s, outsize), out_width) * strides[0] + (
                                                            idxdiv(idxmod(t, k_size), kw) * dilation_h),
                                                    idxmod(idxmod(s, outsize), out_width) * strides[1] + (
                                                            idxmod(idxmod(t, k_size), kw) * dilation_w)],
                              name='data_im2col')
    print("#################################################################")
    print("ic_bn, oc_bn, ow_bn",ic_bn, oc_bn, ow_bn)
    print("data -> data_im2col",time.time()-t1)
    t2 = time.time()
    kernel_shape = (N, K)
    kernel_im2col = tvm.compute(kernel_shape,
                                lambda o, t: kernel[o,
                                                    idxdiv(t, k_size),
                                                    idxdiv(idxmod(t, k_size), kw),
                                                    idxmod(idxmod(t, k_size), kw)],
                                name='kernel_im2col')

    print("kernel -> kernel_im2col",time.time() - t2)
    t3 = time.time()
    ########################

    packw_bn = oc_bn
    packw_shape = (N // packw_bn, K, packw_bn)
    packw = tvm.compute(packw_shape,
                        lambda z, y, x: kernel_im2col[z * packw_bn + x, y], name="packed_weight")

    k = tvm.reduce_axis((0, K), name="k")
    C = tvm.compute((M, N),
                    lambda y, x: tvm.sum(
                        data_im2col[y, k].astype(dtype) *
                        packw[tvm.indexdiv(x, packw_bn), k, tvm.indexmod(x, packw_bn)].astype(dtype),
                        axis=k))
    #########################
    # # create tuning space
    # vec = ic_bn
    # k = tvm.reduce_axis((0, K // vec), "k")
    # CC = tvm.compute((M, N, vec),
    #                  lambda z, y, x: tvm.sum(data_im2col[z, k * vec + x].astype(dtype) * kernel_im2col[y, k * vec + x].astype(dtype),
    #                                          axis=k))
    # kk = tvm.reduce_axis((0, vec), "kk")
    # C = tvm.compute((M, N), lambda y, x: tvm.sum(CC[y, x, kk], axis=kk))
    #########################
    print("data_im2col * kernel_im2col -> C",time.time()-t3)

    t4 = time.time()
    out_shape = (B, OC, out_height, out_width)
    Out = tvm.compute(out_shape, lambda b, oc, oh, ow:
    C[b * out_width * out_height + oh * out_width + ow, oc].astype(out_dtype), name='Out')
    print("C -> Out",time.time()-t4)
    s = tvm.create_schedule(Out.op)

    #data_pad
    if DOPAD:
        s[data_pad].compute_inline()
    # b,ic,h,w = s[data_pad].op.axis
    # b_ic = s[data_pad].fuse(b,ic)
    # s[data_pad].parallel(b_ic)


    #data_im2col
    x, z = s[data_im2col].op.axis
    xo, xi = s[data_im2col].split(x, factor=ow_bn)
    zo, zi = s[data_im2col].split(z, factor=ic_bn)
    s[data_im2col].reorder(xo, zo, xi, zi)
    xzo = s[data_im2col].fuse(xo,zo)
    s[data_im2col].parallel(xzo)
    s[data_im2col].vectorize(zi)


    #kernel_im2col
    y, z = s[kernel_im2col].op.axis
    yo, yi = s[kernel_im2col].split(y, factor=oc_bn)
    zo, zi = s[kernel_im2col].split(z, factor=ic_bn)
    s[kernel_im2col].reorder(yo, zo, yi, zi)
    yzo = s[kernel_im2col].fuse(yo, zo)
    s[kernel_im2col].parallel(yzo)
    s[kernel_im2col].vectorize(zi)
    #####################################
    #same with lpmmv:  C  and CC
    CC = s.cache_write(C, "global")
    y, x = s[C].op.axis
    k, = s[CC].op.reduce_axis
    #yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    #xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    yo, yi = s[C].split(y, factor=ow_bn)
    xo, xi = s[C].split(x, factor=oc_bn)
    s[C].reorder(yo, xo, yi, xi)
    xyo = s[C].fuse(yo, xo)
    s[C].parallel(xyo)
    s[C].unroll(yi)
    s[C].vectorize(xi)

    s[CC].compute_at(s[C], xyo)
    y, x = s[CC].op.axis
    ko, ki = s[CC].split(k, factor=ic_bn)
    s[CC].reorder(ko, ki, y, x)
    s[CC].vectorize(x)
    s[CC].unroll(y)
    s[CC].unroll(ki)

    z, y, x = s[packw].op.axis
    s[packw].reorder(z, x, y)
    s[packw].parallel(z)
    s[packw].vectorize(x)
    #####################################

    b, oc, oh, ow = s[Out].op.axis
    oco, oci = s[Out].split(oc, factor=oc_bn)
    owo, owi = s[Out].split(ow, factor=ow_bn)
    s[Out].reorder(b, oco, oh, owo, oci, owi)
    b_oco_oh = s[Out].fuse(b,oco,oh)
    s[Out].parallel(b_oco_oh)
    s[Out].vectorize(owi)
    #print(tvm.lower(s,[data,kernel,C,Out],simple_mode=True))
    return s, [data, kernel, Out]

def convTestSchedule_opt_tmm222_0304(data_shape, kernel_shape, padding, strides, dilation, dtype):
    B, IC, DH, DW = data_shape
    OC, IC, KH, KW = kernel_shape
    data = tvm.placeholder(data_shape, name='data', dtype=dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=dtype)
    cfg = autotvm.get_config()
    out_dtype = dtype
    pt, pl, pb, pr = padding
    sh, sw = strides
    dilation_h, dilation_w = dilation

    OH = (DH - KH + pt + pb) // sh + 1
    OW = (DW - KW + pl + pr) // sw + 1

    HSTR, WSTR = strides

    pad_h = pt + pb
    pad_w = pl + pr

    pad_height = dh + pad_h
    pad_width = dw + pad_w
    is_kernel_1x1 = KH == 1 and KW == 1
    dilated_kernel_h = (kh - 1) * dilation_h + 1
    dilated_kernel_w = (kw - 1) * dilation_w + 1
    out_height = (dh + pad_h - dilated_kernel_h) // HSTR + 1
    out_width = (dw + pad_w - dilated_kernel_w) // WSTR + 1

    outsize = out_height * out_width
    k_size = kh * kw
    # pack data
    DOPAD = (pad_h != 0 or pad_w != 0)
    if DOPAD:
        data_pad = topi.nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")
    else:
        data_pad = data

    M = int(B * out_height * out_width)
    K = int(kh*kw*IC)
    N = int(OC)
    # M = int(B * out_height * out_width)
    # K = int(dilated_kernel_h * dilated_kernel_h * IC)
    # N = int(OC)

    idxdiv = tvm.indexdiv
    idxmod = tvm.indexmod
    # print(M,K,N)
    # Create schedule config

    cfg.define_split("tile_ic", K, num_outputs=2)
    cfg.define_split("tile_oc", OC, num_outputs=2)
    cfg.define_split("tile_ow", OW, num_outputs=2, filter=lambda y: y.size[-1] <= 64)
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if OH > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])
        unroll_kw = cfg["unroll_kw"].val

    ic_bn, oc_bn, ow_bn = (cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1],cfg["tile_ow"].size[-1])


    col_shape = (M, K)
    t1 = time.time()
    data_im2col = tvm.compute(col_shape,
                              lambda s, t: data_pad[idxdiv(s, outsize),
                                                    idxdiv(t, k_size),
                                                    idxdiv(idxmod(s, outsize), out_width) * strides[0] + (
                                                            idxdiv(idxmod(t, k_size), kw) * dilation_h),
                                                    idxmod(idxmod(s, outsize), out_width) * strides[1] + (
                                                            idxmod(idxmod(t, k_size), kw) * dilation_w)],
                              name='data_im2col')
    print("#################################################################")
    print("ic_bn, oc_bn, ow_bn",ic_bn, oc_bn, ow_bn)
    print("data -> data_im2col",time.time()-t1)
    t2 = time.time()
    kernel_shape = (N, K)
    kernel_im2col = tvm.compute(kernel_shape,
                                lambda o, t: kernel[o,
                                                    idxdiv(t, k_size),
                                                    idxdiv(idxmod(t, k_size), kw),
                                                    idxmod(idxmod(t, k_size), kw)],
                                name='kernel_im2col')

    print("kernel -> kernel_im2col",time.time() - t2)
    t3 = time.time()
    ########################
    z = tvm.reduce_axis((0, K), name='z')
    C = tvm.compute((M, N), lambda i, j: tvm.sum(data_im2col[i, z] * kernel_im2col[j, z], axis=z), name='C')
    #########################
    # # create tuning space
    # vec = ic_bn
    # k = tvm.reduce_axis((0, K // vec), "k")
    # CC = tvm.compute((M, N, vec),
    #                  lambda z, y, x: tvm.sum(data_im2col[z, k * vec + x].astype(dtype) * kernel_im2col[y, k * vec + x].astype(dtype),
    #                                          axis=k))
    # kk = tvm.reduce_axis((0, vec), "kk")
    # C = tvm.compute((M, N), lambda y, x: tvm.sum(CC[y, x, kk], axis=kk))
    #########################
    print("data_im2col * kernel_im2col -> C",time.time()-t3)

    t4 = time.time()
    out_shape = (B, OC, out_height, out_width)
    Out = tvm.compute(out_shape, lambda b, oc, oh, ow:
    C[b * out_width * out_height + oh * out_width + ow, oc].astype(out_dtype), name='Out')
    print("C -> Out",time.time()-t4)
    s = tvm.create_schedule(Out.op)

    #data_pad
    if DOPAD:
        s[data_pad].compute_inline()
    # b,ic,h,w = s[data_pad].op.axis
    # b_ic = s[data_pad].fuse(b,ic)
    # s[data_pad].parallel(b_ic)


    #data_im2col
    x, z = s[data_im2col].op.axis
    xo, xi = s[data_im2col].split(x, factor=ow_bn)
    zo, zi = s[data_im2col].split(z, factor=ic_bn)
    s[data_im2col].reorder(xo, zo, xi, zi)
    xzo = s[data_im2col].fuse(xo,zo)
    s[data_im2col].parallel(xzo)
    s[data_im2col].vectorize(zi)


    #kernel_im2col
    y, z = s[kernel_im2col].op.axis
    yo, yi = s[kernel_im2col].split(y, factor=oc_bn)
    zo, zi = s[kernel_im2col].split(z, factor=ic_bn)
    s[kernel_im2col].reorder(yo, zo, yi, zi)
    yzo = s[kernel_im2col].fuse(yo, zo)
    s[kernel_im2col].parallel(yzo)
    s[kernel_im2col].vectorize(zi)
    #####################################
    #same with lpmmv:  C  and CC
    CC = s.cache_write(C, "global")
    y, x = s[C].op.axis
    k, = s[CC].op.reduce_axis
    #yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    #xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    yo, yi = s[C].split(y, factor=ow_bn)
    xo, xi = s[C].split(x, factor=oc_bn)

    s[C].reorder(xo, yo, xi, yi)
    xyo = s[C].fuse(xo, yo)
    s[C].vectorize(yi)
    # s[C].unroll(zi)
    s[C].parallel(xyo)

    s[CC].compute_at(s[C], xyo)
    ko, ki = s[CC].split(k, factor=ic_bn)
    y, x = s[CC].op.axis
    s[CC].reorder(y, x, ko, ki)
    #s[CC].vectorize(ki)
    #s[CC].unroll(x)
    s[CC].unroll(ko)

    #####################################

    b, oc, oh, ow = s[Out].op.axis
    oco, oci = s[Out].split(oc, factor=oc_bn)
    owo, owi = s[Out].split(ow, factor=ow_bn)
    s[Out].reorder(b, oco, oh, owo, oci, owi)
    b_oco_oh = s[Out].fuse(b,oco,oh)
    s[Out].parallel(b_oco_oh)
    s[Out].vectorize(owi)
    #print(tvm.lower(s,[data,kernel,C,Out],simple_mode=True))
    return s, [data, kernel, Out]


def convTestSchedule_opt_rpmm332_0304(data_shape, kernel_shape, padding, strides, dilation, dtype):
    B, IC, DH, DW = data_shape
    OC, IC, KH, KW = kernel_shape
    data = tvm.placeholder(data_shape, name='data', dtype=dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=dtype)
    cfg = autotvm.get_config()
    out_dtype = dtype
    pt, pl, pb, pr = padding
    sh, sw = strides
    dilation_h, dilation_w = dilation

    OH = (DH - KH + pt + pb) // sh + 1
    OW = (DW - KW + pl + pr) // sw + 1

    HSTR, WSTR = strides

    pad_h = pt + pb
    pad_w = pl + pr

    pad_height = dh + pad_h
    pad_width = dw + pad_w
    is_kernel_1x1 = KH == 1 and KW == 1
    dilated_kernel_h = (kh - 1) * dilation_h + 1
    dilated_kernel_w = (kw - 1) * dilation_w + 1
    out_height = (dh + pad_h - dilated_kernel_h) // HSTR + 1
    out_width = (dw + pad_w - dilated_kernel_w) // WSTR + 1

    outsize = out_height * out_width
    k_size = kh * kw
    # pack data
    DOPAD = (pad_h != 0 or pad_w != 0)
    if DOPAD:
        data_pad = topi.nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")
    else:
        data_pad = data

    M = int(B * out_height * out_width)
    K = int(kh*kw*IC)
    N = int(OC)
    # M = int(B * out_height * out_width)
    # K = int(dilated_kernel_h * dilated_kernel_h * IC)
    # N = int(OC)

    idxdiv = tvm.indexdiv
    idxmod = tvm.indexmod
    # print(M,K,N)
    # Create schedule config

    cfg.define_split("tile_ic", K, num_outputs=2)
    cfg.define_split("tile_oc", OC, num_outputs=3)
    cfg.define_split("tile_ow", OW, num_outputs=3, filter=lambda y: y.size[-1] <= 64)
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if OH > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])
        unroll_kw = cfg["unroll_kw"].val

    ic_bn, oc_bn, ow_bn = (cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1],cfg["tile_ow"].size[-1])


    col_shape = (M, K)

    data_im2col = tvm.compute(col_shape,
                              lambda s, t: data_pad[idxdiv(s, outsize),
                                                    idxdiv(t, k_size),
                                                    idxdiv(idxmod(s, outsize), out_width) * strides[0] + (
                                                            idxdiv(idxmod(t, k_size), kw) * dilation_h),
                                                    idxmod(idxmod(s, outsize), out_width) * strides[1] + (
                                                            idxmod(idxmod(t, k_size), kw) * dilation_w)],
                              name='data_im2col')
    # print("#################################################################")
    # print("ic_bn, oc_bn, ow_bn",ic_bn, oc_bn, ow_bn)


    kernel_shape = (N, K)
    kernel_im2col = tvm.compute(kernel_shape,
                                lambda o, t: kernel[o,
                                                    idxdiv(t, k_size),
                                                    idxdiv(idxmod(t, k_size), kw),
                                                    idxmod(idxmod(t, k_size), kw)],
                                name='kernel_im2col')


    ########################

    packw_bn = oc_bn
    packw_shape = (N // packw_bn, K, packw_bn)
    packw = tvm.compute(packw_shape,
                        lambda z, y, x: kernel_im2col[z * packw_bn + x, y], name="packed_weight")

    k = tvm.reduce_axis((0, K), name="k")
    C = tvm.compute((M, N),
                    lambda y, x: tvm.sum(
                        data_im2col[y, k].astype(dtype) *
                        packw[tvm.indexdiv(x, packw_bn), k, tvm.indexmod(x, packw_bn)].astype(dtype),
                        axis=k))
    #########################
    # # create tuning space
    # vec = ic_bn
    # k = tvm.reduce_axis((0, K // vec), "k")
    # CC = tvm.compute((M, N, vec),
    #                  lambda z, y, x: tvm.sum(data_im2col[z, k * vec + x].astype(dtype) * kernel_im2col[y, k * vec + x].astype(dtype),
    #                                          axis=k))
    # kk = tvm.reduce_axis((0, vec), "kk")
    # C = tvm.compute((M, N), lambda y, x: tvm.sum(CC[y, x, kk], axis=kk))
    #########################

    out_shape = (B, OC, out_height, out_width)
    Out = tvm.compute(out_shape, lambda b, oc, oh, ow:
    C[b * out_width * out_height + oh * out_width + ow, oc].astype(out_dtype), name='Out')

    s = tvm.create_schedule(Out.op)

    #data_pad
    if DOPAD:
        s[data_pad].compute_inline()
    # b,ic,h,w = s[data_pad].op.axis
    # b_ic = s[data_pad].fuse(b,ic)
    # s[data_pad].parallel(b_ic)


    #data_im2col
    x, z = s[data_im2col].op.axis
    # xo, xi = s[data_im2col].split(x, factor=ow_bn)
    # zo, zi = s[data_im2col].split(z, factor=ic_bn)
    xt, xo, xi = cfg["tile_ow"].apply(s, data_im2col, x)
    zo, zi = cfg["tile_ic"].apply(s, data_im2col, z)
    s[data_im2col].reorder(xt, zo,xo, xi, zi)
    xzo = s[data_im2col].fuse(xt,zo)
    s[data_im2col].parallel(xzo)
    s[data_im2col].vectorize(zi)


    #kernel_im2col
    y, z = s[kernel_im2col].op.axis
    yt, yo, yi = cfg["tile_oc"].apply(s, kernel_im2col, y)
    zo, zi = cfg["tile_ic"].apply(s, kernel_im2col, z)
    s[kernel_im2col].reorder(yt, zo, yo,  yi, zi)
    yzo = s[kernel_im2col].fuse(yt, zo)
    s[kernel_im2col].parallel(yzo)
    s[kernel_im2col].vectorize(zi)
    #####################################
    #same with lpmmv:  C  and CC
    CC = s.cache_write(C, "global")
    y, x = s[C].op.axis
    k, = s[CC].op.reduce_axis
    #yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    #xt, xo, xi = cfg["tile_x"].apply(s, C, x)
    yt, yo, yi = cfg["tile_ow"].apply(s, C, y)
    xt, xo, xi = cfg["tile_oc"].apply(s, C, x)
    s[C].reorder(yt, xt, yo, xo, yi, xi)
    xyt = s[C].fuse(yt, xt)
    xyo = s[C].fuse(yo, xo)
    s[C].parallel(xyt)
    s[C].unroll(yi)
    s[C].vectorize(xi)

    s[CC].compute_at(s[C], xyo)
    y, x = s[CC].op.axis
    ko, ki = cfg["tile_ic"].apply(s, CC, k)
    s[CC].reorder(ko, ki, y, x)
    s[CC].vectorize(x)
    s[CC].unroll(y)
    s[CC].unroll(ki)

    z, y, x = s[packw].op.axis
    s[packw].reorder(z, x, y)
    s[packw].parallel(z)
    s[packw].vectorize(x)
    #####################################

    b, oc, oh, ow = s[Out].op.axis
    oct, oco, oci = cfg["tile_oc"].apply(s, Out, oc)
    owt, owo, owi = cfg["tile_ow"].apply(s, Out, ow)
    s[Out].reorder(b, oct, oh, owt, oco, owo, oci, owi)
    b_oct_oh = s[Out].fuse(b,oct,oh)
    s[Out].parallel(b_oct_oh)
    s[Out].vectorize(owi)
    #print(tvm.lower(s,[data,kernel,C,Out],simple_mode=True))
    return s, [data, kernel, Out]

def convTestSchedule_opt_dnmm332_0304(data_shape, kernel_shape, padding, strides, dilation, dtype):
    B, IC, DH, DW = data_shape
    OC, IC, KH, KW = kernel_shape
    data = tvm.placeholder(data_shape, name='data', dtype=dtype)
    kernel = tvm.placeholder(kernel_shape, name='kernel', dtype=dtype)
    cfg = autotvm.get_config()
    out_dtype = dtype
    pt, pl, pb, pr = padding
    sh, sw = strides
    dilation_h, dilation_w = dilation

    OH = (DH - KH + pt + pb) // sh + 1
    OW = (DW - KW + pl + pr) // sw + 1

    HSTR, WSTR = strides

    pad_h = pt + pb
    pad_w = pl + pr

    pad_height = dh + pad_h
    pad_width = dw + pad_w
    is_kernel_1x1 = KH == 1 and KW == 1
    dilated_kernel_h = (kh - 1) * dilation_h + 1
    dilated_kernel_w = (kw - 1) * dilation_w + 1
    out_height = (dh + pad_h - dilated_kernel_h) // HSTR + 1
    out_width = (dw + pad_w - dilated_kernel_w) // WSTR + 1

    outsize = out_height * out_width
    k_size = kh * kw
    # pack data
    DOPAD = (pad_h != 0 or pad_w != 0)
    if DOPAD:
        data_pad = topi.nn.pad(data, (0, 0, pt, pl), (0, 0, pb, pr), name="data_pad")
    else:
        data_pad = data

    M = int(B * out_height * out_width)
    K = int(kh*kw*IC)
    N = int(OC)
    # M = int(B * out_height * out_width)
    # K = int(dilated_kernel_h * dilated_kernel_h * IC)
    # N = int(OC)

    idxdiv = tvm.indexdiv
    idxmod = tvm.indexmod
    # print(M,K,N)
    # Create schedule config

    cfg.define_split("tile_ic", K, num_outputs=2)
    cfg.define_split("tile_oc", OC, num_outputs=3)
    cfg.define_split("tile_ow", OW, num_outputs=3, filter=lambda y: y.size[-1] <= 64)
    if is_kernel_1x1:
        cfg.define_knob("tile_oh", [1, 2] if OH > 1 else [1])
    else:
        cfg.define_knob("unroll_kw", [True, False])
        unroll_kw = cfg["unroll_kw"].val

    ic_bn, oc_bn, ow_bn = (cfg["tile_ic"].size[-1], cfg["tile_oc"].size[-1],cfg["tile_ow"].size[-1])


    col_shape = (M, K)

    data_im2col = tvm.compute(col_shape,
                              lambda s, t: data_pad[idxdiv(s, outsize),
                                                    idxdiv(t, k_size),
                                                    idxdiv(idxmod(s, outsize), out_width) * strides[0] + (
                                                            idxdiv(idxmod(t, k_size), kw) * dilation_h),
                                                    idxmod(idxmod(s, outsize), out_width) * strides[1] + (
                                                            idxmod(idxmod(t, k_size), kw) * dilation_w)],
                              name='data_im2col')
    # print("#################################################################")
    # print("ic_bn, oc_bn, ow_bn",ic_bn, oc_bn, ow_bn)


    kernel_shape = (N, K)
    kernel_im2col = tvm.compute(kernel_shape,
                                lambda o, t: kernel[o,
                                                    idxdiv(t, k_size),
                                                    idxdiv(idxmod(t, k_size), kw),
                                                    idxmod(idxmod(t, k_size), kw)],
                                name='kernel_im2col')


    ########################
    vec = ic_bn
    k = tvm.reduce_axis((0, K // vec), "k")
    CC = tvm.compute((M, N, vec),
                     lambda z, y, x: tvm.sum(data_im2col[z, k * vec + x].astype(dtype) * kernel_im2col[y, k * vec + x].astype(dtype),
                                             axis=k))
    kk = tvm.reduce_axis((0, vec), "kk")
    C = tvm.compute((M, N), lambda y, x: tvm.sum(CC[y, x, kk], axis=kk))
    #########################

    out_shape = (B, OC, out_height, out_width)
    Out = tvm.compute(out_shape, lambda b, oc, oh, ow:
    C[b * out_width * out_height + oh * out_width + ow, oc].astype(out_dtype), name='Out')

    s = tvm.create_schedule(Out.op)

    # data_pad
    if DOPAD:
        s[data_pad].compute_inline()
    # b,ic,h,w = s[data_pad].op.axis
    # b_ic = s[data_pad].fuse(b,ic)
    # s[data_pad].parallel(b_ic)

    # data_im2col
    x, z = s[data_im2col].op.axis
    # xo, xi = s[data_im2col].split(x, factor=ow_bn)
    # zo, zi = s[data_im2col].split(z, factor=ic_bn)
    xt, xo, xi = cfg["tile_ow"].apply(s, data_im2col, x)
    zo, zi = cfg["tile_ic"].apply(s, data_im2col, z)
    s[data_im2col].reorder(xt, zo, xo, xi, zi)
    xzo = s[data_im2col].fuse(xt, zo)
    s[data_im2col].parallel(xzo)
    s[data_im2col].vectorize(zi)

    # kernel_im2col
    y, z = s[kernel_im2col].op.axis
    yt, yo, yi = cfg["tile_oc"].apply(s, kernel_im2col, y)
    zo, zi = cfg["tile_ic"].apply(s, kernel_im2col, z)
    s[kernel_im2col].reorder(yt, zo, yo, yi, zi)
    yzo = s[kernel_im2col].fuse(yt, zo)
    s[kernel_im2col].parallel(yzo)
    s[kernel_im2col].vectorize(zi)
    #####################################
    # same with dnmm332:  C  and CC
    y, x = s[C].op.axis
    kk, = s[C].op.reduce_axis
    yt, yo, yi = cfg["tile_ow"].apply(s, C, y)
    xt, xo, xi = cfg["tile_oc"].apply(s, C, x)
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
    #####################################

    b, oc, oh, ow = s[Out].op.axis
    oct, oco, oci = cfg["tile_oc"].apply(s, Out, oc)
    owt, owo, owi = cfg["tile_ow"].apply(s, Out, ow)
    s[Out].reorder(b, oct, oh, owt, oco, owo, oci, owi)
    b_oct_oh = s[Out].fuse(b, oct, oh)
    s[Out].parallel(b_oct_oh)
    s[Out].vectorize(owi)
    # print(tvm.lower(s,[data,kernel,C,Out],simple_mode=True))
    return s, [data, kernel, Out]


def Evaluation(src, ctx, B, IC, dh, dw, OC, kh, kw, padding, strides, dilation, dtype, schedule, data, kernel, result,out_shape):
    s, arg_bufs = schedule(B, IC, dh, dw, OC, kh, kw, padding, strides, dilation, dtype)
    func = tvm.build(s, arg_bufs)
    c_tvm = tvm.nd.empty(out_shape)
    print("c_tvm shape is:", c_tvm.shape)
    print("data shape is:", data.shape)
    print("kernel shape is:", kernel.shape)
    print("result shape is:", result.shape)
    func(tvm.nd.array(data), tvm.nd.array(kernel), c_tvm)
    res = c_tvm.asnumpy()
    #print(res)
    out_res = res.reshape(1,-1)
    #print(out_res)
    np.savetxt("out_res.txt",out_res)
    print("out_res is:",out_res)
    print("out_res shape is:",out_res.shape)
    print("result is:",result)
    print("result shape is:", result.shape)

    #print("res shape is:", res.shape)
    tvm.testing.assert_allclose(result, out_res, rtol=1e-1)
    evaluator = func.time_evaluator(func.entry_name, ctx, number=100)
    print('TVM: %f' % evaluator(tvm.nd.array(data), tvm.nd.array(kernel), c_tvm).mean)


def TestConv(out_shape, schedule):
    data = np.loadtxt("img.txt").reshape(data_shape).astype(dtype)
    kernel = np.loadtxt("filter.txt").reshape(kernel_shape).astype(dtype)
    out = np.loadtxt("z.txt").reshape(1,-1).astype(dtype)
    Evaluation(src, ctx, B, IC, dh, dw, OC, kh, kw, padding, strides, dilation, dtype,
               schedule=schedule, data=data, kernel=kernel, result=out,out_shape=out_shape)

def buildandevaluation(s,A,B,C,a,b,c,ctx,c_np,impl):
    with relay.build_config(opt_level=3):
        func = tvm.build(s, [A, B, C], target=target, name='conv2d')
    assert func
    func(a, b, c)
    # print(c.asnumpy())
    # print(c_np)

    np.savetxt(impl.__name__+"numpyres.txt",c_np.reshape((1,-1)))
    np.savetxt(impl.__name__+"convres.txt",c.asnumpy().reshape((1,-1)))

if __name__ == '__main__':

    '''
args=(('TENSOR', (1, 512, 7, 7), 'float32'), //input:[batch, in_channel, in_height, in_width]
('TENSOR', (512, 512, 3, 3), 'float32'), //filter:[num_filter, in_channel, filter_height, filter_width]
//int or a list/tuple of two ints
(1, 1), //stride size: [stride_height, stride_width]
(1, 1), //padding size:[pad_height, pad_width]
 (1, 1), //dilation size:[dilation_height, dilation_width]
 'NCHW', //layout: str layout of data
 'float32'),//dtype
    '''


    # B, IC, dw, OC = 1, 3, 3, 3
    # kw = 3
    # p = 1
    # s = 1
    # d = 1

    B = int(sys.argv[1])
    IC = int(sys.argv[2])
    dw = int(sys.argv[3])
    OC = int(sys.argv[4])
    kw = int(sys.argv[5])
    s = int(sys.argv[6])
    p = int(sys.argv[7])
    d = int(sys.argv[8])

    print("B IC DW OC KW s p d",B, IC ,dw ,OC ,kw ,s ,p ,d)

    kh = kw
    dh = dw

    data_shape = (B, IC, dh, dw)
    kernel_shape = (OC, IC, kh, kw)
    padding = (p,p,p,p)
    strides = (s, s)
    dilation = (d, d)
    print("data shape:",data_shape)
    print("kernel shape:",kernel_shape)
    print("padding:",padding)
    print("strides:",strides)
    print("dilation:",dilation)

    # out_shape = (1,3,3,3)
    # col_shape = (9,3)

    dtype = "float32"
    target = 'llvm'
    ctx = tvm.context(target, 0)

    impl = convTestSchedule_opt_rpmm332_0304
    tsk = autotvm.task.create(impl, args=(data_shape, kernel_shape, padding, strides, dilation, dtype), target=target)
    n_trial = 500
    n_trial = min(n_trial, len(tsk.config_space))

    # print(tsk.config_space)
    measure_option = autotvm.measure_option(builder='local', runner=autotvm.LocalRunner(number=3))
    src = impl.__name__ + '_XGBtuner_conv2d_' + str(data_shape) + str(kernel_shape) +str(padding)+str(strides)+str(dilation)+ '.log'
    print("XGBoost:")
    XGBtuner = autotvm.tuner.XGBTuner(tsk)
    XGBtuner.tune(n_trial=n_trial, early_stopping=300, measure_option=measure_option, callbacks=[autotvm.callback.progress_bar(n_trial),autotvm.callback.log_to_file(src)])
    # XGBtuner = autotvm.tuner.XGBTuner(tsk, loss_type="regg", optimizer="reg")
    # XGBtuner.tuneMCL(n_trial=n_trial,
    #                 early_stopping=300,
    #                 measure_option=measure_option,
    #                 callbacks=[autotvm.callback.progress_bar(n_trial),
    #                            autotvm.callback.log_to_file(src)],
    #                 initConfig=True, useFilter=True, useRecommend=False, sch="convim2coldnmm")

    autotvm.record.pick_best(src, src + ".best")


    #
    # data_tensor = tvm.placeholder((B, IC, dh, dw), name='data', dtype=dtype)
    # kernel_tensor = tvm.placeholder((OC, IC, kh, kw), name='weight', dtype=dtype)
    #
    # data_value = np.linspace(-10,10,B*IC*dh*dw).reshape([B,IC,dh,dw]).astype(dtype)
    # kernel_value = np.linspace(-10,10,OC*IC*kh*kw).reshape([OC,IC,kh,kw]).astype(dtype)
    # out = conv(data_value, kernel_value, s, p, d)
    #
    # #out = convTogemmByIm2col([B, IC, dh, dw], [OC, IC, kh, kw], s, p, d).astype(dtype)
    # print("###################################################")
    # print("Input shape is:",data_value.shape)
    # print("kernel shape is:",kernel_value.shape)
    # print("Out shape is:",out.shape)
    # print("###################################################")
    # a = tvm.nd.array(data_value,ctx)
    # b = tvm.nd.array(kernel_value,ctx)
    # c = tvm.nd.array(out, ctx)
    # impl = conv2d0
    # s, [data, kernel, Out] = impl(data_shape, kernel_shape, padding, strides, dilation, dtype)
    # print("###################################################")
    # print("data tensor shape is:",data)
    # print("kernel tensor shape shape is:",kernel)
    # print("Out shape is:",Out)
    # print("###################################################")
    # buildandevaluation(s, data, kernel, Out, a, b, c, ctx, out,impl)




    '''
    conv2d  和  convTestSchedule3 运算结果一致
    只支持dilation=1的校验
    '''
