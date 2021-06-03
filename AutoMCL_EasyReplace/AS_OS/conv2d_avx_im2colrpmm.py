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
# pylint: disable=invalid-name,unused-variable,unused-argument,invalid-name
"""Conv2D schedule on for Intel CPU"""

from __future__ import absolute_import as _abs
'''base'''
import tvm
from tvm.autotvm.task.space import SplitEntity, OtherOptionEntity

from ..nn.util import infer_pad
from ..generic import conv2d as conv2d_generic
from ..util import get_const_tuple
from .tensor_intrin import dot_16x1x16_uint8_int8_int32
from .util import get_fp32_len

def _fallback_schedule(cfg, wkl):
    simd_width = get_fp32_len()
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride
    out_width = (wkl.width + 2 * WPAD - wkl.wkernel) // WSTR + 1

    oc_bn = 1
    for bn in range(simd_width, 0, -1):
        if wkl.out_filter % bn == 0:
            oc_bn = bn
            break

    ic_bn = 1
    for bn in range(oc_bn, 0, -1):
        if wkl.in_filter % bn == 0:
            ic_bn = bn
            break

    reg_n = 1
    for n in range(31, 0, -1):
        if out_width % n == 0:
            reg_n = n
            break

    cfg["tile_ic"] = SplitEntity([wkl.in_filter // ic_bn, ic_bn])
    cfg["tile_oc"] = SplitEntity([wkl.out_filter // oc_bn,1, oc_bn])
    cfg["tile_ow"] = SplitEntity([out_width // reg_n, 1,reg_n])
    cfg["unroll_kw"] = OtherOptionEntity(False)


def _fallback_schedule_int8(cfg, wkl):
    HPAD, WPAD = wkl.hpad, wkl.wpad
    HSTR, WSTR = wkl.hstride, wkl.wstride
    out_width = (wkl.width + 2 * WPAD - wkl.wkernel) // WSTR + 1

    oc_bn = 16
    assert wkl.out_filter % oc_bn == 0

    ic_bn = 1
    for bn in range(oc_bn, 0, -4):
        if wkl.in_filter % bn == 0:
            ic_bn = bn
            break
    assert wkl.in_filter % 4 == 0

    reg_n = 1
    for n in range(31, 0, -1):
        if out_width % n == 0:
            reg_n = n
            break

    cfg["tile_ic"] = SplitEntity([wkl.in_filter // ic_bn, ic_bn])
    cfg["tile_oc"] = SplitEntity([wkl.out_filter // oc_bn, oc_bn])
    cfg["tile_ow"] = SplitEntity([out_width // reg_n, reg_n])
    cfg["unroll_kw"] = OtherOptionEntity(False)


def _schedule_conv(s, cfg, data, data_pad, data_im2col, kernel_im2col, C, Out, last):
    # fetch schedule
    print("im2col rpmm schedule")

    # no stride and padding info here
    padding = infer_pad(data, data_pad)
    HPAD, WPAD = padding
    DOPAD = (HPAD != 0 or WPAD != 0)

    A, W = data, kernel_im2col
    A0, A1 = data_pad, data_im2col
    O0, O = Out, last


    # schedule data
    if DOPAD:
        s[A0].compute_inline()
    # batch, ic_chunk, ih, ic_block, iw = s[A1].op.axis
    # parallel_axis = s[A1].fuse(batch, ic_chunk, ih)
    # s[A1].parallel(parallel_axis)


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
    # same with rpmm:  C  and CC
    _ ,packw = s[C].op.input_tensors
    CC = s.cache_write(C, "global")
    y, x = s[C].op.axis
    k, = s[CC].op.reduce_axis
    # yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    # xt, xo, xi = cfg["tile_x"].apply(s, C, x)
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
    if O0 != O:
        s[O0].compute_inline()

    b, oc, oh, ow = s[Out].op.axis
    oct, oco, oci = cfg["tile_oc"].apply(s, Out, oc)
    owt, owo, owi = cfg["tile_ow"].apply(s, Out, ow)
    s[Out].reorder(b, oct, oh, owt, oco, owo, oci, owi)
    b_oct_oh = s[Out].fuse(b, oct, oh)
    s[Out].parallel(b_oct_oh)
    s[Out].vectorize(owi)
    return s


def _schedule_conv_NCHWc(s, cfg, data, conv_out, last):
    # fetch schedule
    reg_n, unroll_kw = cfg["tile_ow"].size[-1], cfg["unroll_kw"].val
    _, _, _, _, ic_bn = get_const_tuple(data.shape)

    # schedule data
    A = data
    if isinstance(s[A].op, tvm.tensor.ComputeOp):
        batch, ic_chunk, ih, iw, ic_block = s[A].op.axis
        parallel_axis = s[A].fuse(batch, ic_chunk, ih)
        s[A].parallel(parallel_axis)

    # schedule 5-D NCHW[x]c conv
    C, Out = conv_out, last
    CC = s.cache_write(C, 'global')
    data_im2col, packw = s[C].op.input_tensors

    kernel_im2col, = s[packw].op.input_tensors

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
    #zo, zi = cfg["tile_ic"].apply(s, kernel_im2col, z)
    zo, zi = s[kernel_im2col].split(z, factor=ic_bn)

    s[kernel_im2col].reorder(yt, zo, yo, yi, zi)
    yzo = s[kernel_im2col].fuse(yt, zo)
    s[kernel_im2col].parallel(yzo)
    s[kernel_im2col].vectorize(zi)
    #####################################
    # same with rpmm:  C  and CC
    CC = s.cache_write(C, "global")
    y, x = s[C].op.axis
    k, = s[CC].op.reduce_axis
    # yt, yo, yi = cfg["tile_y"].apply(s, C, y)
    # xt, xo, xi = cfg["tile_x"].apply(s, C, x)
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
    #ko, ki = cfg["tile_ic"].apply(s, CC, k)
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
    oct, oco, oci = cfg["tile_oc"].apply(s, Out, oc)
    owt, owo, owi = cfg["tile_ow"].apply(s, Out, ow)
    s[Out].reorder(b, oct, oh, owt, oco, owo, oci, owi)
    b_oct_oh = s[Out].fuse(b, oct, oh)
    s[Out].parallel(b_oct_oh)
    s[Out].vectorize(owi)
    return s


def _schedule_conv_NCHWc_int8(s, cfg, data, conv_out, last):
    return conv2d_generic.schedule_conv_NCHWc_cpu_common_int8(s, cfg, data, conv_out, last,
                                                              int32_lanes=16,
                                                              intrin=dot_16x1x16_uint8_int8_int32())
