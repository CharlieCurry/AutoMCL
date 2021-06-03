# -*- coding:utf8 -*
import mxnet as mx
from mxnet import gluon
import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd
import time
import warnings
import sys
def infer():
    s = time.time()
    output = net.forward(x)
    e = time.time()
    print("infer time(ms):", 1000*(e - s))
    print(output.shape)
    return output


if __name__ == "__main__":

    batch_size = int(sys.argv[1])
    warnings.filterwarnings("ignore")
    network_name = "layer6"
    ctx = mx.cpu(0)
    x = nd.random.uniform(shape=(batch_size, 128))

    net = gluon.SymbolBlock.imports(symbol_file=network_name + '-symbol.json', input_names=['data'],
                                    param_file=network_name + '-0000.params', ctx=ctx)

    print("warm up...")
    infer()
    print("")
    infer()
    infer()
    infer()
    infer()
    infer()




    # print(int(input_size/batch_size))
    # s = time.time()
    # for i in range(int(input_size/batch_size)):
    #     output = net.forward(x)
    # e = time.time()
    # print("infer time(ms):", 1000*(e - s))
    # print(output.shape)

