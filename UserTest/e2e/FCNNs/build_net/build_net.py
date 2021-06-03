# -*- coding:utf8 -*
from mxnet import gluon
import mxnet as mx
from mxnet.gluon import nn
from mxnet import nd
import time

def layer4():
    network_name = "layer4"
    net=gluon.nn.HybridSequential()
    net.add(nn.Dense(128, activation='relu'))
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(512, activation='relu'))
    net.add(nn.Dense(1024, activation='relu'))
    net.add(nn.Dense(1000))
    return net,network_name

def layer6():
    network_name = "layer6"
    net=gluon.nn.HybridSequential()
    net.add(nn.Dense(128, activation='relu'))
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(512, activation='relu'))
    net.add(nn.Dense(1024, activation='relu'))
    net.add(nn.Dense(2048, activation='relu'))
    net.add(nn.Dense(4096, activation='relu'))
    net.add(nn.Dense(1000))
    return net,network_name



if __name__ == "__main__":
    net,network_name = layer6()
    net.hybridize()
    net.initialize()
    x = nd.random.uniform(shape=(2000, 128))
    s = time.time()
    output = net(x)
    e = time.time()
    print("infer time:",e-s)
    print(output)

    # net = gluon.nn.SymbolBlock.imports('fc_cpu_float32-symbol.json',['data'],param_file='resnet18-0000.params',ctx=mx.cpu())
    net.export(network_name, epoch=0)

