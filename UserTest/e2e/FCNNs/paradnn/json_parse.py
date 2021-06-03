# -*- coding:utf8 -*
import mxnet as mx

sym, arg_params, aux_params = mx.model.load_checkpoint("fc_cpu_float32", 0)
print(sym)
# print(arg_params)
# print(aux_params)

# 提取中间某层输出帖子特征层作为输出
all_layers = sym.get_internals()
print(all_layers)
sym = all_layers['fc1_output']

# 重建模型
model = mx.mod.Module(symbol=sym, label_names=None)
model.bind(for_training=False, data_shapes=[('data', (1, 3, 112, 112))])
model.set_params(arg_params, aux_params)