import mxnet as mx
from mxnet import gluon
import numpy as np
import tvm
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime
from mxnet.gluon.model_zoo.vision import get_model
import warnings

warnings.filterwarnings('ignore')
def block2symbol(block):
    data = mx.sym.Variable("data")
    sym = block(data)
    args = {}
    auxs = {}
    for k, v in block.collect_params().items():
        args[k] = mx.nd.array(v.data().asnumpy())
    return sym, args, auxs

def get_network(network, batch_size,hidden_first_layer_number,output_layer_number):
    """Get the symbol definition and random weight of a network"""
    file_predix = "./build_net/"
    input_shape = (batch_size, hidden_first_layer_number)
    output_shape = (batch_size, output_layer_number)
    ctx = mx.cpu(0)
    net = gluon.SymbolBlock.imports(symbol_file=file_predix+network+'-symbol.json',
                                        input_names=['data'], param_file=file_predix+network+'-0000.params', ctx=ctx)
    block = net
    shape_dict = {"data": input_shape}
    mod, params = relay.frontend.from_mxnet(block, shape=shape_dict, dtype=dtype)
    net = mod["main"]
    net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
    mod = relay.Module.from_expr(net)
    return block, mod, params, input_shape, output_shape, shape_dict

def tune_tasks(tasks,measure_option,tuner,n_trial,early_stopping,log_filename,use_transfer_learning,
               try_winograd=True):
    if try_winograd:
        for i in range(len(tasks)):
            try:  # try winograd template
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'winograd')
                input_channel = tsk.workload[1][1]
                if input_channel >= 64:
                    tasks[i] = tsk
            except Exception:
                pass

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    # if os.path.exists(tmp_log_file):
    #     os.remove(tmp_log_file)

    for i, tsk in enumerate(reversed(tasks)):
    #for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))
        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        # if use_transfer_learning:
        #     if info.path.isfile(tmp_log_file):
        #         tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))
        n_trial=100000

        # do tuning
        print("config_space=",len(tsk.config_space))
        print("early_stopping=",early_stopping)
        tuner_obj.tune(n_trial=min(n_trial,len(tsk.config_space)),
                      early_stopping=early_stopping,
                      measure_option=measure_option,
                      callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                          autotvm.callback.log_to_file(tmp_log_file)])
        #XGBtuner = autotvm.tuner.XGBTuner(tsk, loss_type="regg", optimizer="reg")
        #XGBtuner.tuneMCL(n_trial=n_trial,
       #                  early_stopping=early_stopping,
       #                  measure_option=measure_option,
       #                  callbacks=[autotvm.callback.progress_bar(n_trial),
       #                             autotvm.callback.log_to_file(tmp_log_file)],
       #                  initConfig=True, useFilter=True, useRecommend=False, sch="autoschedule")
        # np.savetxt("feas"+str(i+1)+".txt", tuner_obj.cost_model.feas, fmt='%s', delimiter=' ')
        # np.savetxt("x_train"+str(i+1)+".txt", tuner_obj.cost_model.x_train, fmt='%s', delimiter=' ')
        # np.savetxt("y_train"+str(i+1)+".txt", tuner_obj.cost_model.y_train, fmt='%s', delimiter=' ')
    autotvm.record.pick_best(tmp_log_file, log_filename)
    #os.remove(tmp_log_file)


def tune_and_evaluate(tuning_opt,batch_size,hidden_first_layer_number,output_layer_number):
    # extract workloads from relay program
    print("Extract tasks...")
    block, mod, params, input_shape, out_shape, shape_dict = get_network(network, batch_size=batch_size,
                                                                         hidden_first_layer_number=hidden_first_layer_number,
                                                                         output_layer_number=output_layer_number)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params, ops=(relay.op.nn.dense,))

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    with autotvm.apply_history_best(log_file):
        # with relay.build_config(opt_level=3):
        with relay.transform.PassContext(opt_level=3):
            graph, lib, params = relay.build_module.build(mod, target=target, params=params)
            ctx = tvm.context(str(target), 0)
            dtype = "float32"
            import tvm.contrib.graph_runtime as runtime
            module = runtime.create(graph, lib, ctx)
            data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
            module.set_input('data', data_tvm)
            module.set_input(**params)
            # evaluate
            print("Evaluate inference time cost...")
            ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=10)
            prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
            print("Mean inference time (std dev): %.5f ms (%.5f ms)" % (np.mean(prof_res), np.std(prof_res)))


    ######################################################################
    # MXNet often use `arg_params` and `aux_params` to store network parameters
    #mx_sym, args, auxs = block2symbol(block)
    #mx.model.save_checkpoint("resnet18_v1", 0, mx_sym, args, auxs)
    # there are 'resnet18_v1-0000.params' and 'resnet18_v1-symbol.json' on disk

    ######################################################################
    # for a normal mxnet model, we start from here
    #mx_sym, args, auxs = mx.model.load_checkpoint("resnet18_v1", 0)
    #mod, relay_params = relay.frontend.from_mxnet(mx_sym, shape_dict, arg_params=args, aux_params=auxs)


if __name__ == "__main__":
    import sys
    import xgboost
    print(xgboost.__version__)
    # case: python from_mynet.py layer4 1 128 1000
    network = sys.argv[1]   # network = 'layer4'
    batch_size = int(sys.argv[2])
    hidden_first_layer_number = int(sys.argv[3])
    output_layer_number = int(sys.argv[4])

    target = "llvm"
    log_file = network+"-"+str(batch_size)+".log"
    dtype = 'float32'
    tuning_option = {
        'log_filename': log_file,
        'tuner': 'xgb',
        'n_trial': 100000,
        'use_transfer_learning':False,
        'early_stopping': 400,
        'measure_option': autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=200),
            # runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
            runner=autotvm.LocalRunner(number=5)
        ),
    }
    tune_and_evaluate(tuning_option,batch_size=batch_size,
                      hidden_first_layer_number=hidden_first_layer_number,
                      output_layer_number=output_layer_number)



