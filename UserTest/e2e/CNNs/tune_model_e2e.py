

import numpy as np

import tvm
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime



def get_network(name, batch_size):
    """Get the symbol definition and random weight of a network"""
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)

    if "resnet" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.resnet.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif "vgg" in name:
        n_layer = int(name.split('-')[1])
        mod, params = relay.testing.vgg.get_workload(num_layers=n_layer, batch_size=batch_size, dtype=dtype)
    elif name == 'mobilenet':
        mod, params = relay.testing.mobilenet.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'squeezenet_v1.1':
        mod, params = relay.testing.squeezenet.get_workload(batch_size=batch_size, version='1.1', dtype=dtype)
    elif name == 'inception_v3':
        input_shape = (1, 3, 299, 299)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == 'mxnet':
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model
        block = get_model('resnet18_v1', pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={'data': input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs)
        mod = relay.Module.from_expr(net)
    else:
        raise ValueError("Unsupported network: " + name)

    return mod, params, input_shape, output_shape


def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=500,
               log_filename='tuning.log',
               use_transfer_learning=True,
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

    for i, tsk in enumerate((tasks)):
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

        # do tuning
        n_trial = 500
        n_trial = min(n_trial, len(tsk.config_space))


        #任务单独保存
        # import shutil
        # shutil.copyfile('res.txt', 'res'+prefix+'.txt')

        print("n_trial=",n_trial)
        print("early_stopping=",early_stopping)
        # tuner_obj.tune(n_trial=n_trial,
        #                early_stopping=early_stopping,
        #                measure_option=measure_option,
        #                callbacks=[
        #                    autotvm.callback.progress_bar(n_trial, prefix=prefix),
        #                    autotvm.callback.log_to_file(tmp_log_file)])

        XGBtuner = autotvm.tuner.XGBTuner(tsk, loss_type="regg", optimizer="reg")
        XGBtuner.tuneMCL(n_trial=n_trial,
                        early_stopping=early_stopping,
                        measure_option=measure_option,
                        callbacks=[autotvm.callback.progress_bar(n_trial),
                                   autotvm.callback.log_to_file(tmp_log_file)],
                        initConfig=True, useFilter=True, useRecommend=False, sch="conv2d")
        # np.savetxt("feas"+str(i+1)+".txt", tuner_obj.cost_model.feas, fmt='%s', delimiter=' ')
        # np.savetxt("x_train"+str(i+1)+".txt", tuner_obj.cost_model.x_train, fmt='%s', delimiter=' ')
        # np.savetxt("y_train"+str(i+1)+".txt", tuner_obj.cost_model.y_train, fmt='%s', delimiter=' ')
        #他做到了同算子--》空间维度相同的迁移学习,那么实际上resnet18--->resnet50?

    autotvm.record.pick_best(tmp_log_file, log_filename)






def evaluate(tmp_log_file,log_filename,params):
    # pick best records to a cache file
    # autotvm.record.pick_best(tmp_log_file, log_filename)
    #os.remove(tmp_log_file)
    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        # # export library
        # tmp = tempdir()
        # filename = "net.tar"
        # lib.export_library(tmp.relpath(filename))

        # load parameters
        ctx = tvm.context(str(target), 0)
        module = runtime.create(graph, lib, ctx)
        data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        module.set_input('data', data_tvm)
        module.set_input(**params)

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=3, repeat=50)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.3f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))



if __name__ == "__main__":
    import sys
    import warnings

    warnings.filterwarnings("ignore")
    batch_size = int(sys.argv[1])
    target = "llvm"
    network = 'vgg-16'
    #network = 'vgg-13'
    #network = 'mobilenet'
    #network = 'inception_v3'
    log_file = "%s.log" % network
    dtype = 'float32'
    tmp_log_file = log_file + ".tmp"
    tuning_option = {
        'log_filename': log_file,
        'tuner': 'xgb',
        'n_trial': 1000,
        'early_stopping': 300,
        'measure_option': autotvm.measure_option(
            builder=autotvm.LocalBuilder(timeout=2000),
            # runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
            runner=autotvm.LocalRunner(number=3)
        ),
    }
    print("Extract tasks...")
    mod, params, input_shape, out_shape = get_network(network, batch_size=batch_size)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              params=params, ops=(relay.op.nn.conv2d,))
    print("Tuning...")
    tune_tasks(tasks, **tuning_option)
    evaluate(tmp_log_file,log_file,params)



