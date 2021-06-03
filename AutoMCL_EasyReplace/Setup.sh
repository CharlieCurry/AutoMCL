echo Setup Start.
mv $TVM_PATH/python/tvm/autotvm/tuner $TVM_PATH/python/tvm/autotvm/tuner_tvm
cp -r AutoConfig $TVM_PATH/python/tvm/autotvm/tuner
mv $TVM_PATH/python/tvm/autotvm/task $TVM_PATH/python/tvm/autotvm/task_tvm
cp -r InitConfigTask $TVM_PATH/python/tvm/autotvm/task
mv $TVM_PATH/topi/python/topi/x86 $TVM_PATH/topi/python/topi/x86_tvm
cp -r AS_OS $TVM_PATH/topi/python/topi/x86
echo Setup Done.
