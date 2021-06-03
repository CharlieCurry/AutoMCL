User Test: This is a simple guide to use AutoMCL.
©À©¤©¤ e2e
©¦ ©À©¤©¤ CNNs
©¦ ©¦ ©¸©¤©¤ tune_model_e2e.py     (1)
©¦ ©¸©¤©¤ FCNNs
©¦     ©À©¤©¤ build_net
©¦     ©¦ ©À©¤©¤ build_net.py
©¦     ©¦ ©À©¤©¤ infer.py
©¦     ©¦ ©À©¤©¤ infer_mxnet.py
©¦     ©¦ ©À©¤©¤ layer4-0000.params
©¦     ©¦ ©À©¤©¤ layer4-symbol.json
©¦     ©¦ ©À©¤©¤ layer6-0000.params
©¦     ©¦ ©À©¤©¤ layer6-symbol.json
©¦     ©¦ ©À©¤©¤ net
©¦     ©¦ ©¦ ©À©¤©¤ X_2.data-00000-of-00001
©¦     ©¦ ©¦ ©À©¤©¤ X_2.index
©¦     ©¦ ©¦ ©À©¤©¤ X_2.meta
©¦     ©¦ ©¦ ©¸©¤©¤ checkpoint
©¦     ©¦ ©À©¤©¤ test_net-0000.params
©¦     ©¦ ©¸©¤©¤ test_net-symbol.json
©¦     ©À©¤©¤ from_mynet.py         (2)
©¦     ©¸©¤©¤ paradnn
©¦         ©À©¤©¤ fc_cpu_float32-symbol.json
©¦         ©À©¤©¤ fc_trace_10sec.json
©¦         ©À©¤©¤ fc_trace_10sec_opbreakdown.json
©¦         ©¸©¤©¤ json_parse.py
©¸©¤©¤ op
    ©À©¤©¤ conv2d_template
    ©¦ ©¸©¤©¤ conv2d.py             (3)
    ©¸©¤©¤ dense_template
        ©À©¤©¤ dnmm.py               (4)
        ©À©¤©¤ dnmm332.py
        ©À©¤©¤ dpmm.py
        ©À©¤©¤ lpmm.py
        ©À©¤©¤ rpmm.py
        ©À©¤©¤ rpmmv.py
        ©À©¤©¤ tmm.py
        ©¸©¤©¤ ttmm.py


    (1)USEAGE: "python tune_model_e2e.py [batch size]"
      optional networks in [resnet,inception,vgg,mobilenet,squeezenet_v1]

      Relevant experiments at "AutoMCL_Repository/PaperData/e2e_conv2d(10_e2e@intel1)","AutoMCL_Repository/PaperData/e2e_conv2d(e2e20210410@mechrev)",
                              "AutoMCL_Repository/PaperData/e2e_conv2d(e2e@AMD)","AutoMCL_Repository/PaperData/e2e_conv2d(e2e@intel2)" could be reproduce.



    (2)USEAGE: "python from_mynet.py [network] [batch_size] [hidden_first_layer_number] [output_layer_number]"
      optional networks in [FC5,FC7]

      Relevant experiments at "AutoMCL_Repository/PaperData/e2e_dense(20210128mxnet@AMD)","AutoMCL_Repository/PaperData/e2e_dense(20210128mxnet@intel2)".




    (3)USEAGE: "python conv2d.py [B] [IC] [dw] [OC] [kw] [s] [p] [d]" and should preset impl in conv2d.py code.
      optional impl in [convTestSchedule_opt_rpmm332_0304,convTestSchedule_opt_dnmm332_0304,conv2d_opt,conv2d0,conv2d_1x1]

      Relevant experiments at "AutoMCL_Repository/PaperData/conv2d_3_schedule(packed20210409@intel1)".



    (4)USEAGE: "python conv2d.py [M] [K] [N]"
      optional impl in [dnmm,dnmm332,lpmm,dpmm,rpmm,rpmmv,tmm,ttmm]

       Relevant experiments at "AutoMCL_Repository/PaperData/dense_8_schedule(20201111@intel1)".


