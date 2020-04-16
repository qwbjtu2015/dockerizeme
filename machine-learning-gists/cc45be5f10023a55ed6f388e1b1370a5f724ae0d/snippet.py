#!/usr/bin/python3

import nnvm
import nnvm.frontend.darknet
import tvm.relay.testing.yolo_detection
import tvm.relay.testing.darknet
import tvm.relay.transform as _transform
import matplotlib.pyplot as plt
import numpy as np
import tvm
import onnx
import sys
import cv2
import os
import pickle
import multiprocessing as mp

from ctypes import *
from tvm import relay
from tvm.contrib.download import download, download_testdata
from tvm.contrib.download import download_testdata
from tvm.relay.testing.darknet import __darknetffi__
from tvm.contrib.pickle_memoize import memoize

from tvm import autotvm
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime


MODEL_NAME = "yolov3-tiny"
# source https://github.com/brandonjabr/darknet-YOLO-V2-example/tree/master/videos
VIDEO_FILE = "Office-Parkour.mp4"
# samples for KL calibration
CALIBRATION_SAMPLES = 16

# Replace "llvm" with the correct target of your CPU.
# For example, for AWS EC2 c5 instance with Intel Xeon
# Platinum 8000 series, the target should be "llvm -mcpu=skylake-avx512".
# For AWS EC2 c4 instance with Intel Xeon E5-2666 v3, it should be
# "llvm -mcpu=core-avx2".
target = tvm.target.create('llvm')

# Set number of threads used for tuning based on the number of
# physical CPU cores on your machine.
num_threads = 8


os.environ["TVM_NUM_THREADS"] = str(num_threads)
ctx = tvm.context(str(target), 0)
log_file = MODEL_NAME + ".tune.log"

tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    'n_trial': 2000,
    'early_stopping': 600,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
    ),
}


def run_model(model, params, input_dtype, input_shape, evaluate=False):

    # init input data
    data = np.empty(input_shape, dtype=input_dtype)

    # compile kernels with history best records
    print("Compile with [%s]..." % log_file)
    with autotvm.apply_history_best(log_file):

        with relay.build_config(opt_level=4):
            graph, lib, params = relay.build(model, target=target, params=params)

        [neth, netw] = input_shape[2:]

        ######################################################################
        # Execute on TVM Runtime
        # ----------------------
        # The process is no different from other examples.
        from tvm.contrib import graph_runtime

        module = graph_runtime.create(graph, lib, ctx)
        module.set_input('data', tvm.nd.array(data.astype(input_dtype)))
        module.set_input(**params)

        num_outputs = module.get_num_outputs()
        outputs = [[] for i in range(num_outputs)]

        # evaluate
        if (evaluate):
            print("Evaluate inference time cost...")
            ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=5)
            prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
            print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
                  (np.mean(prof_res), np.std(prof_res)))

        thresh = 0.30
        nms_thresh = 0.35
        coco_name = 'coco.names'
        coco_url = 'https://github.com/siju-samuel/darknet/blob/master/data/' + coco_name + '?raw=true'
        font_name = 'arial.ttf'
        font_url = 'https://github.com/siju-samuel/darknet/blob/master/data/' + font_name + '?raw=true'
        download(coco_url, coco_name)
        font_path = download_testdata(font_url, font_name, module='data')
        with open(coco_name) as f:
            content = f.readlines()

        fpos = 0
        names = [x.strip() for x in content]
        vcap = cv2.VideoCapture(VIDEO_FILE)
        nframes = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
        vcap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while(True):

            ret, frame = vcap.read()
            if not ret:
              break

            img = np.array(frame)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.divide(img, 255.0)
            # prepare for network (resized)
            data = cv2.resize(img, (neth, netw), interpolation = cv2.INTER_AREA)
            data = data.transpose((2, 0, 1))
            data = np.flip(data, 0)
            # prepare for display (original)
            img = img.transpose((2, 0, 1))
            img = np.flip(img, 0)

            # set inputs
            sys.stdout.write("\r")
            sys.stdout.flush()
            sys.stdout.write("Inference on frame %i/%i" % (fpos, nframes))
            sys.stdout.flush()
            module.set_input('data', tvm.nd.array(data.astype(input_dtype)))

            # execute
            module.run()

            tvm_out = []
            if MODEL_NAME == 'yolov2':
                layer_out = {}
                layer_out['type'] = 'Region'
                # Get the region layer attributes (n, out_c, out_h, out_w, classes, coords, background)
                layer_attr = m.get_output(2).asnumpy()
                layer_out['biases'] = m.get_output(1).asnumpy()
                out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                             layer_attr[2], layer_attr[3])
                layer_out['output'] = m.get_output(0).asnumpy().reshape(out_shape)
                layer_out['classes'] = layer_attr[4]
                layer_out['coords'] = layer_attr[5]
                layer_out['background'] = layer_attr[6]
                tvm_out.append(layer_out)

            elif MODEL_NAME == 'yolov3-tiny':
              for i in range(2):
                  layer_out = {}
                  layer_out['type'] = 'Yolo'
                  # Get the yolo layer attributes (n, out_c, out_h, out_w, classes, total)
                  layer_attr = module.get_output(i*4+3).asnumpy()
                  layer_out['biases'] = module.get_output(i*4+2).asnumpy()
                  layer_out['mask'] = module.get_output(i*4+1).asnumpy()

                  out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                               layer_attr[2], layer_attr[3])
                  layer_out['output'] = module.get_output(i*4).asnumpy().reshape(out_shape)
                  layer_out['classes'] = layer_attr[4]
                  tvm_out.append(layer_out)

            elif MODEL_NAME == 'yolov3':
              for i in range(3):
                  layer_out = {}
                  layer_out['type'] = 'Yolo'
                  # Get the yolo layer attributes (n, out_c, out_h, out_w, classes, total)
                  layer_attr = module.get_output(i*4+3).asnumpy()
                  layer_out['biases'] = module.get_output(i*4+2).asnumpy()
                  layer_out['mask'] = module.get_output(i*4+1).asnumpy()

                  out_shape = (layer_attr[0], layer_attr[1]//layer_attr[0],
                               layer_attr[2], layer_attr[3])
                  layer_out['output'] = module.get_output(i*4).asnumpy().reshape(out_shape)
                  layer_out['classes'] = layer_attr[4]
                  tvm_out.append(layer_out)
            else:
                raise ValueError("Unsupported model: " + MODEL_NAME)

            _, im_h, im_w = data.shape

            dets = tvm.relay.testing.yolo_detection.fill_network_boxes((netw, neth), (im_w, im_h), thresh, 1, tvm_out)
            last_layer = net.layers[net.n - 1]
            tvm.relay.testing.yolo_detection.do_nms_sort(dets, last_layer.classes, nms_thresh)
            tvm.relay.testing.yolo_detection.draw_detections(font_path, img, dets, thresh, names, last_layer.classes)

            cv2.imshow('VIDEO', img.transpose(1, 2, 0))
            cv2.waitKey(1)

            fpos = fpos + 1


def calibrate_on_dataset(qmodel, params, input_dtype, input_shape):

    profile_graph = relay.quantize.collect_stats(qmodel)

    # init input data
    data = np.empty(input_shape, input_dtype)


    with relay.build_config(opt_level=3):
        graph, lib, params = relay.build(relay.Module.from_expr(profile_graph), target=target)

    [neth, netw] = input_shape[2:]

    ######################################################################
    # Execute on TVM Runtime
    # ----------------------
    # The process is no different from other examples.
    from tvm.contrib import graph_runtime
    module = graph_runtime.create(graph, lib, ctx)
    module.set_input('data', tvm.nd.array(data.astype(input_dtype)))
    module.set_input(**params)

    num_outputs = module.get_num_outputs()
    outputs = [[] for i in range(num_outputs)]
    print("Network output nodes = %i" % num_outputs)

    vcap = cv2.VideoCapture(VIDEO_FILE)
    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    nframes = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Using video '%s' frames=%i fps=%i time=%i (seconds)" % (VIDEO_FILE, nframes, fps, nframes//fps))

    # random sample
    indexes = np.random.randint(0, nframes-1, CALIBRATION_SAMPLES)

    for idx, fpos in enumerate(indexes):

        vcap.set(cv2.CAP_PROP_POS_FRAMES, fpos)

        ret, frame = vcap.read()

        # prepare frame
        img = np.array(frame)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        data = cv2.resize(img, (neth, netw), interpolation = cv2.INTER_AREA)
        data = data.transpose((2, 0, 1))
        data = np.divide(data, 255.0)
        data = np.flip(data, 0)

        sys.stdout.write("\r")
        sys.stdout.write("Extracting random frame %i out of %i (#%i total)" % (idx+1, len(indexes), nframes))
        sys.stdout.flush()

        # set inputs
        module.set_input('data', tvm.nd.array(data.astype(input_dtype)))

        # execute
        module.run()

        for i in range(num_outputs):
            output = module.get_output(i).asnumpy()
            outputs[i].append(output)

    sys.stdout.write("\n")
    sys.stdout.flush()

    print("Concatenate stats...")
    for i in range(num_outputs):
        outputs[i] = np.concatenate(outputs[i]).reshape(-1)

    print("Compute final KL stats...")
    with mp.Pool() as pool:
        scales = list(pool.map(relay.quantize.kl_divergence.kl_divergence_scale, outputs))

    return scales

def quantize_model(model, params, input_dtype, input_shape, qeval='power2'):

    skip_conv_layers = [0]
    with relay.quantize.qconfig(store_lowbit_output=False, skip_conv_layers=skip_conv_layers):
        from tvm.relay.quantize.quantize import _bind_params
        graph = _bind_params(model['main'], params)
        mod = relay.Module.from_expr(graph)
        optimize = _transform.Sequential([_transform.SimplifyInference(),
                                          _transform.FoldConstant(),
                                          _transform.FoldScaleAxis(),
                                          _transform.CanonicalizeOps(),
                                          _transform.FoldConstant()])

        with relay.build_config(opt_level=4):
            mod = optimize(mod)
            mod = relay.quantize.annotate()(mod)

            # find scale
            cache_file = '%s_%s_scales.pkl' % (VIDEO_FILE, MODEL_NAME)
            if os.path.exists(cache_file):
                print("Using cached layer statistics...")
                with open(cache_file, 'rb') as f:
                    scales = pickle.load(f)
            else:
                print("Compute layer statistics...")
                scales = calibrate_on_dataset(mod['main'], params, input_dtype, input_shape)
                with open(cache_file, 'wb') as f:
                    pickle.dump(scales, f)

            if qeval == 'power2':
                scales = list(map(lambda scale: 2**np.math.ceil(np.math.log(scale, 2)) if scale > 0 else 1.0, scales))
                weight_scales = 'power2'
            elif qeval == 'max':
                weight_scales = 'max'
            else:
                raise ValueError("Invalid quantiziation eval: " + qeval)

            mod['main'] = relay.quantize.calibrate(mod['main'], weight_scales=weight_scales,
                                                   scales=scales)
            mod = relay.quantize.realize()(mod)
            mod = relay.transform.FoldConstant()(mod)

    return mod

###################################################################
# Begin Tuning
# ------------
# Now we can extract tuning tasks from the network and begin tuning.
# Here, we provide a simple utility function to tune a list of tasks.
# This function is just an initial implementation which tunes them in sequential order.
# We will introduce a more sophisticated tuning scheduler in the future.

# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               only_parse,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename=None,
               use_transfer_learning=True,
               try_nchwc=True,
               try_winograd=False,
               try_spatial_pack_depthwise=False):


    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if not use_transfer_learning:
        if os.path.exists(tmp_log_file):
            os.remove(tmp_log_file)
    else:
        # select actual best logs
        if not os.path.exists(tmp_log_file):
            os.mknod(tmp_log_file)
        autotvm.record.pick_best(tmp_log_file, log_filename)

    if os.path.exists(tmp_log_file):
        # sort out best historic entries
        print("Load historic training logs...")
        best_context = autotvm.task.ApplyHistoryBest(autotvm.record.load_from_file(tmp_log_file))
        best_tgtkeys = best_context.best_by_targetkey

    print("Total tasks: %s" % len(tasks))

    if try_nchwc:
        for i in range(len(tasks)):
            # converting conv2d tasks to conv2d_NCHWc tasks
            op_name = tasks[i].workload[0]
            if op_name == 'conv2d':
                func_create = 'topi_x86_conv2d_NCHWc'
            elif op_name == 'depthwise_conv2d_nchw':
                func_create = 'topi_x86_depthwise_conv2d_NCHWc_from_nchw'
            try:  # try NCHWc template
                tsk = autotvm.task.create(func_create, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'direct')
                tsk.workload = tasks[i].workload
                print ( "[Override Task %2d/%2d (%s) with NCHWc] {cfg.space: %i -> %i}" % (i+1, len(tasks), tsk.workload[0], len(tasks[i].config_space), len(tsk.config_space)) )
                tasks[i] = tsk
            except Exception:
                pass


    if try_winograd:
        for i in range(0,len(tasks)):
            try:  # try winograd template
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host, 'winograd')
                tasks.append(tsk)
                print ( "[New Task %2d->%2d (%s) winograd] {cfg.space: %i -> %i}" % (i+1, len(tasks), tsk.workload[0], len(tasks[i].config_space), len(tsk.config_space)) )
            except Exception:
                pass


    # if we want to use spatial pack for depthwise convolution
    if try_spatial_pack_depthwise:
        for i in range(len(tasks)):
            if tasks[i].name == 'topi_nn_depthwise_conv2d_nchw':
                tuner = 'xgb_knob'
                tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
                                          tasks[i].target, tasks[i].target_host,
                                          'contrib_spatial_pack')
                tasks.append(tsk)
                print ( "[New Task %2d->%2d (%s) contrib_spatial_pack] {cfg.space: %i -> %i}" % (i+1, len(tasks), tsk.workload[0], len(tasks[i].config_space), len(tsk.config_space)) )


    for i, tsk in enumerate(tasks):

        tsk_template = ''
        tsk_org_flop = -1;

        device_name = target.device_name if target.device_name else "cpu"

        try:
            # compute best historic entry GFLOPS
            tsk_template = best_tgtkeys[(device_name, tsk.workload)][0][2].template_key
            tsk_org_cost = np.mean( best_tgtkeys[(device_name, tsk.workload)][1].costs )
            tsk_org_flop = tsk.flop / tsk_org_cost / 1e9
        except:
            pass

        if tsk_org_flop == -1:
          org_flop_str = "no history"
        else:
          org_flop_str = "%.2f GFLOPS /%s" % (tsk_org_flop, tsk_template)

        prefix = "[Task %2d/%2d %s|%s] (%s) {%s}" % (i+1, len(tasks), tsk.workload[1][:4], tsk.workload[2][:4], tsk.workload[0], org_flop_str)

        if only_parse:
          print ("%s SKIP tunning" % prefix)
          continue

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'xgb_knob':
            tuner_obj = XGBTuner(tsk, loss_type='rank', feature_type='knob')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=50)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        tuner_obj.tune(n_trial=min(n_trial, len(tsk.config_space)),
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, log_filename)
    if not use_transfer_learning:
        os.remove(tmp_log_file)


######################################################################
# Download required files
# -----------------------
# Download cfg and weights file if first time.
CFG_NAME = MODEL_NAME + '.cfg'
WEIGHTS_NAME = MODEL_NAME + '.weights'
REPO_URL = 'https://github.com/dmlc/web-data/blob/master/darknet/'
CFG_URL = 'https://github.com/pjreddie/darknet/raw/master/cfg/' + CFG_NAME + '?raw=true'
WEIGHTS_URL = 'https://pjreddie.com/media/files/' + WEIGHTS_NAME

cfg_path = download_testdata(CFG_URL, CFG_NAME, module="darknet")
weights_path = download_testdata(WEIGHTS_URL, WEIGHTS_NAME, module="darknet")

# Download and Load darknet library
if sys.platform in ['linux', 'linux2']:
    DARKNET_LIB = 'libdarknet2.0.so'
    DARKNET_URL = REPO_URL + 'lib/' + DARKNET_LIB + '?raw=true'
elif sys.platform == 'darwin':
    DARKNET_LIB = 'libdarknet_mac2.0.so'
    DARKNET_URL = REPO_URL + 'lib_osx/' + DARKNET_LIB + '?raw=true'
else:
    err = "Darknet lib is not supported on {} platform".format(sys.platform)
    raise NotImplementedError(err)

lib_path = download_testdata(DARKNET_URL, DARKNET_LIB, module="darknet")

DARKNET_LIB = __darknetffi__.dlopen(lib_path)
net = DARKNET_LIB.load_network(cfg_path.encode('utf-8'), weights_path.encode('utf-8'), 0)

if not os.path.exists(log_file):
    os.mknod(log_file)


######################################################################
# Import the graph to Relay
# -------------------------
# compile the model
batch_size = 1
input_dtype = 'float32'
input_shape = [batch_size, net.c, net.h, net.w]
print("Converting darknet to relay functions...")
model, params = relay.frontend.from_darknet(net, dtype=input_dtype, shape=input_shape)


######################################################################
# Quantize the model graph
# -------------------------
# compile the model
print("Quantize model...")
model = quantize_model(model, params, input_dtype, input_shape, qeval='power2')

######################################################################
# Tune quantized model graph
# -------------------------
# tune the model
print("Extract tasks...")
tasks = autotvm.task.extract_from_program(model['main'], target=target,
                                          params=params,
                                          ops=(relay.op.nn.conv2d,relay.op.nn.dense))
# False to skip real tunne of layers
# set True for tuning (may take a while)
only_parse=True
print("Tuning...")
# run tuning tasks
tune_tasks(tasks, only_parse=only_parse, **tuning_option)


######################################################################
# Run model on demo video
# -------------------------
# run the model on target
print("Run the model...")
run_model(model, params, input_dtype, input_shape, evaluate=True)
