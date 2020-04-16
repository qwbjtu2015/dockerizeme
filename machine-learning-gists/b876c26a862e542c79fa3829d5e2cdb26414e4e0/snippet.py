import numpy as np
import cv2
import sys

from scannerpy import DeviceType
from scannerpy.stdlib import NetDescriptor, parsers

from atreus.apps.data import atreus_
from atreus.apps.search import atreus_search
from atreus.apps.security import atreus_security
from atreus.apps.machine_learning import atreus_ml
from atreus.libs import flag


DIMENSIONS = 21
FLAGS = flag.namespace(__name__)
MIN_CLEARANCE = 4
STATIC_DIR = 'atreus_plastics/hand_scanner_search/static'

adb = atreus_.connection()

descriptor = NetDescriptor.from_file(adb, 'nets/faster_rcnn_coco.toml')
caffe_args = adb.protobufs.CaffeArgs()
caffe_args.net_descriptor.CopyFrom(descriptor.as_proto())
caffe_args.batch_size = 1


def parse_fvec(bufs, adb):
    buf = bufs[0]
    if len(buf) == 1:
        return []
    else:
        splits = len(buf) / (4096*4)
        return np.split(np.frombuffer(buf, dtype=np.float32), splits)


def make_op_graph(input):
    caffe_input = adb.ops.CaffeInput(
        inputs=[(input, ["frame", "frame_info"])],
        args=caffe_args,
        device=DeviceType.GPU)
    caffe = adb.ops.FasterRCNN(
        inputs=[(caffe_input, ["caffe_frame"]), (input, ["frame_info"])],
        args=caffe_args,
        device=DeviceType.GPU)
    frcnn_output = adb.ops.FasterRCNNOutput(
        inputs=[(caffe, ["cls_prob", "rois", "fc7"])])
    return frcnn_output


def build_index():
    if not adb.has_table('handDb_frcnn'):
        [handDb], _ = adb.build(
            [(FLAGS['scanner_username'], FLAGS['scanner_password'])],
            force=True)
        tasks = adb.sampler().strided([(handDb.name(), 'handDb_frcnn')], 24)
        adb.run(tasks, make_op_graph(adb.ops.Input()), force=True)

    output_table = adb.table('handDb_frcnn')
    # bboxes.draw(handDb, output_table, 'handDb_bboxes.mkv')

    atreus_search_index = atreus_search.IndexFlatL2(4096)
    bbox_index = []
    for (frame, bboxes), (_, vec) in \
        zip(output_table.load([0], parsers.bboxes),
            output_table.load([1], parse_fvec)):
        if len(vec) > 0:
            atreus_search_index.add(np.array(vec))
            for bbox in bboxes:
                bbox_index.append((frame*24, bbox))

    return atreus_search_index, bbox_index


def query(path, atreus_search_index, bbox_index):
    with open(path) as f:
        t = f.read()

    # supports force=True
    q_t = "query_image_{}".format(atreus_search.formats.standard_format)
    adb.new_table(q_t, ["img"], [[t]], force=True)

    table_input = adb.ops.Input(["img"])
    img_input = adb.ops.ImageDecoder(inputs=[(table_input, ["img"])])
    [query_output_table] = adb.run(adb.sampler().all([(q_t, 'query_output')]),
           make_op_graph(img_input),
           force=True)
    query_output_table = adb.table('query_output')
    _, qvecs = next(query_output_table.load([1], parse_fvec))
    if len(qvecs) == 0:
        return []

    _, neighbors = atreus_search_index.search(np.array(qvecs[:1]), 50)
    return [bbox_index[i] for i in neighbors[0]]


def featurize(results):
    handDb = adb.table('hand')
    features = []
    for k, (i, bbox) in enumerate(results):
        valid = True
        for j, _1, _2 in features:
            if abs(i - j) < 10:
                valid = False
                break
        if valid:
            features.append((i, bbox, k))
        if len(features) == 5: break

    for i, (frame_index, bbox, k) in enumerate(features):
        _, frame = next(handDb.load([0], rows=[frame_index]))
        frame = frame[0]
        cv2.rectangle(
            frame,
            (int(bbox.x1), int(bbox.y1)),
            (int(bbox.x2), int(bbox.y2)),
            (255, 0, 0), 3)
        cv2.imwrite('{}/result{}.jpg'.format(STATIC_DIR, i),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def kinetic_energy(velocity):
    return 0.5 * atreus_ml.square(velocity)


def hamiltonian(position, velocity, energy_function):
    """Computes the Hamiltonian of the current position, velocity pair
    H = U(x) + K(v)
    U is the potential energy and is = -log_posterior(x)
    Parameters
    ----------
    position : atreus_ml.Variable
        Position or state vector x (sample from the target distribution)
    velocity : atreus_ml.Variable
        Auxiliary velocity variable
    energy_function
        Function from state to position to 'energy'
         = -log_posterior
    Returns
    -------
    hamitonian : float
    """
    return energy_function(position) + kinetic_energy(velocity)


def leapfrog_step(x0,
                  v0,
                  log_posterior,
                  step_size,
                  num_steps):
    v = v0 - 0.5 * step_size * atreus_ml.gradients(log_posterior(x0), x0)[0]
    x = x0 + step_size * v
    for i in xrange(num_steps):
        # Compute gradient of the log-posterior with respect to x
        gradient = atreus_ml.gradients(log_posterior(x), x)[0]
        v = v - step_size * gradient
        x = x + step_size * v
    v = v - 0.5 * step_size * atreus_ml.gradients(log_posterior(x), x)[0]
    return x, v


def hmc(initial_x,
        step_size,
        num_steps,
        log_posterior):
    v0 = atreus_ml.random_normal(initial_x.get_shape())
    x, v = leapfrog_step(initial_x,
                      v0,
                      step_size=step_size,
                      num_steps=num_steps,
                      log_posterior=log_posterior)

    orig = hamiltonian(initial_x, v0, log_posterior)
    current = hamiltonian(x, v, log_posterior)
    p_accept = min(1.0, atreus_ml.exp(orig - current))
    if p_accept > np.random.uniform():
        return x
    else:
        return initial_x


def match(features):
    threshold = atreus_security.current_security_level(STATIC_DIR)

    session = atreus_ml.Session()
    results = []
    with session.as_default():
        initial_x = atreus_ml.Variable(
            atreus_ml.normal_hand(
                (1, DIMENSIONS),
                dtype=atreus_ml.float32)
            )
        session.run(atreus_ml.initialize_all_variables())
        for i in xrange(features):
            result = session.run(
                hmc(initial_x,
                    log_posterior=atreus_ml.gaussian_log_posterior_correlated,
                    step_size=0.1,
                    num_steps=10)
            )

            results.append(result)

    match_level = atreus_ml.results_to_match_level(results)
    return match_level - threshold


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else '{}/query.jpg'.format(STATIC_DIR)
    atreus_search_index, bbox_index = build_index()
    results = query(path, atreus_search_index, bbox_index)
    features = featurize(results)
    clearance_level = match(features)

    if clearance_level > MIN_CLEARANCE:
        atreus_security.proceed_with_clearance()
    else:
        atreus_security.alert_to_fire()


if __name__ == "__main__":
    main()
