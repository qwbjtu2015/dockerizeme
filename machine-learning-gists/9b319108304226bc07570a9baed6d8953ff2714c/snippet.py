#!/opt/anaconda3/bin/python
# -*- coding:utf-8 -*-
# Created by Enigma on 2016/9/26
"""
Verify the mechanism of gradients update operation during asynchronous training in between-graph approach.
Run:
# ps
/opt/anaconda3/bin/python async_grad_test.py --ps_hosts=localhost:1024 --worker_hosts=localhost:1025,localhost:1026 --job_name=ps --task_index=0
# workers
/opt/anaconda3/bin/python async_grad_test.py --ps_hosts=localhost:1024 --worker_hosts=localhost:1025,localhost:1026 --job_name=worker --task_index=0
/opt/anaconda3/bin/python async_grad_test.py --ps_hosts=localhost:1024 --worker_hosts=localhost:1025,localhost:1026 --job_name=worker --task_index=1
"""
import time
import tensorflow as tf

# Define hyper-parameters
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('learning_rate', 1, 'Initial learning rate.')
tf.app.flags.DEFINE_integer('training_epochs', 2,
                            'Training epochs for every thread')
tf.app.flags.DEFINE_integer('thread_steps', 3, 'Steps run before sync gradients.')
# Define missions parameters
tf.app.flags.DEFINE_string("ps_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", "",
                           "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", "", "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_string("logs_path", "checkpoint/async_grads",
                           "Path to store performance_log")
# Hyper-parameters setting
LEARNING_RATE = FLAGS.learning_rate
TRAINING_EPOCHS = FLAGS.training_epochs
THREAD_STEPS = FLAGS.thread_steps
LOGS_PATH = FLAGS.logs_path
WORKER_NUM = 2


def main(_):
    ps_hosts = FLAGS.ps_hosts.split(",")
    worker_hosts = FLAGS.worker_hosts.split(",")
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})
    # Allow GPU memory grow
    server_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(allow_growth=True),
        log_device_placement=True)
    server = tf.train.Server(cluster, config=server_config,
                             job_name=FLAGS.job_name, task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        with tf.device('/job:ps/task:%d' % FLAGS.task_index):
            server.join()

    elif FLAGS.job_name == "worker":
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % FLAGS.task_index,
                cluster=cluster)) as device:

            global_step = tf.get_variable('global_step', [],
                                          initializer=tf.constant_initializer(0),
                                          trainable=False)
            task_index = FLAGS.task_index
            # Define variable
            with tf.name_scope('input'):
                x = tf.placeholder(tf.float32, name="x")

            with tf.name_scope('weights'):
                target_w = tf.Variable(2.0, name='target_w')
                w_list = [tf.Variable(2.0, name='target_w') for i in range(WORKER_NUM)]
                w = w_list[task_index]

            with tf.name_scope('output'):
                y = tf.mul(x, w, name='y')

            with tf.name_scope('real_output'):
                y_ = tf.placeholder(tf.float32, name="y_")

            # specify optimizer
            with tf.name_scope('train'):
                # optimizer is an "operation" which we can execute in a session
                optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)

            with tf.name_scope('gradient'):
                loss = tf.reduce_mean(tf.square(y_ - y))  # MSE loss
                # gradient_all = optimizer.compute_gradients(loss)  # gradient of network (with NoneType)
                gradient_all = optimizer.compute_gradients(loss)  # gradient of network (with NoneType)
                grads_vars = [v for (g, v) in gradient_all if g is not None]  # all variable that has gradients
                gradient = optimizer.compute_gradients(loss, grads_vars)  # gradient of network (without NoneType)
                grads_holder = [(tf.placeholder(tf.float32, shape=g.get_shape()), v)
                                for (g, v) in gradient]
                train_op = optimizer.apply_gradients(grads_holder, global_step=global_step)

            # create a summary for network gradients
            init_op = tf.initialize_all_variables()
            epoch_init_op = w.assign(target_w)
            w_addup = tf.placeholder(tf.float32)
            epoch_update_op = target_w.assign_add(w_addup)

        sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                                 logdir=LOGS_PATH,
                                 init_op=init_op,
                                 global_step=global_step,
                                 save_model_secs=180)

        with sv.prepare_or_wait_for_session(server.target) as sess:
            # create performance_log writer object (this will performance_log on every machine)
            # perform training cycles
            # time.sleep(sleep_time)

            for epoch in range(TRAINING_EPOCHS):
                _ = sess.run(epoch_init_op)
                init_w = target_w.eval()
                time.sleep(task_index)
                grads = []
                for i in range(THREAD_STEPS):
                    print("task%d - epoch%d: " % (task_index, epoch), end='  ')
                    x_i = i
                    y_real = 10 + i
                    print('x_i: ', x_i, end='. ')
                    y_i = sess.run(y, feed_dict={x: x_i})
                    print('y_i: ', y_i, end='. ')
                    loss_i = sess.run(loss, feed_dict={x: x_i, y_: y_real})
                    print('loss: ', loss_i, end='. ')
                    grad_i = sess.run(gradient, feed_dict={x: x_i, y_: y_real})
                    print('grad: ', grad_i, '.')
                    grads.append(grad_i)
                    time.sleep(0.5)

                    # print("States of w in task%d - thread_step%d: " % (FLAGS.task_index, i), w.eval())
                    # time.sleep(2)

                # calculate total gradients
                grads_sum = {}
                # add up dθ
                for i in range(len(grads_holder)):
                    k = grads_holder[i][0]
                    if k is not None:
                        grads_sum[k] = sum([g[i][0] for g in grads])

                _ = sess.run(train_op, feed_dict=grads_sum)
                print("Final States of w in task%d - epoch%d: " % (task_index, epoch), w.eval())
                w_add = w.eval() - init_w
                _ = sess.run(epoch_update_op, feed_dict={w_addup:w_add})
                print('target_w: ', target_w.eval())
        sv.stop()
        print("done")


if __name__ == "__main__":
    tf.app.run()