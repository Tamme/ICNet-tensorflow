"""
This code is based on DrSleep's framework: https://github.com/DrSleep/tensorflow-deeplab-resnet 
"""
from __future__ import print_function

import argparse
import os, sys, time
from tqdm import trange

import tensorflow as tf
import numpy as np

from model import ICNet_BN
from tools import decode_labels, prepare_label, preprocess
from image_reader import ImageReader, read_labeled_image_list

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)

# If you want to apply to other datasets, change following four lines
DATA_DIR = '/home/lauri/lane_detection/Semantic-Segmentation-Suite/Forest'
#DATA_DIR = '/home/lauri/lane_detection/ADEChallengeData2016'
DATA_LIST_PATH = './list/forest_train_list.txt'
VAL_DATA_LIST_PATH = './list/forest_val_list.txt'
#DATA_LIST_PATH = './list/ade20k_train_list.txt' 
IGNORE_LABEL = 255 # The class number of background
#IGNORE_LABEL = 0 # The class number of background
INPUT_SIZE = '480, 870' # Input size for training
#INPUT_SIZE = '500, 500' # Input size for training

BATCH_SIZE = 10 
LEARNING_RATE = 0.001 #1e-3
MOMENTUM = 0.9
#NUM_CLASSES = 5
NUM_CLASSES = 5
NUM_STEPS = 60000
NUM_VAL_STEPS = 132
POWER = 0.9
RANDOM_SEED = 1235
WEIGHT_DECAY = 0.0001
PRETRAINED_MODEL = './model/icnet_cityscapes_trainval_90k_bnnomerge.npy'
SNAPSHOT_DIR = './snapshots_forest'
SAVE_NUM_IMAGES = 4
SAVE_PRED_EVERY = 100

# Loss Function = LAMBDA1 * sub4_loss + LAMBDA2 * sub24_loss + LAMBDA3 * sub124_loss
LAMBDA1 = 0.16
LAMBDA2 = 0.4
LAMBDA3 = 1.0

param = {'name': 'forest',
    'input_size': [480, 870],
    'num_classes': 5,
    'ignore_label': 255,
    'num_steps': NUM_VAL_STEPS,
    'data_dir': DATA_DIR, 
    'data_list': VAL_DATA_LIST_PATH}

def get_arguments():
    parser = argparse.ArgumentParser(description="ICNet")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-val-steps", type=int, default=NUM_VAL_STEPS,
        help="Number of validation steps.")
    parser.add_argument("--random-mirror", action="store_true", default=False,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true", default=False,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--restore-from", type=str, default=PRETRAINED_MODEL,
                        help="Where restore model parameters from.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--update-mean-var", action="store_true", default=False,
                        help="whether to get update_op from tf.Graphic_Keys")
    parser.add_argument("--train-beta-gamma", action="store_true", default=False,
                        help="whether to train beta & gamma in bn layer")
    parser.add_argument("--filter-scale", type=int, default=1,
                        help="1 for using pruned model, while 2 for using non-pruned model.",
                        choices=[1, 2])
    return parser.parse_args()

def save(saver, sess, logdir, step):
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def get_mask(gt, num_classes, ignore_label):
    less_equal_class = tf.less_equal(gt, num_classes-1)
    not_equal_ignore = tf.not_equal(gt, ignore_label)
    mask = tf.logical_and(less_equal_class, not_equal_ignore)
    indices = tf.squeeze(tf.where(mask), 1)

    return indices

def _debug_print_func(pred_val):
    print('->>>>>>>>>>>>>>>>PRED : {}'.format(pred_val.shape))
    return False

def create_loss(output, label, num_classes, ignore_label):
    
    raw_output_tmp = tf.image.resize_bilinear(output, size=(480, 870), align_corners=True)
    raw_output_tmp = tf.argmax(raw_output_tmp, axis=3)
    raw_pred_tmp = tf.expand_dims(raw_output_tmp, dim=3)
    pred_tmp = tf.reshape(raw_pred_tmp, [-1,])

    #debug_print_op = tf.py_func(_debug_print_func, [pred_tmp], [tf.bool])
    #with tf.control_dependencies(debug_print_op):
    #    out = tf.identity(out, name='out')

    #mask = tf.not_equal(raw_gt, param['ignore_label'])
    #indices = tf.squeeze(tf.where(mask), 1)
    #gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    #pred = tf.gather(pred_flatten, indices)


    raw_pred = tf.reshape(output, [-1, num_classes])
    label = prepare_label(label, tf.stack(output.get_shape()[1:3]), num_classes=num_classes, one_hot=False)
    label = tf.reshape(label, [-1,])

    indices = get_mask(label, num_classes, ignore_label)
    gt = tf.cast(tf.gather(label, indices), tf.int32)
    pred = tf.gather(raw_pred, indices)

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=gt)
    reduced_loss = tf.reduce_mean(loss)

    return reduced_loss, pred_tmp

def prepare_evaluation_input(image_filename, anno_filename):
    img = tf.image.decode_image(tf.read_file(image_filename), channels=3)
    img.set_shape([None, None, 3])
    anno = tf.image.decode_image(tf.read_file(anno_filename), channels=1)
    anno.set_shape([None, None, 1])

    ori_shape = tf.shape(img)
    return preprocess(img, param), anno, ori_shape

def main():
    """Create the model and start the training."""
    args = get_arguments()
    
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    
    coord = tf.train.Coordinator()
    
    is_evaluation = tf.placeholder_with_default(False, shape=[], name="is_evaluation")

    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            DATA_DIR,
            DATA_LIST_PATH,
            input_size,
            args.random_scale,
            args.random_mirror,
            args.ignore_label,
            IMG_MEAN,
            coord)
        image_batch, label_batch = reader.dequeue(args.batch_size)
    
    #Evaluation section

    #img = preprocess(img, param)
    #image_batch = preprocess(img, param)
    #image_batch = tf.cond(tf.equal(is_evaluation, True), lambda: preprocess(img, param), lambda: image_batch)

    #Rewrite input depending on if evaluation or training
    image_filename, anno_filename = tuple(tf.cond(
        tf.equal(is_evaluation, True),
        lambda: (tf.placeholder(dtype=tf.string), tf.placeholder(dtype=tf.string)),
        lambda: ("", "")
    ))
    image_batch, anno, ori_shape = tuple(tf.cond(
        tf.equal(is_evaluation, True),
        lambda: prepare_evaluation_input(image_filename, anno_filename),
        #lambda: prepare_evaluation_input(image_filename, anno_filename)
        lambda: (image_batch, tf.cast(0, tf.uint8), tf.cast(0, tf.int32))
    ))

    #model = model_config[args.model]
    #net = model({'data': img}, num_classes=param['num_classes'], 
    #                filter_scale=args.filter_scale, evaluation=True)
    net = ICNet_BN({'data': image_batch}, is_training=True, num_classes=args.num_classes, filter_scale=args.filter_scale)
    sub4_out = net.layers['sub4_out']
    sub24_out = net.layers['sub24_out']
    sub124_out = net.layers['conv6_cls']
    
    # Predictions.
    def do_eval_predictions(net):
        raw_output = net.layers['conv6_cls'] #Tensor("conv6_cls/BiasAdd:0", shape=(1, 119, 119, 150), dtype=float32)

        #raw_output_up = tf.image.resize_bilinear(raw_output, size=ori_shape[:2], align_corners=True)
        raw_output_up = tf.image.resize_bilinear(raw_output, size=(436, 800), align_corners=True)
        raw_output_up = tf.argmax(raw_output_up, axis=3)
        raw_pred = tf.expand_dims(raw_output_up, dim=3)

        # mIoU
        pred_flatten = tf.reshape(raw_pred, [-1,])
        raw_gt = tf.reshape(anno, [-1,])

        mask = tf.not_equal(raw_gt, param['ignore_label'])
        indices = tf.squeeze(tf.where(mask), 1)
        gt_e = tf.cast(tf.gather(raw_gt, indices), tf.int32)
        pred_e = tf.gather(pred_flatten, indices)

        if param["name"] == 'ade20k':
            pred = tf.add(pred, tf.constant(1, dtype=tf.int64))
            mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred_e, gt_e, num_classes=param['num_classes']+1)
        elif param["name"] == 'cityscapes':
            mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred_e, gt_e, num_classes=param['num_classes'])
        elif param["name"] == 'forest':
            mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred_e, gt_e, num_classes=5)
            #acc = tf.metrics.accuracy(labels=gt, predictions=pred)
        return (mIoU, update_op)

    mIoU, update_op = tuple(tf.cond(
        tf.equal(is_evaluation, True), 
        lambda: do_eval_predictions(net), 
        lambda: (tf.cast(0.0, tf.float32),tf.cast(0.0, tf.float64))
    ))

    #restore_var = [v for v in tf.global_variables() if 'conv6_cls' not in v.name]
    restore_var = [v for v in tf.global_variables()]
    
    all_trainable = [v for v in tf.trainable_variables() if ('beta' not in v.name and 'gamma' not in v.name) or args.train_beta_gamma]
    
    loss_sub4, _ = create_loss(sub4_out, label_batch, args.num_classes, args.ignore_label)
    loss_sub24, _ = create_loss(sub24_out, label_batch, args.num_classes, args.ignore_label)
    loss_sub124, pred = create_loss(sub124_out, label_batch, args.num_classes, args.ignore_label)
    l2_losses = [args.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    
    reduced_loss = LAMBDA1 * loss_sub4 +  LAMBDA2 * loss_sub24 + LAMBDA3 * loss_sub124 + tf.add_n(l2_losses)

    # Using Poly learning rate policy 
    base_lr = tf.constant(args.learning_rate)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / args.num_steps), args.power))
    
    # Gets moving_mean and moving_variance update operations from tf.GraphKeys.UPDATE_OPS
    if args.update_mean_var == False:
        update_ops = None
    else:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        opt_conv = tf.train.MomentumOptimizer(learning_rate, args.momentum)
        grads = tf.gradients(reduced_loss, all_trainable)
        train_op = opt_conv.apply_gradients(zip(grads, all_trainable))
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    local_init = tf.local_variables_initializer()
    sess.run(local_init)

    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=5)

    ckpt = tf.train.get_checkpoint_state(args.snapshot_dir)
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('Restore from pre-trained model...')
        net.load(args.restore_from, sess, NUM_CLASSES)

    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        
        feed_dict = {step_ph: step}
        if step % args.save_pred_every == 0 and step != 0:
            loss_value, loss1, loss2, loss3, _ = sess.run([reduced_loss, loss_sub4, loss_sub24, loss_sub124, train_op], feed_dict=feed_dict)
            save(saver, sess, args.snapshot_dir, step)
        else:
            loss_value, loss1, loss2, loss3, _ = sess.run([reduced_loss, loss_sub4, loss_sub24, loss_sub124, train_op], feed_dict=feed_dict)
        duration = time.time() - start_time
        if step % 10 == 0:
            print('step {:d} \t total loss = {:.3f}, sub4 = {:.3f}, sub24 = {:.3f}, sub124 = {:.3f} ({:.3f} sec/step)'.format(step, loss_value, loss1, loss2, loss3, duration))

        if (step +1) % 300 == 0:
            #validation
            img_files, anno_files = read_labeled_image_list(DATA_DIR, VAL_DATA_LIST_PATH)

            for i in trange(args.num_val_steps, desc='evaluation', leave=True):

                feed_dict = {image_filename: img_files[i], anno_filename: anno_files[i], is_evaluation: True}
                _, ori_shape_o = sess.run([update_op, ori_shape], feed_dict=feed_dict)
            print('mIoU: {}'.format(sess.run(mIoU, feed_dict={is_evaluation: True})))

    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
