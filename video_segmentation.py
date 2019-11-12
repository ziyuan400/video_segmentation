"""
Detects Cars in an image using KittiSeg.

Input: Image
Output: Image (with Cars plotted in Green)

Utilizes: Trained KittiSeg weights. If no logdir is given,
pretrained weights will be downloaded and used.

Usage:
python demo.py --input_image data/demo.png [--output_image output_image]
                [--logdir /path/to/weights] [--gpus 0]

--------------------------------------------------------------------------------

The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann

Details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import logging
import os
import sys

import collections

# configure logging

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import cv2
import skvideo.io
import numpy as np
import scipy as scp
import scipy.misc
import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

sys.path.insert(1, 'incl')

from seg_utils import seg_utils as seg

try:
    # Check whether setup was done correctly

    import tensorvision.utils as tv_utils
    import tensorvision.core as core
except ImportError:
    # You forgot to initialize submodules
    logging.error("Could not import the submodules.")
    logging.error("Please execute:"
                  "'git submodule update --init --recursive'")
    exit(1)


flags.DEFINE_string('logdir', None,
                    'Path to logdir.')
flags.DEFINE_string('input_image', None,
                    'Image to apply KittiSeg.')
flags.DEFINE_string('output_image', None,
                    'Image to apply KittiSeg.')


default_run = 'KittiSeg_pretrained'
weights_url = ("ftp://mi.eng.cam.ac.uk/"
               "pub/mttt2/models/KittiSeg_pretrained.zip")


def maybe_download_and_extract(runs_dir):
    logdir = os.path.join(runs_dir, default_run)

    if os.path.exists(logdir):
        # weights are downloaded. Nothing to do
        return
      
    if not os.path.exists(runs_dir):
        os.makedirs(runs_dir)
    download_name = tv_utils.download(weights_url, runs_dir)
    logging.info("Extracting KittiSeg_pretrained.zip")

    import zipfile
    zipfile.ZipFile(download_name, 'r').extractall(runs_dir)

    return


def resize_label_image(image, gt_image, image_height, image_width):
    image = scp.misc.imresize(image, size=(image_height, image_width),
                              interp='cubic')
    shape = gt_image.shape
    gt_image = scp.misc.imresize(gt_image, size=(image_height, image_width),
                                 interp='nearest')

    return image, gt_image


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
 #   config.gpu_options.per_process_gpu_memory_fraction = 0.7

    tv_utils.set_gpus_to_use()

    if FLAGS.input_image is None:
        logging.error("No input_image was given.")
        logging.info(
            "Usage: python demo.py --input_image data/test.png "
            "[--output_image output_image] [--logdir /path/to/weights] "
            "[--gpus GPUs_to_use] ")

    if FLAGS.logdir is None:
        # Download and use weights from the MultiNet Paper
        if 'TV_DIR_RUNS' in os.environ:
            runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                    'KittiSeg')
        else:
            runs_dir = 'RUNS'
        maybe_download_and_extract(runs_dir)
        logdir = os.path.join(runs_dir, default_run)
    else:
        logging.info("Using weights found in {}".format(FLAGS.logdir))
        logdir = FLAGS.logdir

    # Loading hyperparameters from logdir
    hypes = tv_utils.load_hypes_from_logdir(logdir, base_path='hypes')

    logging.info("Hypes loaded successfully.")

    # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
    modules = tv_utils.load_modules_from_logdir(logdir)
    logging.info("Modules loaded successfully. Starting to build tf graph.")
    num_classes = 5

    # Create tf graph and build module.
    with tf.Graph().as_default():




        # Create placeholder for input
        image_pl = tf.placeholder(tf.float32)
        image = tf.expand_dims(image_pl, 0)

        # build Tensorflow graph using the model from logdir
        prediction = core.build_inference_graph(hypes, modules,
                                                image=image)

        logging.info("Graph build successfully.")

        # Create a session for running Ops on the Graph.
        sess = tf.Session(config=config)
        saver = tf.train.Saver()

        # Load weights from logdir
        core.load_weights(logdir, sess, saver)

        logging.info("Weights loaded successfully.")

    input_image = FLAGS.input_image
    logging.info("Starting inference using {} as input".format(input_image))

    #Dealing with video segmentation with frame by frame
    #TODO: build a commandline
    loaded_video = skvideo.io.vread('t1.avi')[:100]
    writer = skvideo.io.FFmpegWriter("outputvideo.avi")
    for image in loaded_video:
        # Load and resize input image
        if hypes['jitter']['reseize_image']:
                # Resize input only, if specified in hypes
            image_height = hypes['jitter']['image_height']
            image_width = hypes['jitter']['image_width']
            image = scp.misc.imresize(image, size=(image_height, image_width),
                                                                  interp='cubic')

        # Run KittiSeg model on image
        feed = {image_pl: image}
        softmax = prediction['softmax']
        output = sess.run(softmax, feed_dict=feed)

        print(len(output), type(output), output.shape)
        # Reshape output from flat vector to 2D Image
        output = np.transpose(output)
        shape = image.shape
        output = output.reshape(num_classes, shape[0], shape[1])
        output_image = output[0].reshape(shape[0], shape[1])
       
        # Plot confidences as red-blue overlay
        rb_image = seg.make_overlay(image, output_image)

        # Accept all pixel with conf >= 0.5 as positive prediction
        # This creates a `hard` prediction result for class street
        threshold = 0.5
        street_prediction = output_image > threshold
        street_predictions = output > threshold
        # Plot the hard prediction as green overlay
        green_image = tv_utils.fast_overlay(image, street_prediction)

        green_images = []
        rb_images = []
        output_images = []
        for c in range(0,5):          
            green_images.append(tv_utils.fast_overlay(image, street_predictions[c]))
            rb_images.append(seg.make_overlay(image, output[c]))
            output_images.append(output[c].reshape(shape[0], shape[1]))

        # Save output images to disk.
        if FLAGS.output_image is None:
            output_base_name = input_image
        else:
            output_base_name = FLAGS.output_image

        #Name and save the the red blue segmentation for first 5 frames as png to debug.
        green_image_names = []
        rb_image_names = []
        raw_image_names = []
        for c in range(1,6):
            green_image_names.append(output_base_name.split('.')[0] + str(c) + '_green.png')
            rb_image_names.append(output_base_name.split('.')[0] + str(c) + '_rb.png')
            raw_image_names.append(output_base_name.split('.')[0] + str(c) + '_raw.png')
  
        for c in range(0,5):
            print(green_image_names[c], green_images[c].shape)
            scp.misc.imsave(raw_image_names[c], output_images[c])
            scp.misc.imsave(rb_image_names[c], rb_images[c])
            scp.misc.imsave(green_image_names[c], green_images[c])

        #Output the green masked video as a file and show it to screen
        writer.writeFrame(green_images[4])
        cv2.imshow('frame', green_images[4])

        #user can press p to quit during processing.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    writer.close()
def video():
    cap = skvideo.io.vread('t1.avi')
    for frame in cap[:3]:
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




if __name__ == '__main__':
    tf.app.run()
