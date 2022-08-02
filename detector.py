import os
import re
import shutil
from pathlib import Path

import numpy as np
import tensorflow
import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
import tensorflow_hub as hub 
import time

tf.compat.v1.disable_eager_execution()

hub_dict = {'FasterRCNN_ResNet152': "https://tfhub.dev/tensorflow/faster_rcnn/resnet152_v1_640x640/1",
            'FasterRCNN_ResNet50': "https://tfhub.dev/tensorflow/faster_rcnn/resnet50_v1_640x640/1"}

class Detector2():         # with eager_execution for tensorflow debugging purposes
  def __init__(self, input_size=(1,640,640,3), model='FasterRCNN_ResNet50'):
      self.image_tensor = np.random.rand(1,80,160,3).astype(np.uint8)
      self.model = hub.load(hub_dict[model])

  def detect(self, x=None):
      # Need to map it into x eventually
      output = self.model(self.image_tensor)
      return output

class Detector():           # without eager_execution
    def __init__(self, input_shape=(80,160,3), model='FasterRCNN_ResNet152'):
        if len(input_shape) < 3:
            input_shape = (*input_shape, 3)
        self.input_shape = input_shape
        self.input_image = tf.placeholder(shape=(None, *self.input_shape), dtype=np.uint8, name='detection_input_image_placeholder')
        pretrained_model = hub.load(hub_dict[model])
        self.model = pretrained_model(self.input_image)
        # self.model = tf.layers.conv2d(x, filters=32,  kernel_size=4, strides=2, activation=tf.nn.relu, padding="valid", name="conv1")

    def init_session(self, sess=None, init_logging=True):
        if sess is None:
            self.sess = tf.Session()
            self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        else:
            self.sess = sess 
        # if init_logging:
        #     self.train_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "train"), self.sess.graph)
        #     self.val_writer = tf.summary.FileWriter(os.path.join(self.log_dir, "val"), self.sess.graph)

    def detect(self, source_inputs=None):
        #OVERRIDE source_inputs
        # print(type(source_inputs))
        #exit()
        # source_inputs = tf.random.uniform(shape=[1,640,640,3])
        # source_inputs = np.asarray(source_inputs)
        return self.sess.run(self.model, feed_dict={self.input_image: source_inputs})


if __name__ == "__main__":
    source_inputs = np.random.rand(1,80,160,3).astype(np.uint8)
    object_detector = Detector(model='FasterRCNN_ResNet50')  
    object_detector.init_session()  

    while True:
        start = time.time()
        detection_outputs = object_detector.detect(source_inputs)
        # print(detection_outputs)
        elapsed = time.time() - start

        print(elapsed)

  