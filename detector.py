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

def mask_detection_boxes(detections, min_score_threshold=0.5):
    scores = np.squeeze(detections['detection_scores'])
    boxes = np.squeeze(detections['detection_boxes'])
    masked_detections = []
    for i in range(boxes.shape[0]):
        if scores[i] > min_score_threshold:
            masked_detections.append(boxes[i])
        else:
            masked_detections.append(np.zeros(4))      # ymin, xmin, ymax, xmax
    masked_detections = np.asarray(masked_detections, dtype=np.float32)
    assert masked_detections.shape == boxes.shape
    return masked_detections

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

    def init_session(self, sess=None, init_logging=True):
        if sess is None:
            self.sess = tf.Session()
            self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        else:
            self.sess = sess 

    def detect(self, source_inputs=None):
        return self.sess.run(self.model, feed_dict={self.input_image: source_inputs})


if __name__ == "__main__":
    source_inputs = np.random.rand(1,80,160,3).astype(np.uint8)
    object_detector = Detector(model='FasterRCNN_ResNet50')  
    object_detector.init_session()  

    while True:
        start = time.time()
        detection_outputs = object_detector.detect(source_inputs)
        elapsed = time.time() - start

        print(elapsed)

  