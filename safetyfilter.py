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

tf.disable_eager_execution()

model = tf.keras.models.load_model('3.h5')

data_point = np.array([[[3.1401231,  -0.13485435]]])

print(model.predict(data_point))



# class SafetyFilter():         # with eager_execution for tensorflow debugging purposes
#     def __init__(self):
#         self.input_shape = (1,1,2)
#         self.input_image = tf.placeholder(shape=self.input_shape, dtype=np.float32, name='xi_and_r')
#         model = tf.keras.models.load_model('3.h5')
#         self.tf_model1 = model(self.input_image)

#     def init_session(self,sess=None, init_logging=True):
#         if sess is None:
#             self.sess = tf.Session()
#             self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
#         else:
#             self.sess = sess

#     def filter_control(self):
#         input = np.array([[[3.1084964,  0.12823616]]])
#         input = input.astype(np.float32)

#         # output = self.tf_model1.run(input)['PathDifferences3_Add'][0,0,0]
#         output = self.sess.run(self.tf_model1, feed_dict={self.input_image: input})

#         return output

#     # def detect(self, source_inputs=None):
#     #     return self.sess.run(self.model, feed_dict={self.input_image: source_inputs})


# if __name__ == "__main__":
#     safety_filter = SafetyFilter()  
#     safety_filter.init_session()  

#     while True:
#         start = time.time()
#         outputs = safety_filter.filter_control()
#         # print(detection_outputs)
#         elapsed = time.time() - start

#         print(outputs)
#         exit()

  