import tensorflow as tf
from detector import Detector

object_detector = Detector(model='FasterRCNN_ResNet50')            # needs proper input size and architecture choice as arguments

