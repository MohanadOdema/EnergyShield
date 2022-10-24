#    This file was created by
#    MATLAB Deep Learning Toolbox Converter for TensorFlow Models.
#    18-Oct-2022 12:05:39

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_model():
    InputLayer = keras.Input(shape=(None,2))
    XiPathIn = layers.Dense(1, name="XiPathIn_")(InputLayer)
    fc_1_XiPath = layers.Dense(10, name="fc_1_XiPath_")(XiPathIn)
    relu_XiPath = layers.ReLU()(fc_1_XiPath)
    fc_2_XiPath = layers.Dense(1, name="fc_2_XiPath_")(relu_XiPath)
    fc_3_XiPath = layers.Dense(1, name="fc_3_XiPath_")(fc_2_XiPath)
    relu_1_XiPath = layers.ReLU()(fc_3_XiPath)
    fc_4_XiPath = layers.Dense(1, name="fc_4_XiPath_")(relu_1_XiPath)
    relu_2_XiPath = layers.ReLU()(fc_4_XiPath)
    XiPathOut = layers.Dense(1, name="XiPathOut_")(relu_2_XiPath)
    NegXiPathIn = layers.Dense(1, name="NegXiPathIn_")(InputLayer)
    fc_1_NegXiPath = layers.Dense(10, name="fc_1_NegXiPath_")(NegXiPathIn)
    relu_NegXiPath = layers.ReLU()(fc_1_NegXiPath)
    fc_2_NegXiPath = layers.Dense(1, name="fc_2_NegXiPath_")(relu_NegXiPath)
    fc_3_NegXiPath = layers.Dense(1, name="fc_3_NegXiPath_")(fc_2_NegXiPath)
    relu_1_NegXiPath = layers.ReLU()(fc_3_NegXiPath)
    fc_4_NegXiPath = layers.Dense(1, name="fc_4_NegXiPath_")(relu_1_NegXiPath)
    relu_2_NegXiPath = layers.ReLU()(fc_4_NegXiPath)
    NegXiPathOut = layers.Dense(1, name="NegXiPathOut_")(relu_2_NegXiPath)
    BetaPathIn = layers.Dense(1, name="BetaPathIn_")(InputLayer)
    ConcatenationLayer = layers.Concatenate(axis=2)([XiPathOut, NegXiPathOut, BetaPathIn])
    PathDifferences1 = layers.Dense(4, name="PathDifferences1_")(ConcatenationLayer)
    relu_Differences1 = layers.ReLU()(PathDifferences1)
    PathDifferences2 = layers.Dense(4, name="PathDifferences2_")(relu_Differences1)
    relu_Differences2 = layers.ReLU()(PathDifferences2)
    PathDifferences3 = layers.Dense(2, name="PathDifferences3_")(relu_Differences2)
    FinalOutputLayer = PathDifferences3

    model = keras.Model(inputs=[InputLayer], outputs=[FinalOutputLayer])
    return model
