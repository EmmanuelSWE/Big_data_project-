import tensorflow as tf 
import numpy as np 

# const input shape and output shape
MODEL_INPUT = (224,224,3)
MODEL_OUTPUT = 7
# making the first model using ResNet50
class ModelA:
    def __init__(self):
        # baseline model
        baseModel = tf.keras.applications.ResNet50(
            weights = "imagenet",
            include_top = False,
            input_shape = MODEL_INPUT
        )

        # freeze model 
        baseModel.trainable = False 

        inputImage = tf.keras.Input(shape = MODEL_INPUT)

        # the layers
        x = baseModel(inputImage,training = False)

        #average pooling 
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        output = tf.keras.layers.Dense(MODEL_OUTPUT, activation="softmax")(x)

        # final Model

        finalModel = tf.keras.Model(inputs= inputImage, output= output)

        finalModel.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print(finalModel.summary())

        self.model = finalModel


