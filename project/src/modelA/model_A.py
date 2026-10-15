import tensorflow as tf 
import numpy as np 

# const input shape and output shape
MODEL_INPUT = (224,224,3) # 224x224 RGB images
MODEL_OUTPUT = 4 # 4 species label
# making the first model using ResNet50
class ModelA:
    '''
        This class represents the ResNet50 model belonging to the ResNet family architecture.
        It is a pretrained model consisting of 50 layers and was trained on the ImageNet dataset.  
    '''
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

        finalModel = tf.keras.Model(inputs= inputImage, outputs= output)

        finalModel.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        print(finalModel.summary())

        self.model = finalModel


