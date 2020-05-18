# An abstract class for encapsulate all deep learning 
# models for this preict Corona cases
from os.path import basename, join, splitext
import logging
import src.settings.base as sg
sg.enable_logging(log_filename=f'{splitext(basename(__file__))[0]}.log', logging_level=logging.DEBUG)

import abc
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dropout, Dense, Input, GlobalAveragePooling2D, MaxPooling2D, UpSampling2D, Conv2D, concatenate
from tensorflow.keras import Model, Sequential
from tensorflow.keras.applications.resnet_v2 import ResNet50V2

import gc

class BaseModel(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def _build_network(self):
        pass

class Naive(BaseModel):

    @classmethod
    def build_network(cls):
        logging.info('Use VGG16 model ...')
        baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
        logging.info('Create the next step model ...')
        headModel = baseModel.output
        headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(64, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        headModel = Dense(2, activation="softmax")(headModel)
        model = Model(inputs=baseModel.input, outputs=headModel)
        for layer in baseModel.layers:
            layer.trainable = False
        return model

class CovidNet(BaseModel):
    "covid net model class"
    flatten = True
    num_classes = 2
    checkpoint = ''

    @classmethod
    def build_network(cls):
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        x = base_model.output
        if cls.flatten:
            x = Flatten()(x)
        else:
            x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        predictions = Dense(cls.num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        if len(cls.checkpoint):
            model.load_weights(checkpoint)
        return model

class Unet(BaseModel):

    """
    U-Net architecture
    https://miro.medium.com/max/2824/1*f7YOaE4TWubwaFF7Z1fzNw.png
    https://miro.medium.com/max/2000/1*yzbjioOqZDYbO6yHMVpXVQ.jpeg
    U-Net code inspiration
    https://towardsdatascience.com/u-net-b229b32b4a71
    https://github.com/zhixuhao/unet/blob/master/model.py
    https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47

    Build a U-Net neural network
    Methods
    -------
    build_network
    """

    @classmethod
    def build_network(cls, input_size = (224, 224, 3)):
        logging.info('Use U-Net model ...')
        inputs = Input(input_size)
        nb_filters = 64
        # Contracting phase
        block1, conv1, nb_filters = cls.contracting_block(nb_filters, inputs)
        block2, conv2, nb_filters = cls.contracting_block(nb_filters, block1)
        del block1
        gc.collect()
        block3, conv3, nb_filters = cls.contracting_block(nb_filters, block2)
        del block2
        gc.collect()
        # Center phase
        block4, conv4, nb_filters = cls.center_block(nb_filters, block3)
        del block3
        gc.collect()
        block4 = MaxPooling2D(pool_size=(2, 2))(block4)
        block5, conv5, nb_filters = cls.center_block(nb_filters, block4)
        del block4
        gc.collect()
        # Expensive phase
        block6, nb_filters = cls.expansive_block(nb_filters, block5, conv4)
        del block5
        gc.collect()
        block7, nb_filters = cls.expansive_block(nb_filters, block6, conv3)
        del block6
        gc.collect()
        block8, nb_filters = cls.expansive_block(nb_filters, block7, conv2)
        del block7
        gc.collect()
        block9, nb_filters = cls.expansive_block(nb_filters, block8, conv1)
        del block8
        gc.collect()
        # Final output
        block10 = Conv2D(3, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', name='Conv_interest')(block9)
        del block9
        gc.collect()
        block10 = Flatten(name="flatten")(block10)
        block10 = Dense(2, activation="softmax")(block10)
        headModel = Model(inputs = inputs, outputs = block10)
        return headModel

    @classmethod
    def contracting_block(cls, new_filters, baseModel):
        conv = Conv2D(new_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(baseModel)
        conv = Conv2D(new_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
        headModel = MaxPooling2D(pool_size=(2, 2))(conv)
        nb_filters = int(new_filters * 2)
        return headModel, conv, nb_filters

    @classmethod
    def center_block(cls, new_filters, baseModel):
        conv = Conv2D(new_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(baseModel)
        conv = Conv2D(new_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv)
        headModel = Dropout(0.5)(conv)
        nb_filters = int(new_filters * 2)
        return headModel, conv, nb_filters

    @classmethod
    def expansive_block(cls, new_filters, baseModel, concat_model):
        headModel = UpSampling2D(size = (2,2))(baseModel)
        headModel = Conv2D(new_filters, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(headModel)
        headModel = concatenate([concat_model, headModel], axis = 3)
        headModel = Conv2D(new_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(headModel)
        headModel = Conv2D(new_filters, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(headModel)
        nb_filters = int(new_filters / 2)
        return headModel, nb_filters






