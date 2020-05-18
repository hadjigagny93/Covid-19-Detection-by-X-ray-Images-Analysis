

from os.path import basename, join, splitext
import logging
import src.settings.base as sg
sg.enable_logging(log_filename=f'{splitext(basename(__file__))[0]}.log', logging_level=logging.DEBUG)


from imutils import paths
import numpy as np
import argparse
import cv2
import os

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D, Flatten, Dropout, Dense, Input
from tensorflow.keras import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import src.settings.base as sg
from src.domain.optimization import Intelligibility
from src.application.model import Naive, CovidNet
from src.infrastructure.data_preparation import DataPreparation


class Training_pipeline:

        def __init__(self, model, learning_rate=1e-3, nb_epoch=1, batch_size=8, test_size=.2, see_opt=False, plot=False):

            self.model = model
            self.learning_rate = learning_rate
            self.nb_epoch = nb_epoch
            self.batch_size = batch_size
            self.info = {}
            self.test_size = test_size
            self.see_opt = see_opt
            self.plot = plot

        @staticmethod
        def optimizers_analysis(features, target, model, target_layer):
            intelligibility = Intelligibility(features, target, model, target_layer)
            intelligibility.save_intelligibility()

        @staticmethod
        def save_model(model, network="Naive"):
            logging.info("saving COVID-19 detector model...")
            file_name = network + ".model"
            if network=="Naive":
                model.save(os.path.join(sg.PATH_MODEL, file_name), save_format="h5")
                return
            model.save(os.path.join(sg.PATH_MODEL, file_name))

        @staticmethod
        def classification_metrics(true, predict):
            mat = confusion_matrix(true, predict)
            total = sum(sum(mat))
            acc = (mat[0, 0] + mat[1, 1]) / total
            sensitivity = mat[0, 0] / (mat[0, 0] + mat[0, 1])
            specificity = mat[1, 1] / (mat[1, 0] + mat[1, 1])
            return {
                "accuracy": acc,
                "sensitivity": sensitivity,
                "specificity": specificity,}

        def  get_model(self):
            if self.model == Naive:
                return Naive.build_network()
            else:
                return CovidNet.build_network()

        def graph_performance(self):
            N = self.nb_epoch
            plt.style.use("ggplot")
            plt.figure()
            plt.plot(np.arange(1, N+1), self.history.history["loss"], label="train_loss")
            plt.plot(np.arange(1, N+1), self.history.history["val_loss"], label="val_loss")
            plt.plot(np.arange(1, N+1), self.history.history["accuracy"], label="train_acc")
            plt.plot(np.arange(1, N+1), self.history.history["val_accuracy"], label="val_acc")
            plt.title("Training Loss and Accuracy on COVID-19 Dataset")
            plt.xlabel("Number of epoch")
            plt.ylabel("Loss/Accuracy")
            plt.legend(loc="lower left")
            plt.savefig(sg.PATH_GRAPH)

        def train(self):
            data, labels = DataPreparation().do_preprocessing()
            trainX, testX, trainY, testY = train_test_split(data, labels, test_size=self.test_size, stratify=labels, random_state=42)
            logging.info('Compiling model ...')
            opt = Adam(lr = self.learning_rate, decay = self.learning_rate / self.nb_epoch)
            model = self.get_model()
            model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
            logging.info('Training head ...')
            trainAug = ImageDataGenerator(rotation_range=15, fill_mode="nearest")
            model.fit_generator(
                trainAug.flow(trainX, trainY, batch_size=self.batch_size),
                steps_per_epoch=len(trainX) // self.batch_size,
                validation_data=(testX, testY),
                validation_steps=len(testX) // self.batch_size,
                epochs=self.nb_epoch)
            self.save_model(model, network=self.model)
            logging.info('Evaluating network ...')
            predIdxs = model.predict(testX, batch_size=self.batch_size)
            predIdxs = np.argmax(predIdxs, axis=1)
            self.info = {
                **self.info,
                **self.classification_metrics(testY.argmax(axis=1), predIdxs)
                }
            if self.see_opt:
                self.optimizers_analysis(testX[0:3], testY[0:3], model, "block5_conv3")
            if self.plot:
                self.graph_performance()
class Testing_pipeline:

    def __init__(self, image_path, model):
        self.image_path = image_path
        self.model = model

    def load_model(self):
        filename = self.model + ".model"
        logging.info("loading COVID-19 detector model...")
        model = load_model(os.path.join(sg.PATH_MODEL, filename))
        return model

    def predict(self):
        """return the predicted class -- covid or not"""
        processed_image = DataPreparation(deploy=True).image_uniformisation(image_path=self.image_path)
        processed_image = processed_image.reshape((1, 224, 224, 3))
        model = self.load_model()
        model.compile(
            loss='binary_crossentropy',
            optimizer='Adam',
            metrics=['accuracy'])
        return np.argmax(model.predict(processed_image), axis=1)
