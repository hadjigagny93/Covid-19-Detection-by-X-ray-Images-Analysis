import numpy as np
import src.settings.base as sg
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
import os
from imutils import paths


class DataPreparation():
    """
    Process date feature.

    Methods
    -------
    do_preprocessing
    """

    max_pixel = 255.0
    inres = (224, 224)
    imagePaths = list(paths.list_images(sg.PATH_DATASET))


    def __init__(self, deploy=False):
        self.data = np.empty([1, 1])
        self.labels = np.empty([1, 1])
        self.deploy = False


    def do_preprocessing(self):
        """ Do the image preprocessing on the selected images.

        Parameters
        ----------

        Returns
        -------
        data: Transform the processed images as a features
        labels: Extract the target features
        """
        data, labels = self.create_data_labels()
        data = np.array(data) / self.max_pixel
        labels = np.array(labels)
        labels = self.label_to_categorical(labels)
        return data, labels

    def image_uniformisation(self, image_path):
        image_treat = cv2.imread(image_path)
        image_treat = cv2.cvtColor(image_treat, cv2.COLOR_BGR2RGB)
        image_treat = cv2.resize(image_treat, self.inres)
        if self.deploy:
            return image_treat / self.max_pixel
        return image_treat

    def create_data_labels(self):
        data, labels = [], []
        for imagePath in self.imagePaths:
            label = imagePath.split(os.path.sep)[-2]
            image = self.image_uniformisation(imagePath)
            data.append(image)
            labels.append(label)
        return data, labels

    def label_to_categorical(self, labels):
        lb = LabelBinarizer()
        labels = lb.fit_transform(labels)
        labels = to_categorical(labels)
        return labels
