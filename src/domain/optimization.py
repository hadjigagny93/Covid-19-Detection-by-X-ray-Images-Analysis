"""
Module to transform date.

Classes
-------
Intelligibility

Inspiration
-------
https://github.com/sicara/tf-explain/tree/master/examples/core
https://tf-explain.readthedocs.io/en/latest/methods.html#activations-visualization
https://pypi.org/project/tf-explain/#vanilla-gradients
https://www.quantmetry.com/intelligibilite-deep-learning-image/
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array

from tf_explain.core.activations import ExtractActivations
from tf_explain.core.vanilla_gradients import VanillaGradients
from tf_explain.core.gradients_inputs import GradientsInputs
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
from tf_explain.core.grad_cam import GradCAM
from tf_explain.core.smoothgrad import SmoothGrad
from tf_explain.core.integrated_gradients import IntegratedGradients

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

import src.settings.base as sg 
import os 

class Intelligibility:

    def __init__(self, setX, setY, model, target_layers):
        self.setX = setX
        self.setY = setY
        self.model = model
        self.labels = []
        self.target_layers = target_layers

    def save_intelligibility(self):
        logging.info("Begin intelligibility...")
        self.unique_label()
        for index, value in enumerate(self.setX):
            X = self.Process(value)
            self.vanilla_gradients(X, index)
            self.grad_cam(X, index)
            self.occlusion_sensitivity(X, index)
            self.integrated_gradients(X, index)
            self.gradients_input(X, index)
            self.extract_activations(X, index)
        logging.info("Intelligibility done...")

    def Process(self, value):
        X = img_to_array(value)
        X = np.array([value], None)
        return X

    def unique_label(self):
        unique_label = len(self.setY[0])
        for label in range(unique_label):
            self.labels.append(int(label))

    def vanilla_gradients(self, value, index):
        for classe in self.labels:
            vanilla_grad = VanillaGradients()
            grid = vanilla_grad.explain((value, self.setY[index]), self.model, class_index=classe)
            name = "n_{}_".format(index) + str(classe) + "_vanilla_grad.png"
            path = sg.PATH_VANILLA + name
            vanilla_grad.save(grid, ".", path)

    def grad_cam(self, value, index):
        for classe in self.labels:
            grad_cam = GradCAM()
            grid = grad_cam.explain((value, self.setY[index]), self.model, class_index=classe, layer_name=self.target_layers)
            name = "n_{}_".format(index) + str(classe) + "_grad_cam.png"
            grad_cam.save(grid, ".", sg.PATH_GRAD + name)

    def occlusion_sensitivity(self, value, index):
        for classe in self.labels:
            occlusion_sensitivity = OcclusionSensitivity()
            grid = occlusion_sensitivity.explain((value, self.setY[index]), self.model, class_index=classe, patch_size=20)
            name = "n_{}_".format(index) + str(classe) + "_occlusion_sensitivity.png"
            occlusion_sensitivity.save(grid, ".", sg.PATH_OCCLUSION + name)

    def integrated_gradients(self, value, index):
        for classe in self.labels:
            integrated_gradients = IntegratedGradients()
            grid = integrated_gradients.explain((value, self.setY[index]), self.model, class_index=classe, n_steps=10)
            name = "n_{}_".format(index) + str(classe) + "_integrated_gradients.png"
            integrated_gradients.save(grid, ".", sg.PATH_INTEGRATED + name)

    def gradients_input(self, value, index):
        for classe in self.labels:
            gradients_input = GradientsInputs()
            grid = gradients_input.explain((value, self.setY[index]), self.model, class_index=classe)
            name = "n_{}_".format(index) + str(classe) + "_gradients_input.png"
            gradients_input.save(grid, ".",  sg.PATH_GRADIENTS + name)
    
    def extract_activations(self, value, index):
        extract_activations = ExtractActivations()
        grid = extract_activations.explain((value, self.setY[index]), self.model, self.target_layers)
        name = "n_{}_gradients_input.png".format(index)
        extract_activations.save(grid, ".", sg.PATH_ACTIVATION + name)
