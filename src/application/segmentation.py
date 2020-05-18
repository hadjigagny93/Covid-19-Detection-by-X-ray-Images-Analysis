import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import PIL
import os

class ImageSegmentation:
    """class for image  segmentation with Kmeans algorithms
    
    attributes
    -----------
    image_RGB: color format image (Red - Green- Blue )
    image_GS_LA: grayscale  image (Ndim = 2)
    image_GS_L: grayscale image (Ndim = 1)
    image_BW: same function like the previous
    w: image width
    h: height of the image
    p: deep of the image
    c_pixel: image matrix representation in 3D for colored image
    gs_pixel: image matrix representation  2D
    bw_pixel: image matrix representation 1D
    data_dict: dict of vectors that will be used for performing three parallel Kmeans algorithms
    results_dict: dict of result of clustering algo defined previously

    methods
    -------
    Kmeans_clustering: sklearn implemented methods"""

    def __init__(self, image_path):
        """image extension must be jpeg"""
        self.image_RGB   = Image.open(image_path)
        self.image_GS_LA = self.image_RGB.convert("LA")
        self.image_BW = self.image_RGB.convert("1")
        self.w, self.h, self.p = np.asarray(self.image_RGB).shape
        self.c_pixel  = np.asarray(self.image_RGB).reshape(self.w * self.h, self.p)
        self.gs_pixel = np.asarray(self.image_GS_LA).reshape(self.w * self.h, self.p - 1)
        self.bw_pixel = np.asarray(self.image_BW).flatten().reshape(-1, 1)
        self.results_dict = {}
        self.data_dict = {"c": self.c_pixel,
                          "gs": self.gs_pixel,
                          "bw": self.bw_pixel
                         }

    def __str__(self):
        return """
        color image infos: {}\n
        grayscale LA image infos: {}\n
        binarize image infos:
        """.format(
            np.asarray(self.image_RGB).shape,
            np.asarray(self.image_GS_LA).shape,
            np.asarray(self.image_BW).shape
            )

    def fit_kmeans_clustering(self, centroids=3):
        """
        perform Kmenas clustering algorithm for all images format
        """
        _ = "c"
        kmeans = KMeans(n_clusters = centroids, random_state=0).fit(self.data_dict[_])
        result = kmeans.labels_.reshape(self.w, self.h)
        self.results_dict[_] = self.segmentation_color(result)


    @staticmethod
    def segmentation_color(matrix):

        def dict_color(matrix):
            keys = list(set(matrix.flatten()))
            n_cluster = len(keys)
            values = [[np.random.randint(0,256), np.random.randint(0,256), np.random.randint(0,256)] for _ in range(n_cluster)] 
            color_dict = dict(zip(keys, values))
            return color_dict

        dict_ = dict_color(matrix)
        w, h = matrix.shape
        X = np.zeros((w, h, 3), dtype= np.uint8)
        for i in range(w):
            for j in range(h):
                X[i, j] = dict_[matrix[i, j]]
        return Image.fromarray(X)



