# main function // fonction principale
from src.application.segmentation import ImageSegmentation
import  src.settings.base as sg 
import os

if __name__ == "__main__":
    image_path = os.path.join(sg.PATH_DATASET, "covid/1-s2.0-S0929664620300449-gr2_lrg-b.jpg")
    myImageSegmentation = ImageSegmentation(image_path=image_path) 
    centroids = 3
    myImageSegmentation.fit_kmeans_clustering(centroids = centroids)
    W = myImageSegmentation.results_dict
    methods = list(W.keys())[-1]
    target_path = os.path.join(sg.PATH_SEG, "BW_{}_{}.png".format(centroids, methods))
    W[methods].save(target_path)
