import os
import logging
# Path of the train and the case of prediction
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
LOGS_DIR = os.path.join(REPO_DIR, 'logs')
PATH_DATASET = os.path.join(REPO_DIR, 'data/undersampling')
PATH_PRED = os.path.join(REPO_DIR, 'data/predict') 
PATH_MODEL = os.path.join(REPO_DIR, 'data/model')

# Logging
def enable_logging(log_filename, logging_level=logging.DEBUG):
    """Set loggings parameters.

    Parameters
    ----------
    log_filename: str
    logging_level: logging.level

    """
    with open(os.path.join(LOGS_DIR, log_filename), 'a') as file:
        file.write('\n')
        file.write('\n')

    LOGGING_FORMAT = '[%(asctime)s][%(levelname)s][%(module)s] - %(message)s'
    LOGGING_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

    logging.basicConfig(
        format=LOGGING_FORMAT,
        datefmt=LOGGING_DATE_FORMAT,
        level=logging_level,
        filename=os.path.join(LOGS_DIR, log_filename)
    )

# Intelligibility
PATH_VANILLA = os.path.join(REPO_DIR, 'src/interface/intelligibility/vanilla_gradient/')
PATH_GRAD = os.path.join(REPO_DIR, 'src/interface/intelligibility/grad_cam/')
PATH_OCCLUSION = os.path.join(REPO_DIR, 'src/interface/intelligibility/occlusion_sensitivity/')
PATH_INTEGRATED = os.path.join(REPO_DIR, 'src/interface/intelligibility/integrated_gradients/')
PATH_GRADIENTS = os.path.join(REPO_DIR, 'src/interface/intelligibility/gradients_input/')
PATH_ACTIVATION = os.path.join(REPO_DIR, 'src/interface/intelligibility/extract_activations/')
PATH_GRAPH = os.path.join(REPO_DIR, 'src/interface/intelligibility/graph/graph.png')
PATH_IMAGE = os.path.join(REPO_DIR, 'test/image/COVID.jpeg')

# segmentation 
PATH_SEG = os.path.join(REPO_DIR, "src/interface/segmentation")


