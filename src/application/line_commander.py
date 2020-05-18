import argparse
import src.settings.base as sg

class TrainCommandLineParser():

    def __init__(self):
        """Initialize class."""
        self.parser = argparse.ArgumentParser()

    def _add_arguments(self):
        """Add arguments to the parser."""
        self.parser.add_argument(
            "-mo",
            "--model",
            type = str,
            default = "Naive",
	        help = "Path to save and output the model"
        )
        self.parser.add_argument(
            "-p", 
            "--plot", 
            type = bool, 
            default = False,
	        help = "True or False if you can use matplotlib under your IDE"
        )
        self.parser.add_argument(
            "-i", 
            "--intelligibility", 
            type = bool, 
            default = False,
	        help = "True or False if you want to see the performance od the model"
        )

    def do_parse_args(self):
        """Parse train command line arguments.

        Returns
        -------
        self.parser.parse_args(): Namespace
        """
        self._add_arguments()
        args = vars(self.parser.parse_args())
        return args


class TestCommandLineParser():

    def __init__(self):
        """Initialize class."""
        self.parser = argparse.ArgumentParser()

    def _add_arguments(self):
        """Add arguments to the parser."""
        self.parser.add_argument(
            "-i", 
            "--image", 
            type = str, 
            default = sg.PATH_IMAGE,
	        help = "Path of the image to test"
        )
        self.parser.add_argument(
            "-mo", 
            "--model", 
            type = str, 
            default = "Naive",
	        help = "Path to save and output the model"
        )

    def do_parse_args(self):
        """Parse test command line arguments.

        Returns
        -------
        self.parser.parse_args(): Namespace
        """
        self._add_arguments()
        args = vars(self.parser.parse_args())
        return args

