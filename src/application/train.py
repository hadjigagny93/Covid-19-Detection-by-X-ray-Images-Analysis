from src.application.pipeline import Training_pipeline, Testing_pipeline
from src.application.line_commander import TestCommandLineParser, TrainCommandLineParser


if __name__ == "__main__":
    args_train = TrainCommandLineParser()
    args_train = args_train.do_parse_args()
    train_pipeline = Training_pipeline(model = args_train['model'], see_opt = args_train['intelligibility'], plot = args_train['plot'])
    train_pipeline.train()
