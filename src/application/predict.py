from src.application.pipeline import Testing_pipeline
from src.application.line_commander import TestCommandLineParser


if __name__ == "__main__":
    args_test = TestCommandLineParser()
    args_test = args_test.do_parse_args()
    test_pipeline = Testing_pipeline(image_path = args_test['image'], model = args_test['model'])
    print(test_pipeline.predict())
