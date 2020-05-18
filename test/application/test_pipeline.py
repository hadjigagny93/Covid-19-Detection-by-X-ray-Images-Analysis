from src.application.pipeline import Training_pipeline, Testing_pipeline
import src.settings.base as sg 

def main():


    print("train naive")
    inst = Training_pipeline(model="Naive")
    inst.train()
    print(inst.info)

    print("test naive")
    inst = Testing_pipeline(image_path=sg.PATH_IMAGE, model="Naive")
    inst.predict()

    print("train covid")
    inst = Training_pipeline(model="CovidNet")
    inst.train()
    print(inst.info)

    print("test covid")
    inst = Testing_pipeline(image_path=sg.PATH_IMAGE, model="CovidNet")
    inst.predict()



if __name__ == "__main__":
    main()