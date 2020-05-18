from src.application.model import CovidNet


def main():
    instance = CovidNet.build_network()
    print(type(instance))

if __name__  == "__main__":
    main()
