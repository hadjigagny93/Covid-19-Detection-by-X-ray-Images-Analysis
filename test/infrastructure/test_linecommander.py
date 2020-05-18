from src.infrastructure.line_commander import TrainCommandLineParser

def main():
    args = TrainCommandLineParser().do_parse_args()
    print(args)
    print(args['dataset'])

if __name__ == "__main__":
    main()