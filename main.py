import logging
import argparse

# from datasets import DataLoader
# from data_transformer import DataTransformer
# from models import BoostingModel
# from scorer import ROIChecker

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def main():
    """ Main function"""
    parser = argparse.ArgumentParser(description='main arguments parser')
    parser.add_argument('--mode',
                        type=str,
                        default='fit',
                        help='choose mode: fit or predict (default: fit)'
                        )

    parser.add_argument('--model_path',
                        type=str,
                        default=None,
                        help='Define model path for predict mode (default: None)'
                        )

    parser.add_argument('--test_path',
                        type=str,
                        default=None,
                        help='Define path to test data for predict mode (default: None)'
                        )

    parser.add_argument('--target',
                        type=str,
                        default='result',
                        help='Define target, for example --target="result" or "total" etc. (default: "result")'
                        )

    parser.add_argument('--config_dir',
                        type=str,
                        default='configs/',
                        help='Define configs directory path (default: configs/)')

    print(f"Config dir path is {parser.parse_args().config_dir}")
    print(f"Config dir path is {parser.parse_args().target}")
    print(f"Config dir path is {parser.parse_args().mode}")


if __name__ == '__main__':
    main()
