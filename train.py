import sys
import logging
import os
import argparse

sys.path.append('lib/helper_funcs.py')
from lib.training_helper import train_model
from lib.helper_funcs import yaml_loader
from lib.enums import ArchitectureMode
root_dir = os.getcwd()


def argParser() -> dict:
    """
    Argument parser for vad and separator tools.

    Returns:
        A dictionary of all the positional and optional arguments specified.
    """
    parser = argparse.ArgumentParser(description='Load directory of the tool you want to use')
    parser.add_argument('name',
                        type=str,
                        help='Model name to save'
                        )
    parser.add_argument('data_directory',
                        type=str,
                        help='set dataset directory')
    parser.add_argument('mode',
                        type=ArchitectureMode,
                        help='set mode, available options `SVS`, `VAD`. `DAE`. Each option corresponds to different'
                             ' kind of architecture'
                        )
    parser.add_argument('-e',
                        '--epochs',
                        required=False,
                        metavar='',
                        type=int,
                        help='set epochs')
    parser.add_argument('-b',
                        '--batch_size',
                        required=False,
                        metavar='',
                        type=int,
                        help='set batch size')
    parser.add_argument('-tl',
                        '--transfer_learning',
                        required=False,
                        metavar='',
                        type=bool,
                        help='set True for transfer learning to this model to be applied')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('-q',
                       '--quiet',
                       action='store_true',
                       help='print quiet')
    group.add_argument('-v',
                       '--verbose',
                       action='store_true',
                       help='print verbose')

    args = parser.parse_args()
    return args


def yaml_config(data: dict, args: dict) -> dict:
    """
    Configures a yaml file according the arguments a user specified while running the tool

    Args:
        data: Data of the configuration file
        args: Arguments the user passed
    Returns:
        A dictionary of the configured data
    """
    data['name'] = args.name
    data['data_directory'] = args.data_directory
    data['mode'] = args.mode
    if args.epochs is not None:
        data['epochs'] = args.epochs
    if args.batch_size is not None:
        data['batch_size'] = args.batch_size
    if args.transfer_learning is not None:
        data['transfer_learning'] = args.transfer_learning
    return data


def main():
    args = argParser()
    if args.mode.value == ArchitectureMode.svs.value:
        cf_path = "configuration_files/vocal_separator_tool/init_separator_tool.yaml"
    else:
        cf_path = "configuration_files/vocal_activity_detection_tool/init_vad_tool.yaml"
    separator_tool_params = yaml_loader(filepath=cf_path)

    logger_path = os.path.join(root_dir, 'logger')
    separator_tool_logger_path = os.path.join(logger_path, 'separator_tool')
    logger_file_name = os.path.join(separator_tool_logger_path, f'{args.name}_logger.log')
    if not os.path.isdir(separator_tool_logger_path):
        os.mkdir(separator_tool_logger_path)
    logging.basicConfig(level=logging.INFO, filename=logger_file_name, filemode="w",
                        format="%(asctime)s - %(levelname)s - %(message)s")
    separator_tool_data = yaml_config(data=separator_tool_params, args=args)

    logging.info(f'Tool data:\n {separator_tool_data}')
    train_model(separator_tool_data)


if __name__ == "__main__":
    main()
