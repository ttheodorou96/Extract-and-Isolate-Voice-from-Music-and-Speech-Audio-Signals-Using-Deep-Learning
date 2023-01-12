import logging
from lib.custom_musdb_dataset_create import load_params
from lib.helper_funcs import yaml_loader
from lib.enums import ArchitectureMode
import argparse
import os


def argParser() -> dict:
    """
    Argument parser for the Custom musdb dataset.

    Returns:
        args: A dictionary of all the positional and optional arguments specified.
    """
    parser = argparse.ArgumentParser(description='Preprocessing data tool. Creates training data from MUSDB18 database')
    parser.add_argument('data_directory',
                        type=str,
                        help='set name of output directory'
                        )
    parser.add_argument('iterations',
                        type=int,
                        help='set number of iterations for the preprocessing algorithm'
                        )
    parser.add_argument('mode',
                        type=ArchitectureMode,
                        help='set mode, available options `SVS`, `VAD`. `DAE`. Each option corresponds to different'
                             ' kind of training data created'
                        )
    parser.add_argument('-f',
                        '--frame_size',
                        required=False,
                        metavar='',
                        type=int,
                        help='set frame size')
    parser.add_argument('-H',
                        '--hop_length',
                        required=False,
                        metavar='',
                        type=int,
                        help='set hop length')
    parser.add_argument('-w',
                        '--win_length',
                        required=False,
                        metavar='',
                        type=int,
                        help='set window size of each frame of audio')
    parser.add_argument('-sr',
                        '--sample_rate',
                        required=False,
                        metavar='',
                        type=int,
                        help='set sample rate')
    parser.add_argument('-iw',
                        '--width',
                        required=False,
                        metavar='',
                        type=int,
                        help='set width for input image size - feature')
    parser.add_argument('-sp',
                        '--starting_point',
                        required=False,
                        metavar='',
                        type=int,
                        help='set starting point for spectrogram excerpts, if None default is zero')
    parser.add_argument('-ow',
                        '--label_width',
                        required=False,
                        metavar='',
                        default=9,
                        type=int,
                        help='set width for output image size - label')

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


def yaml_config(data: dict, arguments: dict) -> dict:
    """
    Configures a yaml file according the arguments a user specified while running the tool

    Args:
        data : Data of the configuration file
        arguments : Arguments the user passes
    Returns:
        data: A dictionary of the configured data
    """
    data['data_directory'] = arguments.data_directory
    data['iterations'] = arguments.iterations
    data['mode'] = arguments.mode
    if arguments.frame_size is not None:
        data['frame_size'] = arguments.frame_size
    if arguments.hop_length is not None:
        data['hop_length'] = arguments.hop_length
    if arguments.win_length is not None:
        data['win_length'] = arguments.win_length
    if arguments.sample_rate is not None:
        data['sample_rate'] = arguments.sample_rate
    if arguments.width is not None:
        data['img_size']['width'] = arguments.width
    if arguments.starting_point is not None:
        data['starting_point'] = arguments.starting_point
    if arguments.label_width is not None:
        data['label_width'] = arguments.label_width
    return data


def main():
    cf_path = "configuration_files/vocal_separator_tool/init_separator_tool.yaml"
    custom_db_tool_params = yaml_loader(filepath=cf_path)
    try:
        args = argParser()
    except SystemExit:
        print('Argument error please fill out all positional arguments and check tool usage for help.')
    else:
        print(args.data_directory)

        root_dir = os.getcwd()
        logger_path = os.path.join(root_dir, 'logger')
        logger_file_name = os.path.join(logger_path, args.data_directory + '_preprocess_tool_logger.log')
        logging.basicConfig(level=logging.INFO, filename=logger_file_name, filemode="w",
                            format="%(asctime)s - %(levelname)s - %(message)s")

        custom_db_data = yaml_config(data=custom_db_tool_params, arguments=args)
        logging.info(f'\nInput Arguments: {custom_db_data}')
        load_params(custom_db_data)


if __name__ == "__main__":
    main()
