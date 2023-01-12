import librosa
import musdb
import numpy as np
import yaml
import os
import sys
from numpy import ndarray

sys.path.append('lib/enums.py')
from lib.enums import Paths, ArchitectureMode


def yaml_loader(filepath: str, loader=yaml.Loader) -> dict:
    """
    Loads a yaml file

    Args:
        filepath: Path to the relevant yaml file
        loader: Instance of Loader class
    Returns:
        A dictionary that contains the configuration data from the yaml file.
    """
    try:
        with open(filepath, "r") as file:
            data = yaml.load(file.read(), Loader=loader) or {}
    except OSError as err:
        msg = "Please ensure the file in " + filepath + " path exists."
        raise Exception(msg)

    return data


def yaml_dumper(added_values: dict, filepath: str) -> None:
    """
    Dumps key-value pair/pairs into a yaml file

    Args:
        added_values: A dictionary with key-value pairs
        filepath: Path to the relevant yaml file
    Returns:
        None
    """
    with open(filepath, "w") as file_descriptor:
        data = yaml.dump(added_values, file_descriptor)


def crop_center(img: ndarray, crop_x: int, crop_y: int) -> ndarray:
    """
    Crops an image/numpy array into into the selected center frames and bins

    Args:
        img: Selected numpy array to edit
        crop_x:  crop numpy array to center, selected width
        crop_y: crop numpy array to center, selected height
    Returns:
        The selected image/numpy array cropped to center
    """
    y, x = img.shape
    start_x = x // 2 - (crop_x // 2)
    start_y = y // 2 - (crop_y // 2)
    return img[start_y:start_y + crop_y, start_x:start_x + crop_x]


def yaml_deconstruct(my_data: dict) -> tuple:
    """
    A function to deconstruct the initial configuration file for the given tool, augmented with the user's arguments.

    Args:
        my_data: Tool configuration file augmented with the user's arguments
    Returns:
        Deconstructed yaml object variables
    """
    model_name = my_data.get('name')
    epochs = my_data.get('epochs')
    batch_size = my_data.get('batch_size')
    frame_size = my_data.get('frame_size')
    hop_length = my_data.get('hop_length')
    win_length = my_data.get('win_length')
    sample_rate = my_data.get('sample_rate')
    iterations = my_data.get('iterations')
    img_width = my_data.get('img_size').get('width')
    img_height = my_data.get('img_size').get('height')
    data_directory = my_data.get('data_directory')
    starting_point = my_data.get('starting_point')
    label_width = my_data.get('label_width')
    mode = str(my_data.get('mode').value) if my_data.get('mode') else None

    return model_name, epochs, batch_size, frame_size, hop_length, win_length, sample_rate, iterations, img_width, \
        img_height, data_directory, starting_point, label_width, mode


def get_chunk_duration_in_seconds(hop_length: int, sample_rate: int, img_width: int) -> float:
    """
    Calculates the chunk duration for the songs in the training data given the user's input or the default values from
    the configuration file
    Args:
        hop_length: Hop length of sampling
        sample_rate: Sample rate of sampling
        img_width: image width of the calculated spectrogram or else time frames of sampling
    Returns:
        Chunk duration, calculated given the user's input or the default values from the configuration file
    """
    duration = (hop_length / sample_rate) * img_width
    duration = round(duration, 2)

    return duration


def get_sliding_window_in_seconds(hop_length: int, sample_rate: int) -> float:
    """
    A function that calculates the sliding window of sampling, given the user's input or the initial configuration file
    values.

    Args:
        sample_rate: sample rate of sampling
        hop_length: hop length of sampling
    Returns:
         The calculated sliding window of sampling rounded for 4 decimals
    """
    sliding_window = (hop_length / sample_rate * 25)
    sliding_window = round(sliding_window, 4)
    return sliding_window


def create_training_data_folder_structure(root_dir: str, main_directory: str, output_dir: str,
                                          mode: ArchitectureMode) -> tuple:
    """
    Creates if needed the folder structure for the augmented training data from musdb18 and the selected model then
    returns the corresponding paths. If directories already exist from another run just return the corresponding paths.

    Args:
        root_dir: root directory of the repo
        main_directory: main directory of the selected model to be trained
        output_dir: selected dataset directory given from the configuration files or the user for model training
            and is an positional argument for data augmentation tool
        mode: mode for pre processing tool
    Returns:
         All the paths created or needed for the execution of the selected model training
    """
    mixture_spectrogram_dir = Paths.MIXTURE_DIR.value
    non_vocals_spectrogram_dir = Paths.NON_VOCALS_DIR.value
    vocals_spectrogram_dir = Paths.VOCALS_DIR.value
    vocals_masks_dir = Paths.VOCAL_MASKS_DIR.value

    main_tool_path = os.path.join(root_dir, main_directory)
    output_path = os.path.join(main_tool_path, output_dir)

    path_specs = os.path.join(output_path, mixture_spectrogram_dir)
    path_non_vocals = os.path.join(output_path, non_vocals_spectrogram_dir)
    path_vocals = os.path.join(output_path, vocals_spectrogram_dir)
    path_masks = os.path.join(output_path, vocals_masks_dir)
    if not os.path.isdir(output_path):
        os.mkdir(output_path)
        os.mkdir(path_specs)
        if mode == ArchitectureMode.svs.value:
            os.mkdir(path_masks)
        elif mode == ArchitectureMode.dae.value:
            os.mkdir(path_vocals)
        else:
            os.mkdir(path_non_vocals)

    return path_specs, path_non_vocals, path_vocals, path_masks, output_path, main_tool_path


def pre_process_training_data(input_signal: ndarray, track: musdb, sr: int, n_fft: int, hop_length: int,
                              win_length: int) -> ndarray:
    """
    Pre-process input sample for training dataset.

    Firstly, we down-sampling from 44100 to 22050 Hz and
    convert stereo signal to mono by averaging left and right channels.
    Then apply STFT algorithm to convert input into a time-frequency representation.

    Args:
        input_signal: loaded audio input signal
        track: musdb object for a specific track
        sr: sample rate of the loaded track
        n_fft: length of the windowed signal after padding with zeros
        hop_length: number of audio samples between adjacent STFT columns
        win_length: window size of each frame of audio
    Returns:
        A numpy array of the input's spectrogram.
    """

    # normalized = input_signal - input_signal.mean() / input_signal.std()
    resample_input = librosa.resample(input_signal, track.rate, sr)
    resample_mono = librosa.to_mono(resample_input)
    s_input = librosa.stft(resample_mono, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window='hann',
                           center=True, pad_mode='constant')

    return s_input


def ibm_compute(vocals_softmask: ndarray, acc_softmask: ndarray) -> ndarray:
    """
        Convert vocal softmask into an Ideal Binary Mask. As a normalization technique introduced by
        Kin Wah Edward Lin, IBM's values set to 0.02, 0.98 instead of 0, 1.

        Args:
            vocals_softmask: 2 dimensional numpy array of the computed vocal softmask, (freq_bins, time_frames)
            acc_softmask: 2 dimensional numpy array of the computed accompaniment softmask, (freq_bins, time_frames)
        Returns:
            ibm_vocals: 2 dimensional numpy array of the Ideal Binary Mask, (freq_bins, time_frames)
    """
    ibm = np.greater_equal(vocals_softmask, acc_softmask)
    ibm_vocals = np.where(ibm, 0.02, 0.98)
    return ibm_vocals


if __name__ == '__main__':
    pass
