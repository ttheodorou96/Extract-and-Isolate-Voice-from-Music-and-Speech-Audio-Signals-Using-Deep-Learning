import sys
import os
import pickle

import musdb
import numpy as np

sys.path.append('lib/helper_funcs.py')
from lib.helper_funcs import crop_center, yaml_deconstruct, get_chunk_duration_in_seconds, \
    get_sliding_window_in_seconds, create_training_data_folder_structure, pre_process_training_data, ibm_compute
from tqdm import tqdm

sys.path.append('lib/enums.py')
from lib.enums import Paths, ArchitectureMode


def preprocess_data(iterations: int, duration: float, sample_rate: int, mode: ArchitectureMode, mix_save_dir: str,
                    non_vocals_save_dir: str, vocals_save_dir: str, masks_save_dir: str, frame_size: int,
                    hop_length: int, starting_point: int, win_length: int, label_width: int):
    """
    Function that loops through the musdb tracks and calls all the corresponding functions to create and save the
    intended images as my_data specified. If no custom dataset is selected the tool downloads a 7sec segment sample from
    musdb18.

    Args:
        iterations: Sets number of chunks each song will be split.
        duration: Sets duration of the chunks, default value is 0.29
        sample_rate: Sets sample rate for resampling, default value is 22050
        mode: Mode of tool, options are `SVS`, `DAE`, `VAD`
        mix_save_dir: Directory of sftf input excerpts
        non_vocals_save_dir: Directory of non vocals stft excerpts
        vocals_save_dir: Directory of vocal stft excerpts
        masks_save_dir: directory of binary masks excerpts
        frame_size: length of the windowed signal after padding with zeros
        hop_length: number of audio samples between adjacent STFT columns
        starting_point: starting point of spectrogram excerpts
        win_length: window size of audio
        label_width: IBM's output label width, default value 9 to create (2049, 9) shape.
    Returns
    -------
        None
    """
    root_dir = os.getcwd()
    db_path = os.path.join(root_dir, 'musdb18')
    mus = musdb.DB(root=db_path, subsets='train')
    samples, itr, set = 0, 0, 1
    vocal_masks_array = []
    stft_input_array = []
    stft_vocals_array = []
    stft_accompaniments_array = []
    sp = 0 if starting_point is None else float(starting_point)
    img_height = int((frame_size / 2) + 1)
    eps = np.finfo(float).eps
    for x in tqdm(range(iterations)):
        for track in mus:
            itr = itr + 1
            print(f'{itr}) {track.name}')
            print(f'total samples: {samples}')
            print(f'set: {set}/{iterations}')
            print(f'{len(mus) - itr} remaining samples for this set...')
            # Initiate starting point for first iteration
            track.chunk_start = sp if x == 0 and sp else track.chunk_start
            samples = samples + 1
            track.chunk_duration = duration
            # Αdd chunk duration for each iteration to create non overlapping excerpts.
            track.chunk_start += track.chunk_duration
            track.chunk_start = round(track.chunk_start, 4)
            # For Random Starting point
            mix_signal = track.audio.T
            vocals_signal = track.targets['vocals'].audio.T
            accompaniment_signal = track.targets['accompaniment'].audio.T

            s_mixture = pre_process_training_data(mix_signal, track, sample_rate, frame_size, hop_length, win_length)
            s_vocals = pre_process_training_data(vocals_signal, track, sample_rate, frame_size, hop_length, win_length)
            s_accompaniments = pre_process_training_data(accompaniment_signal, track, sample_rate, frame_size,
                                                         hop_length, win_length)

            if mode == ArchitectureMode.svs.value:
                # Only crop center frame for label output of training data -> shape (img_height, 1) CNN_separator
                if label_width == 1:
                    s_vocals = crop_center(s_vocals, 1, img_height)
                    s_accompaniments = crop_center(s_accompaniments, 1, img_height)
                model = eps + np.abs(s_vocals) + np.abs(s_accompaniments)
                vocals_mask = np.divide(np.abs(s_vocals), model)
                acc_mask = np.divide(np.abs(s_accompaniments), model)
                ibm = ibm_compute(vocals_mask, acc_mask)
                vocal_masks_array.append(ibm)
                stft_input_array.append(s_mixture)
            elif mode == ArchitectureMode.dae.value:
                stft_vocals_array.append(s_vocals)
            else:
                stft_input_array.append(s_mixture)
                stft_accompaniments_array.append(s_accompaniments)
        stft_input_array = np.array(stft_input_array)
        if mode == ArchitectureMode.svs.value:
            vocal_masks_array = np.array(vocal_masks_array)
            with open(masks_save_dir + f"\\set{x + 1}_y_train.pickle", 'wb') as handle:
                pickle.dump(vocal_masks_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(mix_save_dir + f"\\set{x + 1}_X_train.pickle", 'wb') as handle:
                pickle.dump(stft_input_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
        elif mode == ArchitectureMode.dae.value:
            stft_vocals_array = np.array(stft_vocals_array)
            with open(vocals_save_dir + f"\\set{x + 1}_X_train_y_train.pickle", 'wb') as handle:
                pickle.dump(stft_vocals_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(non_vocals_save_dir + f"\\set{x + 1}_X_train.pickle", 'wb') as handle:
                pickle.dump(stft_accompaniments_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with open(mix_save_dir + f"\\set{x + 1}_X_train.pickle", 'wb') as handle:
                pickle.dump(stft_input_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
        vocal_masks_array = []
        stft_input_array = []
        stft_vocals_array = []
        itr = 0
        set += 1

    print(f"Save training data Completed!")
    print(f"Number of total samples saved : {samples}")
    print(f'Number of sets saved: {iterations}')
    return


def load_params(my_data: dict):
    """
    Function that initiates the preprocess pipeline by user's input or by default initial values. Creates the folder
    structure for the augmented training data to be saved and saves the numpy arrays with pickle by calling
    corresponding functions.

    Args:
        my_data: A dictionary that contains data from the loaded configuration file and the argument parser
    Returns:
        None
    """
    root_dir = os.getcwd()
    _, _, _, frame_size, hop_length, win_length, sample_rate, iterations, img_width, _, output_directory, \
        starting_point, label_width, mode = yaml_deconstruct(my_data)
    duration = get_chunk_duration_in_seconds(hop_length, sample_rate, img_width)
    if mode == ArchitectureMode.dae.value:
        main_directory = Paths.AE.value
    elif mode == ArchitectureMode.svs.value:
        main_directory = Paths.SEPARATOR_ROOT_DIR.value
    else:
        main_directory = Paths.VAD_ROOT_DIR.value
    if not os.path.isdir(main_directory):
        os.mkdir(main_directory)

    mix_save_dir, non_vocals_save_dir, vocals_save_dir, masks_save_dir, output_path, _ = \
        create_training_data_folder_structure(root_dir, main_directory, output_directory, mode)

    preprocess_data(iterations=iterations,
                    duration=duration,
                    sample_rate=sample_rate,
                    mode=mode,
                    mix_save_dir=mix_save_dir,
                    non_vocals_save_dir=non_vocals_save_dir,
                    vocals_save_dir=vocals_save_dir,
                    masks_save_dir=masks_save_dir,
                    frame_size=frame_size,
                    hop_length=hop_length,
                    starting_point=starting_point,
                    win_length=win_length,
                    label_width=label_width
                    )
    return


if __name__ == '__main__':
    pass
