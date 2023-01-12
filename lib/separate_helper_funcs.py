import warnings

import musdb
import numpy as np
import logging
import librosa
import librosa.display
import os
import soundfile as sf
from musdb.audio_classes import Track
from numpy import newaxis, ndarray
import tensorflow as tf
import seaborn as sns
import museval
from pysndfx import AudioEffectsChain
from tqdm import tqdm
from lib.helper_funcs import yaml_loader, yaml_deconstruct
from scipy.signal import wiener
import pandas as pd
import matplotlib.pyplot as plt

cf_path = "./configuration_files/vocal_separator_tool/init_separator_tool.yaml"
pre_process_params = yaml_loader(filepath=cf_path)
_, _, _, frame_size, hop_length, win_length, sample_rate, iterations, img_width, \
           img_height, data_directory, _, _, mode = yaml_deconstruct(pre_process_params)


def save_plot_box(path_array: list, selected_song_dir: str):
    """
    Function that filters from the csv loaded file that contains all the scores for each second of the separated song
    only the SDR metric for the vocals and the accompaniments. Then plots and saves the scores for each method into a
    boxplot
    Args:
        path_array: path of the saved csv file
        selected_song_dir: name of the selected song directory
    Returns:
         None
    """
    appended_vocals = []
    appended_acc = []
    for p in path_array:
        df = pd.read_csv(p)
        _df_vocals = df[df['metric'].str.contains('SDR') & df['target'].str.contains('vocals')]
        appended_vocals.append(_df_vocals)
        _df_accompaniment = df[df['metric'].str.contains('SDR') & df['target'].str.contains('accompaniment')]
        appended_acc.append(_df_accompaniment)
    appended_vocals = pd.concat(appended_vocals)
    appended_acc = pd.concat(appended_acc)
    sns.set(rc={'figure.figsize': (20, 6)})
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    boxplot = sns.boxplot(x='score', y='method', data=appended_vocals, palette='colorblind', width=0.3, ax=ax1)
    ax1.set(title='Vocals/SDR (dB) comparison')
    sns.boxplot(x='score', y='method', data=appended_acc, palette='colorblind', width=0.3, ax=ax2)
    ax2.set(title='Accompaniment/SDR (dB) comparison')
    fig.tight_layout()
    boxplot.plot()
    img_save = os.path.join(selected_song_dir, 'plot_box_tools_SDR.png')
    plt.savefig(img_save)
    return


def librosa_decomposition(input_mix: ndarray, sr: int) -> tuple:
    """
    Function that performs vocal separation with the method REPET_SIM introduced by Rafii and Pardo, 2012
    https://librosa.org/doc/main/auto_examples/plot_vocal_separation.html?highlight=vocal
    Added the harmonic percussive separation technique to remove residue percussive sounds. Proven to improve the score
    in a metric SDR scale

    Args:
        input_mix: numpy array of the STFT of the input signal
        sr: Sample rate of input's sampling
    Returns:
        The separated vocals, vocal mask, accompaniments, phase, harmonic part
    """
    input_signal, phase = librosa.magphase(input_mix)
    S_filter = librosa.decompose.nn_filter(input_signal, aggregate=np.median, metric='cosine',
                                           width=int(librosa.time_to_frames(2, sr=sr)))
    S_filter = np.minimum(input_signal, S_filter)
    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (input_signal - S_filter),
                                   power=power)
    mask_v = librosa.util.softmask(input_signal - S_filter,
                                   margin_v * S_filter,
                                   power=power)

    S_vocals = mask_v * input_signal
    S_background = mask_i * input_signal

    return S_vocals, S_background, phase, input_signal, mask_v


def convert_to_stereo(input_sample: ndarray) -> ndarray:
    """
    Function that get a numpy array as input and converts it to STEREO by doubling its channels

    Args:
        input_sample: a numpy array of the input signal
    Returns:
         A numpy array converted to STEREO
    """
    return np.array([input_sample, input_sample])


def mask_compute(S_vocals: ndarray, S_accompaniment: ndarray) -> ndarray:
    """
    Function that computes the vocal mask given we have its vocal and accompaniment components

    Args:
        S_vocals: STFT of the vocal component
        S_accompaniment: STFT of the accompaniment component
    Returns:
         A numpy array of the computes vocal mask
    """
    eps = np.finfo(float).eps
    model = eps + np.abs(S_vocals) + np.abs(S_accompaniment)
    mask_vocals = np.divide(np.abs(S_vocals), model)

    return mask_vocals


def evaluate_SDR_format(S_input: ndarray, sr: int, track_sample: int, phase: ndarray = None):
    """
        Returns the desired np array shape for the SDR evaluation of the audio which is
        (n_components, audio_time_series, n_channels). The input format is a stft np array
        (n_freq_bins, n_time_frames).
    """
    if phase is not None:
        i_input = librosa.istft(S_input * phase, hop_length=hop_length, win_length=win_length)
    else:
        i_input = librosa.istft(S_input, hop_length=hop_length, win_length=win_length)
    resample = librosa.resample(i_input, sr, track_sample)
    y = convert_to_stereo(resample)
    data = y.T.astype('float64')
    return data


def save_my_estimate_scores(vocals_save_path: str, accompaniment_save_path: str, track: Track, my_estimates_dir: str,
                            sr: int, method: list):
    vocals, _ = librosa.load(accompaniment_save_path, sr=sr * 2)
    accompaniment, _ = librosa.load(vocals_save_path, sr=sr * 2)
    vocals_stereo = convert_to_stereo(vocals)
    accompaniment_stereo = convert_to_stereo(accompaniment)
    estimates_json = {
        'vocals': vocals_stereo.T,
        'accompaniment': accompaniment_stereo.T
    }
    track_scores_my_model = museval.eval_mus_track(
        track=track, user_estimates=estimates_json, output_dir=my_estimates_dir
    )
    results_my_model = museval.EvalStore()
    method = [method]
    track_scores = [track_scores_my_model]
    scores_dir = os.path.join(my_estimates_dir, 'scores')

    if not os.path.isdir(scores_dir):
        os.mkdir(scores_dir)
    for name, score in zip(method, track_scores):
        results_my_model.add_track(score.df)
        estimate_csv = f'{scores_dir}/{name}.csv'
        results_my_model.save(f'{scores_dir}/{name}.PANDAS')
        methods = museval.MethodStore()
        methods.add_evalstore(results_my_model, name=name)
        methods.df.to_csv(rf'{estimate_csv}')
        comparison = methods.agg_frames_tracks_scores()
        print(comparison)
        logging.info(f'Results: \n{comparison}')
    print('Scores Saved!')
    return estimate_csv


def save_audio_as_wav(stem: ndarray, label: str, song_name: str, stems_dir: str) -> str:
    """
    Saves stems in wav format in corresponding directories.

    Args:
        stem: numpy array of the stem to be saved
        label: label of stem
        song_name: name of the song
        stems_dir: path of the selected stem to be saved
    Return:
        The path of each stem that gets saved
    """
    song_save_path = os.path.join(stems_dir, f'{song_name}_{label}.wav')
    sf.write(file=song_save_path, data=stem.T, samplerate=44100)

    return song_save_path


def save_stems(stem: ndarray, label: str, sr: int, song_name: str = None, stems_dir: str = None, phase: ndarray = None,
               noise_redux: bool = None) -> str:
    """
    Dynamic save stem function

    Args:
        stem: numpy array of the stem to be saved
        label: label of stem
        song_name: name of the song
        sr: Sample rate of input's sampling
        stems_dir: path of the selected stem to be saved
        phase: Phase of the input's magnitude
        noise_redux: is True when function is called from the noise reduction tool
    Returns:
        The path of each stem that gets saved
    """
    if phase is not None or noise_redux:
        estimate = librosa.istft(np.squeeze(stem) * phase, hop_length=256)
    else:
        estimate = librosa.istft(np.squeeze(stem), hop_length=256, win_length=win_length)

    estimate = librosa.resample(estimate, sr, 44100)
    resample_stereo = convert_to_stereo(estimate)
    if noise_redux:
        output_dir = os.path.join(stems_dir, f'{label}_output.wav')
    else:
        song_save_path = os.path.join(stems_dir,
                                      f'{song_name}_{label}.wav') if stems_dir is not None else f'{song_name}_{label}.wav'

    output_dir = output_dir if noise_redux else song_save_path
    sf.write(file=output_dir, data=resample_stereo.T, samplerate=44100)

    return output_dir


def save_images(mix: ndarray, masks: ndarray, vocals: ndarray, song_name: str, images_dir: str = None,
                accompaniments: ndarray = None, noise_redux: bool = None):
    """
    Download images of spectrograms to corresponding directory

    Args:
        mix: numpy array of the mixture of the song
        masks: numpy array of the vocal mask
        vocals: numpy array of the separated vocals
        song_name: name of the song
        images_dir: path of the stems to be saved
        accompaniments: a numpy array with the STFT of the accompaniments component
        noise_redux: is True when function is called from the noise reduction tool
    Returns:
        None
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(mix), ref=np.max),
                             y_axis='log', x_axis='time', ax=ax1)
    ax1.set(title='Original audio' if noise_redux is True else 'Mixture')

    librosa.display.specshow(librosa.amplitude_to_db(np.abs(masks), ref=np.max), y_axis='log', x_axis='time', ax=ax2)
    ax2.set(title='Vocal Mask')

    librosa.display.specshow(librosa.amplitude_to_db(np.abs(vocals), ref=np.max), y_axis='log', x_axis='time', ax=ax3)
    ax3.set(title='DeNoise Output' if noise_redux is True else 'Separated Vocals')
    if accompaniments is not None:
        librosa.display.specshow(librosa.amplitude_to_db(accompaniments, ref=np.max), y_axis='log', x_axis='time',
                                 ax=ax4)
    else:
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(mix - vocals), ref=np.max), y_axis='log',
                                 x_axis='time', ax=ax4)
    ax4.set(title='Separated Accompaniments')

    fig.tight_layout()
    image_save = os.path.join(images_dir, f'{song_name}.jpg') if images_dir is not None else f'{song_name}.jpg'
    plt.savefig(image_save)
    return


def save_image_input_song(mixture: ndarray, vocals: ndarray, accompaniment: ndarray, input_name: str,
                          output_dir: str) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(mixture), ref=np.max),
                             y_axis='log', x_axis='time', ax=ax1)
    ax1.set(title='Original Audio')
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(vocals), ref=np.max), y_axis='log', x_axis='time', ax=ax2)
    ax2.set(title='Extracted Vocals')
    librosa.display.specshow(librosa.amplitude_to_db(np.abs(accompaniment), ref=np.max), y_axis='log', x_axis='time',
                             ax=ax3)
    ax3.set(title='Extracted Background')
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    image_save = os.path.join(output_dir, f'{input_name}.jpg')
    plt.savefig(image_save)
    return


def create_folder_structure(selected_song_dir: str, tool_name: str, input_song_flag: bool = None) -> tuple:
    """
    Function that create the folder structure for the vocal extraction tool

    Args:
        selected_song_dir: selected song name, root directory for the nested directories that contain the extracted
            features
        tool_name: name of the selected method of separation
        input_song_flag: flag is true if this is a folder structure for an input song given bu the user
    Returns:
         Directories of the audio, image and .csv saved features
    """
    selected_tool_dir = os.path.join(selected_song_dir, tool_name)
    if not os.path.isdir(selected_tool_dir):
        os.mkdir(selected_tool_dir)

    stems_dir = os.path.join(selected_tool_dir, 'stems')
    if not os.path.isdir(stems_dir):
        os.mkdir(stems_dir)

    images_dir = os.path.join(selected_tool_dir, 'images')
    if not os.path.isdir(images_dir):
        os.mkdir(images_dir)
    if input_song_flag:
        estimates_dir = None
    else:
        estimates_dir = os.path.join(selected_tool_dir, 'estimates')
        if not os.path.isdir(estimates_dir):
            os.mkdir(estimates_dir)

    return stems_dir, images_dir, estimates_dir, selected_tool_dir


def crop_center(img: ndarray, crop_x: int, crop_y: int) -> ndarray:
    """
    Crops an image/numpy array into into the selected size

    Args:
        img: Selected numpy array to edit
        crop_x: crop image to center, width
        crop_y: icrop image to center, height
    Returns:
        The selected image/numpy array cropped to center
    """
    y, x = img.shape
    start_x = x // 2 - (crop_x // 2)
    start_y = y // 2 - (crop_y // 2)
    return img[start_y:start_y + crop_y, start_x:start_x + crop_x]


def invert_audio_resample_stft(stft_signal: ndarray, sr: int, track, phase: ndarray = None) -> ndarray:
    """
    A function that resamples the input signal to back to the original, convert to stereo and applies inverted STFT

    Args:
        stft_signal: The STFT that represents a signal in the time-frequency domain
        sr: Sample rate of input's sampling
        track: track object that contains all the class information about the selected track
        phase: Phase of the input's magnitude
    Returns:
        The inverted input signal to its original spectral and temporal context
    """
    inverted_signal = librosa.istft(stft_signal * phase) if phase is not None else librosa.istft(stft_signal, hop_length=256, win_length=win_length)
    inverted_signal = enhance(inverted_signal) if phase is not None else inverted_signal
    resample = librosa.resample(inverted_signal, sr, track.rate)
    y = convert_to_stereo(resample)
    return y


def audio_resample_stft(target_audio: ndarray, track: Track, sr: int, n_fft: int, hop_length: int) -> ndarray:
    """
    Function that is down sampling from 44100 to 22050 the input signal, convert signal to mono and computes the
    STFT for the given parameters

    Args:
        target_audio: Transposed numpy array of targeted audio
        track : musdb track object loaded using musdb
        sr: Sample rate of loaded track
        n_fft: length of the windowed signal after padding with zeros
        hop_length: number of audio samples between adjacent STFT columns
    Returns:
         A numpy array after the with the STFT of the input signal
    """
    signal_resample = librosa.resample(target_audio, track.rate, sr)
    signal_mono = librosa.to_mono(signal_resample)
    signal_stft = librosa.stft(signal_mono, n_fft=n_fft, win_length=win_length, hop_length=hop_length)

    return signal_stft


def concat_excerpts(audio_excerpt: ndarray, concat_array: list, output_width: int = img_width):
    """
    Helper function to crop center frame, append each stft excerpt into an array and finally concat the output into
    a numpy array.
    """
    if output_width != 1:
        # Block runs only for (2049, 9). (CNN_deep_separator & reference input concatenation)
        audio_excerpt = crop_center(audio_excerpt, 1, img_height)
    concat_array.append(audio_excerpt)
    concat_data = np.concatenate(concat_array, axis=1)
    return concat_data


def postprocess_filtering(audio_input: ndarray, wienerFlag: bool = False):
    """
    Helper function that applies median and wiener filtering to the estimated audio. This a post process technique
    a lot of state-of-the-art models use to the estimated output for better audio quality.
    """
    data, _ = librosa.decompose.hpss(audio_input)
    if wienerFlag:
        data = wiener(data, (300, 300))
    return data


def predict(y: ndarray, sr: int, output_path: str, song_name: str):
    """
    Predictions for input songs outside MUSDB18 datasets.
    Predicts the soft-masks of the targets then computes the estimated stems by an element-wise multiplication of the
    the mixture magnitude and the estimated masks. Finally, saves the output stems into a .wav format.
    """
    # Load model path
    model = tf.keras.models.load_model(
        "source_separation_model/separator_model/CNN_deep_separator_v2_best_model.h5")
    input_chunks = []
    prediction_chunks = []
    stft = librosa.stft(y, n_fft=frame_size, hop_length=hop_length)
    num_time_frames = stft.shape[1]
    stft = stft.T
    for time_frame in tqdm(range(num_time_frames)):
        chunk_array = np.array(stft[time_frame:9 + time_frame])
        chunk_array = chunk_array.T
        if chunk_array.shape[1] < 9:
            chunk_array = np.pad(chunk_array, [(0, 0), (0, 9 - chunk_array.shape[1])], mode='constant')

        convert_to_input_model = chunk_array[newaxis, :, :, newaxis]
        prediction = model.predict(convert_to_input_model)
        prediction = prediction.T
        prediction = np.squeeze(prediction) if prediction.shape[1] > 1 else prediction
        prediction = np.around(prediction)
        prediction_cf = crop_center(prediction.T, 1, img_height) if prediction.shape[1] > 1 else prediction

        input_signal_cf = crop_center(chunk_array, 1, img_height)
        prediction_chunks.append(prediction_cf)
        input_chunks.append(input_signal_cf)
        concatenated_input = np.concatenate(input_chunks, axis=1)
        concatenated_masks = np.concatenate(prediction_chunks, axis=1)
    concatenated_masks_acc = concatenated_masks
    concatenated_masks = 1 - concatenated_masks
    S_vocals = np.multiply(concatenated_input, concatenated_masks)
    S_acc = np.multiply(concatenated_input, concatenated_masks_acc)
    S_vocals, _ = librosa.decompose.hpss(S_vocals)
    estimate = librosa.istft(S_vocals, hop_length=hop_length, win_length=win_length)
    resample = librosa.resample(estimate, sr, 44100)
    resample_vocals = convert_to_stereo(resample)
    acc = librosa.istft(S_acc, hop_length=hop_length, win_length=win_length)
    resample_acc = librosa.resample(acc, sr, 44100)
    resample_acc = convert_to_stereo(resample_acc)
    output_path_vocals = os.path.join(output_path, f'{song_name}_vocals.wav')
    output_path_acc = os.path.join(output_path, f'{song_name}_accompaniments.wav')
    sf.write(file=output_path_acc, data=resample_acc.T, samplerate=44100)
    sf.write(file=output_path_vocals, data=resample_vocals.T, samplerate=44100)
    return


def model_predictions(mus: musdb, mus_id: int, hop_len: int, sr: int, n_samples: int, n_fft: int,
                      full_song: bool = None) -> tuple:
    """
    Function that estimates the input's vocal mask. In order for the predictions to happen the input needs to be split
    into chunks of 300ms to fit the shape (513, 25): (bins, frames) and apply a sliding window for one frame at a time.
    The shape specified is the shape of the STFTs numpy arrays MY_SEPARATOR_MODEL was trained to predict. So after
    the predictions are done I concatenate the split input and predictions. An additional harmonic, percussive and vocal
    separation by nn-filtering is performed to enhance the output's score in a SDR metric scale.

    Args:
        mus: A mus object from the musdb library
        mus_id: id of the selected song
        hop_len: number of audio samples between adjacent STFT columns
        sr: Sample rate of input's sampling
        n_samples: number of samples, a number ensued by the length of the separated song the user selected
        n_fft: length of the windowed signal after padding with zeros
        full_song: whether separation of full duration is selected
    Returns:
        The concatenated masks and input after the predictions are made.
    """
    # Load pre-trained model path
    model = tf.keras.models.load_model(
        "source_separation_model/separator_model/CNN_deep_separator_v2_best_model.h5")
    prediction_chunks = []
    prediction_acc_chunks = []
    input_chunks = []
    vocals_chunks = []
    accompaniment_chunks = []
    track = mus[int(mus_id)]
    sliding = round(hop_len / sr, 4)
    track.chunk_start = float(0) if full_song else float(30)
    print(track.chunk_start)
    duration = (hop_len / sr) * img_width
    duration = round(duration, 2)
    try:
        for x in tqdm(range(n_samples)):
            track.chunk_duration = duration
            track.chunk_start += sliding
            # extract stft mix for predictions. Vocals and accompaniments for reference reconstruction
            input_signal_stft = audio_resample_stft(track.audio.T, track, sr, n_fft, hop_len)
            vocals_stft = audio_resample_stft(track.targets['vocals'].audio.T, track, sr, n_fft, hop_len)
            accompaniment_stft = audio_resample_stft(track.targets['accompaniment'].audio.T, track, sr, n_fft, hop_len)
            # right pad with 0 only for full song, in case last chunk is less than 9 time frames
            if input_signal_stft.shape[1] < 9:
                input_signal_stft = np.pad(input_signal_stft, [(0, 0), (0, 9 - input_signal_stft.shape[1])], mode='constant')
            convert_to_input_model = input_signal_stft[newaxis, :, :, newaxis]
            # prediction output shape. (1, 2049, 9) -> CNN_deep_separator, (1, 2049) -> CNN_separator
            prediction = model.predict(convert_to_input_model)
            # reshape. (1, 2049, 9) -> to (2049, 9) for CNN_deep_separator || (1, 2049) -> (2049, 1) for CNN_separator
            prediction = np.squeeze(prediction, axis=0) if prediction.ndim > 2 else prediction.T
            # output's width defines whether to crop to center frame, executes only for CNN_deep_separator
            concatenated_masks = concat_excerpts(prediction, prediction_chunks, prediction.shape[1])
            concatenated_acc_masks = concat_excerpts(np.around(prediction), prediction_acc_chunks, prediction.shape[1])
            concatenated_input = concat_excerpts(input_signal_stft, input_chunks)
            concatenated_ref_vocals = concat_excerpts(vocals_stft, vocals_chunks)
            concatenated_ref_accompaniment = concat_excerpts(accompaniment_stft, accompaniment_chunks)
    except Exception:
        print('out of range!')
        pass
    accompaniment_masks = concatenated_acc_masks
    concatenated_masks = 1 - concatenated_masks
    # round values under 0.05 to 0.002. Works as a filter to reduce very low dB background sounds
    soft_mask_vocals = np.where(concatenated_masks <= 0.15, 0, concatenated_masks)
    my_track = {
        'name': track.name,
        'rate': track.rate,
        'vocals': concatenated_ref_vocals,
        'accompaniment': concatenated_ref_accompaniment
    }
    return soft_mask_vocals, concatenated_input, accompaniment_masks, my_track


def load_service_menu(mus: musdb) -> tuple:
    """
    Function that prints a list of all available songs and takes as input the id of the song the user selected.

    Args:
        mus: mus object from the musdb library
    Returns:
        song name and id from the test subset
    """
    song_labels_array = []
    for i in range(len(mus)):
        print(f'{i + 1}) {mus[i]}')
        song_labels_array.append(mus[i])

    num = input("Select a song to separate giving its ID: ")
    song_name = song_labels_array[int(num) - 1]
    mus_id = int(num) - 1

    return song_name, mus_id


def enhance(y: ndarray) -> ndarray:
    """
    Function that enhances the input vocal signal

    Args:
        y: input signal STFT
    Returns:
         An enhanced audio signal of the primary spectral information
    """
    apply_audio_effects = AudioEffectsChain().lowshelf(gain=30.0, frequency=260, slope=0.1).reverb(reverberance=25,
                                                                                                   hf_damping=5,
                                                                                                   room_scale=5,
                                                                                                   stereo_depth=50,
                                                                                                   pre_delay=20,
                                                                                                   wet_gain=0,
                                                                                                   wet_only=False)
    fx = apply_audio_effects
    y_enhanced = fx(y)

    return y_enhanced


def load_input(y: str) -> tuple:
    """
    Function that loads an audio file as a floating point time series.

    Args:
        y: path of the input signal
    Returns:
         input's signal numpy array and sample rate
    """
    y, sr = librosa.load(y)
    return y, sr


def right_pad_if_necessary(signal: ndarray, num_samples: int) -> ndarray:
    length_signal = signal.shape[1]
    print(signal.shape, num_samples, length_signal)
    if length_signal < num_samples:
        num_missing_samples = num_samples - length_signal
        audio_estimates = np.pad(
            signal,
            [
                (0, 0),
                (0, num_missing_samples),
                [0, 0]
            ],
            mode='constant'
        )
    return audio_estimates


def eval_mus_track(
    user_reference,
    user_estimates,
    track,
    output_dir=None,
    mode='v4',
    win=1.0,
    hop=1.0
):
    """Compute all bss_eval metrics for the musdb track and estimated signals,
    given by a `user_estimates` dict.

    Parameters
    ----------
    user_reference : Dict
        dictionary, containing the user references as np.arrays.
    user_estimates : Dict
        dictionary, containing the user estimates as np.arrays.
    track : Dict
        dictionary, containing my track attributes.
    output_dir : str
        path to output directory used to save evaluation results. Defaults to
        `None`, meaning no evaluation files will be saved.
    mode : str
        bsseval version number. Defaults to 'v4'.
    win : int
        window size in

    Returns
    -------
    scores : TrackStore
        scores object that holds the framewise and global evaluation scores.
    """
    audio_estimates = []
    audio_reference = []
    track_targets = ['vocals', 'accompaniment']


    # make sure to always build the list in the same order
    # therefore track.targets is an OrderedDict
    eval_targets = []  # save the list of target names to be evaluated
    for key in list(track_targets):
        try:
            # try to fetch the audio from the user_results of a given key
            user_estimates[key]
        except KeyError:
            # ignore wrong key and continue
            continue

        # append this target name to the list of target to evaluate
        eval_targets.append(key)

    if hasattr(track, 'name'):
        track_name = track.name
        track_rate = track.rate
    else:
        track_name = track['name']
        track_rate = track['rate']
    data = museval.TrackStore(win=win, hop=hop, track_name=track_name)
    # check if vocals and accompaniment is among the targets
    has_acc = all(x in eval_targets for x in track_targets)
    if has_acc:
        # remove accompaniment from list of targets, because
        # the voc/acc scenario will be evaluated separately
        eval_targets.remove('accompaniment')

    if len(eval_targets) >= 2:
        # compute evaluation of remaining targets
        for target in eval_targets:
            audio_estimates.append(user_estimates[target])
            audio_reference.append(user_reference[target])
        SDR, ISR, SIR, SAR = museval.evaluate(
            audio_reference,
            audio_estimates,
            win=int(win*track_rate),
            hop=int(hop*track_rate),
            mode=mode
        )

        # iterate over all evaluation results except for vocals
        for i, target in enumerate(eval_targets):
            if target == 'vocals' and has_acc:
                continue

            values = {
                "SDR": SDR[i].tolist(),
                "SIR": SIR[i].tolist(),
                "ISR": ISR[i].tolist(),
                "SAR": SAR[i].tolist()
            }

            data.add_target(
                target_name=target,
                values=values
            )
    elif not has_acc:
        warnings.warn(
            UserWarning(
                "Incorrect usage of BSSeval : at least two estimates must be provided. Target score will be empty."
            )
        )

    # add vocal accompaniment targets later
    if has_acc:
        # add vocals and accompaniments as a separate scenario
        eval_targets = ['vocals', 'accompaniment']

        audio_estimates = []
        audio_reference = []

        for target in eval_targets:
            audio_estimates.append(user_estimates[target])
            audio_reference.append(user_reference[target])

        SDR, ISR, SIR, SAR = museval.evaluate(
            audio_reference,
            audio_estimates,
            win=int(win*track_rate),
            hop=int(hop*track_rate),
            mode=mode
        )

        # iterate over all targets
        for i, target in enumerate(eval_targets):
            values = {
                "SDR": SDR[i].tolist(),
                "SIR": SIR[i].tolist(),
                "ISR": ISR[i].tolist(),
                "SAR": SAR[i].tolist()
            }

            data.add_target(
                target_name=target,
                values=values
            )

    if output_dir:
        # validate against the schema
        x = data.validate()

        try:
            subset_path = os.path.join(
                output_dir,
                'test'
            )

            if not os.path.exists(subset_path):
                os.makedirs(subset_path)

            with open(
                os.path.join(subset_path, track_name) + '.json', 'w+'
            ) as f:
                f.write(data.json)

        except IOError:
            pass

    return data, x


if __name__ == '__main__':
    print('Helper Functions for vocal separation. Run >> prediction_service.py -h for help.')
    pass
