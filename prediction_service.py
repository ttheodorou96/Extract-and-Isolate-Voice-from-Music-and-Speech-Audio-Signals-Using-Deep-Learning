import argparse
import sys

import librosa
import musdb
import numpy as np
import museval
import os

from lib.enums import Method, Tools, Tools_resp
from lib.estimates_scores import crop_song, get_estimates_openumnix, sample_and_save_image, \
    evaluate_open_unmix_estimates, save_separation_scores, get_estimates_and_scores, estimate_scores_user_reference
from lib.separate_helper_funcs import create_folder_structure, save_stems, save_images, \
    load_service_menu, audio_resample_stft, \
    model_predictions, save_plot_box, librosa_decomposition, save_audio_as_wav, invert_audio_resample_stft, \
    load_input, save_image_input_song, evaluate_SDR_format, postprocess_filtering, predict, mask_compute
from lib.enums import Modes
import torch
import soundfile as sf

sys.path.append('lib/helper_funcs.py')
from lib.helper_funcs import yaml_loader

from spleeter.separator import Separator

root_dir = os.getcwd()
cf_path = "configuration_files/vocal_separator_tool/init_separator_tool.yaml"
config = yaml_loader(filepath=cf_path)
db_path = os.path.join(root_dir, 'musdb18')
mus = musdb.DB(root=db_path, subsets='test')
hop_length = config.get('hop_length')
sr = config.get('sample_rate')
n_fft = config.get('frame_size')
img_width = config.get('img_size').get('width')


def open_unmix(song_name: str, tool_name: str, mus_id: int = None, selected_song_dir: str = None,
               n_samples: int = None, input_song: dict = None, resp_flag: bool = None) -> str or None:
    """
    A function that executes separation of the input signal via OPEN-UNMIX's pre-trained models. Open-Unmix is a deep
    neural network reference implementation for music source separation, applicable for researchers, audio engineers
    and artists. In this service the the input signal is separated to 2 stems, Vocals and Accompaniments.
    The results are being compared to the other available tools based on aggregated scores, images and audio files.
    Args:
        song_name: name of the selected song
        mus_id: id of the selected song, from the mus test dataset
        tool_name: name of the selected tool
        selected_song_dir: a path of the selected song directory, also is level one of the prediction service folder
        structure.
        n_samples: number of samples for desired song duration
        input_song: json of the loaded input song and its sample rate
        resp_flag: is true when mode is not acapella
    Returns:
        None
    """
    print(resp_flag)
    if mus_id is not None:
        track = crop_song(mus=mus, mus_id=mus_id, hop_length=hop_length, sr=sr, n_samples=n_samples)

        stems_dir, images_dir, estimates_dir, selected_tool_dir = create_folder_structure(
            selected_song_dir=selected_song_dir,
            tool_name=tool_name)
        estimates_openunmix, vocal_estimates, accompaniment_estimates = get_estimates_openumnix(track=track,
                                                                                                stems_dir=stems_dir)
        mixture = track.audio.T
        input_vocals = audio_resample_stft(vocal_estimates, track, sr, n_fft, hop_length)
        input_accompaniment = audio_resample_stft(accompaniment_estimates, track, sr, n_fft, hop_length)
        input_mix = audio_resample_stft(mixture, track, sr, n_fft, hop_length)

        input_mask = mask_compute(input_vocals, input_accompaniment)
        save_images(mix=input_mix, masks=input_mask, vocals=input_vocals, song_name=song_name,
                    images_dir=images_dir)
        print('Image Saved!')

        track_scores_openunmix = evaluate_open_unmix_estimates(track=track, estimates=estimates_openunmix,
                                                               estimates_dir=estimates_dir)

        method = [Method.open_unmix.value]
        results = museval.EvalStore()
        track_scores = [track_scores_openunmix]
        estimate_path = save_separation_scores(results=results, track_scores=track_scores, method=method,
                                               estimates_dir=estimates_dir)
        return estimate_path
    else:
        track = input_song.get('song')
        sample_rate = input_song.get('sr')
        selected_tool_dir = os.path.join(selected_song_dir, tool_name)
        if not os.path.isdir(selected_tool_dir):
            os.mkdir(selected_tool_dir)
        _, voice_estimates, accompaniment_estimates = get_estimates_openumnix(track=track,
                                                                              stems_dir=selected_tool_dir,
                                                                              sr=sample_rate, song_name=song_name)
        if resp_flag:
            separator = torch.hub.load('sigsep/open-unmix-pytorch', 'umxse')
            voice = torch.from_numpy(voice_estimates)
            voice = voice[None]
            voice_isolation = separator(voice)
            # output of separator's shape is torch.Size([1, 2, 2, n_samples])
            voice_isolation = voice_isolation.cpu().detach().numpy()[0]
            voice_isolation = np.squeeze(voice_isolation)
            for idx, x in enumerate(voice_isolation):
                if idx == 0:
                    high_voice_estimates = x
                else:
                    low_bg_estimates = x
            # create isolated voice content with 4 channels
            isolated = np.vstack((high_voice_estimates, low_bg_estimates))
            isolated_save_path = os.path.join(selected_tool_dir, f'{song_name}_isolated.wav')
            sf.write(file=isolated_save_path, data=isolated.T, samplerate=sample_rate * 2)
            s_mixture = librosa.stft(librosa.to_mono(track), n_fft=n_fft, hop_length=hop_length)
            s_vocals = librosa.stft(librosa.to_mono(isolated), n_fft=n_fft, hop_length=hop_length)
            s_background = librosa.stft(librosa.to_mono(accompaniment_estimates), n_fft=n_fft, hop_length=hop_length)
            save_image_input_song(mixture=s_mixture, vocals=s_vocals, accompaniment=s_background,
                                  input_name=song_name, output_dir=selected_tool_dir)
            print('Image Saved!')

    return


def spleeter(song_name: str, tool_name: str, mus_id: int = None, selected_song_dir: str = None,
             n_samples: int = None, input_song: dict = None) -> str:
    """
    A function that executes Spleeter separation to the input signal. Spleeter is a State-of-the-art music source
    separation neural network model that performs an audio signal separation from two to five components. For this run
    the input signal is separated to 2 stems, Vocals and Accompaniments. n this service the the input signal is
    separated to 2 stems, Vocals and Accompaniments. The results are being compared to the other available
    tools based on aggregated scores, images and audio files.
    Args:
        song_name: name os the selected song
        mus_id: id of the selected song, from the mus test dataset
        tool_name: a list of the all the available tools
        selected_song_dir: a path of the selected song directory, also is level one of the prediction service folder
        structure.
        n_samples: number of samples for desired song duration
        input_song: json of the loaded input song and its sample rate
    Returns:
        None
    """
    separator = Separator('spleeter:2stems')
    if mus_id is not None:
        track = crop_song(mus=mus, mus_id=mus_id, hop_length=hop_length, sr=sr, n_samples=n_samples)

        stems_dir, images_dir, estimates_dir, selected_tool_dir = create_folder_structure(selected_song_dir,
                                                                                          tool_name)
        mixture = track.audio
        prediction = separator.separate(mixture)
        vocal_prediction = prediction.get('vocals').T
        accompaniment_prediction = prediction.get('accompaniment').T
        mixture = mixture.T
        input_vocals = audio_resample_stft(vocal_prediction, track, sr, n_fft, hop_length)
        input_accompaniment = audio_resample_stft(accompaniment_prediction, track, sr, n_fft, hop_length)
        input_mix = audio_resample_stft(mixture, track, sr, n_fft, hop_length)

        input_mask = mask_compute(input_vocals, input_accompaniment)
        save_images(mix=input_mix, masks=input_mask, vocals=input_vocals, song_name=song_name,
                    images_dir=images_dir)
        print('Image Saved!')

        save_audio_as_wav(mixture, 'mixture', song_name, stems_dir)
        save_audio_as_wav(vocal_prediction, 'vocals', song_name, stems_dir)
        save_audio_as_wav(accompaniment_prediction, 'accompaniment', song_name, stems_dir)
        print('Stems Saved!')

        method = [Method.spleeter.value]
        vocals = vocal_prediction.T.astype('float64')
        accompaniments = accompaniment_prediction.T.astype('float64')
        estimate_path = get_estimates_and_scores(method, vocals, accompaniments, track, estimates_dir)

        return estimate_path
    else:
        track = input_song.get('song')
        selected_tool_dir = os.path.join(selected_song_dir, tool_name)
        if not os.path.isdir(selected_tool_dir):
            os.mkdir(selected_tool_dir)
        mixture = track
        prediction = separator.separate(mixture)
        vocal_prediction = prediction.get('vocals').T
        accompaniment_prediction = prediction.get('accompaniment').T
        mixture = mixture.T
        save_audio_as_wav(mixture, 'mixture', song_name, selected_tool_dir)
        save_audio_as_wav(vocal_prediction, 'vocals', song_name, selected_tool_dir)
        save_audio_as_wav(accompaniment_prediction, 'accompaniment', song_name, selected_tool_dir)
        print('Stems Saved!')


def my_model(song_name: str, tool_name: str, selected_song_dir: str, n_samples: int, mus_id: int = None,
             wienerFlag: bool = False, input_song: dict = None, full_song: bool = None) -> str:
    """
    A function that executes a pre-trained CNN model created by me. The dataset I used for the training was from the
    musdb18 training dataset, while implemented a data augmentation method to increase number of samples. Model's
    weights have been shaped before this run. The model takes as input the selected song and separates the vocal
    component, the accompaniment component is evaluated buy subtracting the vocal numpy array estimates from the input
    mix. Method of separation is inspired and based on vocal binary masks applied to a mixture
    spectrogram image to extract the desired vocal component.

    Args:
        song_name: name os the selected song
        tool_name: a list of the all the available tools
        selected_song_dir: a path of the selected song directory, also is level one of the prediction service folder
        structure.
        n_samples: number of samples for desired song duration
        mus_id: id of the selected song, from the mus test dataset
        wienerFlag: given on execution, when true activates wiener filtering in the post-process
        input_song: json of the loaded input song and its sample rate
        full_song: whether separation of full duration is selected
    Returns:
        None
    """
    print('Loading model input... Please wait...')
    print('Predicting estimates please wait...')
    if mus_id is not None:
        predicted_soft_mask, input_mix, predicted_accompaniment_masks, my_track = model_predictions(mus,
                                                                                                    mus_id,
                                                                                                    hop_length,
                                                                                                    sr,
                                                                                                    n_samples,
                                                                                                    n_fft,
                                                                                                    full_song)

        stems_dir, images_dir, estimates_dir, selected_tool_dir = create_folder_structure(selected_song_dir, tool_name)
        S_vocals = np.multiply(input_mix, predicted_soft_mask)
        S_vocals_filtered = postprocess_filtering(S_vocals, wienerFlag)
        S_accompaniments = np.multiply(input_mix, predicted_accompaniment_masks)

        save_stems(stem=input_mix, label='mixture', song_name=song_name, stems_dir=stems_dir, sr=sr)
        save_stems(stem=S_vocals_filtered, label='vocals', song_name=song_name, stems_dir=stems_dir, sr=sr)
        save_stems(stem=S_accompaniments, label='accompaniments', song_name=song_name, stems_dir=stems_dir, sr=sr)
        print('Stems Saved!')

        save_images(mix=input_mix, masks=predicted_soft_mask, vocals=S_vocals_filtered, song_name=song_name,
                    images_dir=images_dir)
        print('Image Saved!')

        method = [Method.my_separator.value]
        audio_reference = {
            'vocals': evaluate_SDR_format(my_track['vocals'], sr, my_track['rate']),
            'accompaniment': evaluate_SDR_format(my_track['accompaniment'], sr, my_track['rate'])
        }
        audio_estimates = {
            'vocals': evaluate_SDR_format(S_vocals, sr, my_track['rate']),
            'accompaniment': evaluate_SDR_format(S_accompaniments, sr, my_track['rate'])
        }
        estimate_path = estimate_scores_user_reference(method, audio_estimates, my_track, audio_reference,
                                                       estimates_dir)

        return estimate_path
    else:
        y = input_song.get('song')
        sample_rate = input_song.get('sr')
        predict(y, sample_rate, selected_song_dir, song_name)
        print('Stems Saved!')


def repet_sim(song_name: str, tool_name: str, mus_id: int = None, selected_song_dir: str = None,
              n_samples: int = None, input_song: dict = None) -> str:
    """
    A function that executes REPET_SIM method of Rafii and Pardo, 2012. A method introduced to many people by librosa
    that performs a vocal separation from an input mix by converting a non-local filtering into a soft mask by Wiener
    filtering, similar in spirit to the soft-masking method used by Fitzgerald, 2012. This method is not based on a
    machine nor deep learning model so the results are clearly not as good as the other tools.

    Args:
        song_name: name os the selected song
        mus_id: id of the selected song, from the mus test dataset
        tool_name: a list of the all the available tools
        selected_song_dir: a path of the selected song directory, also is level one of the prediction service folder
        structure.
        n_samples: number of samples for desired song duration
        input_song: json of the loaded input song and its sample rate
    Returns:
        None
    """
    if mus_id is not None:
        track = crop_song(mus, mus_id, hop_length, sr, n_samples)
        input_mix = audio_resample_stft(track.audio.T, track, sr, n_fft, hop_length)

        S_vocals, S_background, phase, S_full, mask_v = librosa_decomposition(input_mix=input_mix, sr=sr)
        stems_dir, images_dir, estimates_dir, selected_tool_dir = create_folder_structure(selected_song_dir, tool_name)
        save_stems(stem=S_full, label='mixture', song_name=song_name, stems_dir=stems_dir, sr=sr, phase=phase)
        save_stems(stem=S_vocals, label='vocals', song_name=song_name, stems_dir=stems_dir, sr=sr, phase=phase)
        save_stems(stem=S_background, label='accompaniments', song_name=song_name, stems_dir=stems_dir, sr=sr,
                   phase=phase)
        print('Stems Saved!')

        save_images(mix=input_mix, masks=mask_v, vocals=S_vocals, song_name=song_name,
                    images_dir=images_dir, accompaniments=S_background)
        print('Images Saved!')

        method = [Method.librosa_separator.value]
        estimates_vocals = evaluate_SDR_format(S_vocals, sr, track.rate, phase)
        estimates_acc = evaluate_SDR_format(S_background, sr, track.rate, phase)
        estimate_path = get_estimates_and_scores(method, estimates_vocals, estimates_acc, track, estimates_dir)
        return estimate_path
    else:
        track = input_song.get('song')
        sample_rate = input_song.get('sr')
        selected_tool_dir = os.path.join(selected_song_dir, tool_name)
        if not os.path.isdir(selected_tool_dir):
            os.mkdir(selected_tool_dir)
        s_mixture = librosa.stft(librosa.to_mono(track), n_fft=n_fft, hop_length=hop_length)
        S_vocals, S_background, phase, _, _ = librosa_decomposition(input_mix=s_mixture, sr=sample_rate)
        save_stems(stem=S_vocals, label='vocals', song_name=song_name, stems_dir=selected_tool_dir, sr=sr, phase=phase)
        save_stems(stem=S_background, label='accompaniments', song_name=song_name, stems_dir=selected_tool_dir, sr=sr,
                   phase=phase)


def noise_reduction_tool(input_song: dict, input_name_dir: str, tool_name: str, input_name: str):
    """
    Function that takes as input the audio file the user selected and executes a noise reduction process via the REPET-
    SIM method with Nearest-Neighbour and Median filtering to eliminate environmental sounds that deviate from the
    spectral information of interest. After the separation of the intended audio signal and the noise the audio and
    image output is saved into the respective selected or default folder. Also a SDR metric evaluation is performed in
    the original and de-noised signal.

    Args:
        input_song: a json with the input signal and its sample rate
        input_name_dir: output directory of the results of the noise reduction tool, is either specified by
            the user or is by default the root directory of the repository
        tool_name: name of the selected method of separation
        input_name: name of the input signal
    Returns:

    """
    track = input_song.get('song')
    sample_rate = input_song.get('sr')
    S_audio = librosa.stft(track, n_fft=n_fft, hop_length=hop_length)
    S_vocals, S_background, phase, _, _ = librosa_decomposition(input_mix=S_audio, sr=sample_rate)
    selected_tool_dir = os.path.join(input_name_dir, tool_name)
    if not os.path.isdir(selected_tool_dir):
        os.mkdir(selected_tool_dir)

    save_stems(stem=S_vocals, label='vocals', song_name=input_name, stems_dir=selected_tool_dir, sr=sr, phase=phase,
               noise_redux=True)
    save_stems(stem=S_background, label='background', song_name=input_name, stems_dir=selected_tool_dir, sr=sr,
               phase=phase, noise_redux=True)
    print('Stems Saved!')
    save_image_input_song(mixture=S_audio, vocals=S_vocals, accompaniment=S_background, input_name=input_name,
                          output_dir=selected_tool_dir)
    print('Image Saved!')

    return


def all_selected(song_name: str, mus_id: int, tool_name: list, selected_song_dir: str, n_samples: int,
                 full_song: bool = None) -> None:
    """
    A function that executes all the available tools and save plot box image comparing all their aggregated scores.

    Args:
        song_name: name os the selected song
        mus_id: id of the selected song, from the mus test dataset
        tool_name: a list of the all the available tools
        selected_song_dir: a path of the selected song directory, also is level one of the prediction service folder
        structure.
        n_samples: number of samples for desired song duration
        full_song: whether separation of full duration is selected
    Returns:
        None
    """
    open_unmix_csv = open_unmix(song_name=song_name, tool_name=tool_name[0], mus_id=mus_id,
                                selected_song_dir=selected_song_dir, n_samples=n_samples)
    my_model_csv = my_model(song_name=song_name, tool_name=tool_name[2], selected_song_dir=selected_song_dir,
                            n_samples=n_samples, mus_id=mus_id, wienerFlag=False, full_song=full_song)
    spleeter_csv = spleeter(song_name=song_name, tool_name=tool_name[1], mus_id=mus_id,
                            selected_song_dir=selected_song_dir, n_samples=n_samples)
    repet_sim_csv = repet_sim(song_name=song_name, mus_id=mus_id, tool_name=tool_name[3],
                              selected_song_dir=selected_song_dir, n_samples=n_samples)
    scores_path_array = [open_unmix_csv,  spleeter_csv, my_model_csv, repet_sim_csv]
    save_plot_box(path_array=scores_path_array, selected_song_dir=selected_song_dir)
    return


def tool_switch(tool_lst: list, prediction_service_dir: str, song_obj: tuple = None, n_samples: int = None,
                input_path: str = None, resp_flag: bool = None, wienerFlag: bool = None, full_song: bool = None) -> None:
    """
    A function that decides what to run depending the user's input.

    Args:
        tool_lst: user input for selected tool to run
        song_obj: song object, contains song_name, mus_id
        n_samples: number of samples for desired song duration
        prediction_service_dir: path of the output to be saved
        input_path: path of the input song
        resp_flag: is true when mode is not acapella
        wienerFlag: given on execution, when true activates wiener filtering in the post-process
        full_song: whether separation of full duration is selected
    Returns:
        None
    """
    dict_tool = yaml_loader(filepath='dictionaries/tool_dictionary.yaml')
    if song_obj is not None:
        song_name, mus_id = song_obj
    else:
        input_song, sample_rate = load_input(input_path)
        input_song_json = {
            'song': input_song,
            'sr': sample_rate
        }
        song_name_w_extention = os.path.basename(os.path.normpath(input_path))
        song = os.path.splitext(song_name_w_extention)
        song_name = song[0]
    selected_song_dir = os.path.join(prediction_service_dir, f'{song_name}')
    if not os.path.isdir(selected_song_dir):
        os.mkdir(selected_song_dir)
    if resp_flag:
        for tool_id in tool_lst:
            if tool_id == Tools_resp.nn_filtering.value:
                tool_name = dict_tool.get('nn_filtering')
                noise_reduction_tool(input_song_json, selected_song_dir, tool_name, song_name)
            elif tool_id == Tools_resp.dnn_lstm.value:
                tool_name = dict_tool.get('dnn_lstm')
                open_unmix(song_name=song_name, tool_name=tool_name, selected_song_dir=selected_song_dir,
                           input_song=input_song_json, resp_flag=True)
    else:
        for tool_id in tool_lst:
            if tool_id == Tools.open_unmix.value:
                tool_name = dict_tool.get('open_unmix')
                if input_path is not None:
                    open_unmix(song_name=song_name, tool_name=tool_name, selected_song_dir=selected_song_dir,
                               input_song=input_song_json)
                else:
                    open_unmix(song_name=song_name, tool_name=tool_name, mus_id=mus_id,
                               selected_song_dir=selected_song_dir, n_samples=n_samples)
            elif tool_id == Tools.spleeter.value:
                tool_name = dict_tool.get('spleeter')
                if input_path is not None:
                    spleeter(song_name=song_name, tool_name=tool_name, selected_song_dir=selected_song_dir,
                             input_song=input_song_json)
                else:
                    spleeter(song_name=song_name, tool_name=tool_name, mus_id=mus_id,
                             selected_song_dir=selected_song_dir, n_samples=n_samples)
            elif tool_id == Tools.my_separator.value:
                tool_name = dict_tool.get('my_separator')
                if input_path is not None:
                    my_model(song_name=song_name, tool_name=tool_name, selected_song_dir=selected_song_dir,
                             n_samples=n_samples, input_song=input_song_json)
                else:
                    my_model(song_name=song_name, tool_name=tool_name, selected_song_dir=selected_song_dir,
                             n_samples=n_samples, mus_id=mus_id, wienerFlag=wienerFlag, full_song=full_song)
            elif tool_id == Tools.librosa_separator.value:
                tool_name = dict_tool.get('librosa_separator')
                if input_path is not None:
                    repet_sim(song_name=song_name, tool_name=tool_name, selected_song_dir=selected_song_dir,
                              input_song=input_song_json)
                else:
                    repet_sim(song_name=song_name, mus_id=mus_id, tool_name=tool_name,
                              selected_song_dir=selected_song_dir, n_samples=n_samples)
            elif tool_id == Tools.all.value:
                tool_name = dict_tool.get('all')
                all_selected(song_name, mus_id, tool_name, selected_song_dir, n_samples, full_song)
            else:
                print("Oops! That was no valid Tool ID.  Try again...")
            print(f'Tool: {tool_name} \nSong: {song_name}\nExecuted Successfully!\n'
                  f'\nCheck corresponding directories for saved images, stems and estimates.')

    return


def argParser():
    """
    Argument parser for the prediction service tool.

    Returns:
    -------
    args: Dict
        A dictionary of all the positional and optional arguments specified.
    """
    parser = argparse.ArgumentParser(description='Acapella extraction from a music track and voice/respiratory '
                                                 'extraction from a `noisy` environment Tool')
    parser.add_argument('-f',
                        '--mode',
                        required=True,
                        metavar='',
                        type=Modes,
                        help='Select mode for the tool, available options are `acapella`, `voice`, `respiratory`')
    parser.add_argument('-p',
                        '--path',
                        required=False,
                        metavar='',
                        type=str,
                        help='Set full path for the input signal')
    parser.add_argument('-o',
                        '--output',
                        required=False,
                        metavar='',
                        type=str,
                        help='set output folder for the results to save')
    parser.add_argument('-w',
                        '--wiener',
                        required=False,
                        metavar='',
                        default=False,
                        type=bool,
                        help='set wiener flag, whether to use wiener filtering in the post-process. Applied only '
                             'on My_Model_Separator')

    args = parser.parse_args()
    return args


def main():
    """
    The main menu of the separation prediction service. User chooses between the two main tools, vocal extraction from
    an audio mixture and noise reduction from environmental sounds.

    If user selects the vocal extraction tool then he/she chooses one or more method(s) of separation to execute for
    a song in the list provided from the musdb18 testing dataset (50 songs). One method includes my own Deep Convolution
    Neural Network trained with augmented data from musdb18, two state of the art DNN source separation models and a
    Nearest Neighbour filtering technique from an open source library, Librosa.
    User can also choose the duration for the input's results. Options for the service are 5s, 30s or full
    song. 5 seconds for a quick demo, full song for more elaborate results.

    If user selects the noise reduction tool then he/she selects an audio file (supported audio codecs .mp4, .wav, .acc)
    from his device to de-noise and the extracted features are saved to path of preference.

    Corresponding sub-directories are dynamically created for each song. Image, csv and audio files are saved to each
    directory. User can also specify the output folder, if not output directory will automatically created inside the
    repository folder

    `Usage: prediction_service.py -f <mode> -p <input_path> -o <output_path>`

    Returns:
        None
    """
    args = argParser()
    root_directory = args.output if args.output is not None else root_dir
    prediction_service_dir = os.path.join(root_directory, 'output')
    if not os.path.isdir(prediction_service_dir):
        os.mkdir(prediction_service_dir)
    mode = args.mode.value
    path = args.path
    wienerFlag = args.wiener
    if mode == Modes.acapella.value:
        print('Singing Voice and Accompaniments Separation from a music tracks')
        print('>> Open-unmix [1]\n>> Spleeter [2]\n>> My_Model_Separator [3]\n>> REPET-SIM [4]')
        tool_lst = input('Please select one or more methods for the vocal separation by its ID or type `All` to select '
                         'all: ').split(',')
        if path is not None:
            tool_switch(tool_lst=tool_lst, prediction_service_dir=prediction_service_dir,
                        input_path=path, wienerFlag=wienerFlag)
        else:
            print('>> 5 second chunk [1]\n>> 30 second chunk [2]\n>> Full song [3]\n>> testing [4]')
            duration_choice = input('Please select the duration of the separated song you prefer: ')
            duration_dict = yaml_loader('dictionaries/duration_dictionary.yaml')
            duration_choice = duration_dict.get(f'{duration_choice}')
            print('Please wait...')
            song_obj = load_service_menu(mus=mus)
            _, mus_id = song_obj
            track = mus[mus_id]
            full_song: bool = False
            if duration_choice == 'full_song':
                duration_choice = track.duration
                full_song = True
            print(duration_choice)
            n_samples = round(float(duration_choice) * sr / hop_length)
            tool_switch(tool_lst=tool_lst, prediction_service_dir=prediction_service_dir,
                        song_obj=song_obj, n_samples=n_samples, wienerFlag=wienerFlag, full_song=full_song)
    else:
        print('Voice/Respiratory extraction and isolation from a `noisy` environment')
        print('>> Deep bi-directional LSTM [1]\n>> Nearest-Neighbour Filtering [2]\n')
        tool_lst = input('Please select a method for the voice/respiratory extraction ').split(',')
        if path is not None:
            tool_switch(tool_lst=tool_lst, prediction_service_dir=prediction_service_dir,
                        input_path=path, resp_flag=True)
        else:
            optional_path = input('Input signal is required for this mode. Please type input`s full path.\n')
            tool_switch(tool_lst=tool_lst, prediction_service_dir=prediction_service_dir,
                        input_path=optional_path, resp_flag=True)

    return


if __name__ == '__main__':
    main()
