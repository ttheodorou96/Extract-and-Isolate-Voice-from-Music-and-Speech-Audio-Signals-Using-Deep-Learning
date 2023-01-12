import logging
import os
import museval
import torch
from musdb.audio_classes import Track
from numpy import ndarray
from openunmix import predict
from lib.separate_helper_funcs import mask_compute, save_images, right_pad_if_necessary, eval_mus_track, \
    audio_resample_stft, save_audio_as_wav

root_dir = os.getcwd()
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def estimate_scores_user_reference(method: list, audio_estimates: dict, my_track: dict, audio_ref: dict,
                                   estimates_dir: str) -> str:
    """
        Estimates SDR, ISR, SAR, SIR scores with museval. We use user reference instead of Track reference in order to
        align estimates and references. Since some audio information lost during prediction phase as it predicts the
        center frame from the input excerpts.
        Example.
            input shape: (n_freq_bins, n_time_frames)

    """
    track_scores, _ = eval_mus_track(
        user_reference=audio_ref, user_estimates=audio_estimates, track=my_track, output_dir=estimates_dir
    )
    results = museval.EvalStore()
    track_scores = [track_scores]
    estimate_path = save_separation_scores(
        results=results, track_scores=track_scores, method=method, estimates_dir=estimates_dir
    )
    return estimate_path


def get_estimates_and_scores(method: list, vocals: ndarray, accompaniments: ndarray, track, estimates_dir: str) -> str:
    estimates_json = {
        'vocals': vocals,
        'accompaniment': accompaniments
    }
    track_scores = museval.eval_mus_track(
        track=track, user_estimates=estimates_json, output_dir=estimates_dir
    )
    results = museval.EvalStore()
    track_scores = [track_scores]
    estimate_path = save_separation_scores(
        results=results, track_scores=track_scores, method=method, estimates_dir=estimates_dir
    )
    return estimate_path


def sample_and_save_image(track, hop_length: int, sr: int, n_fft: int, song_name: str, images_dir: str,
                          vocal_estimates: ndarray, mixture: ndarray, accompaniment_estimates: ndarray) -> None:
    """
    A Function that save the inputs separated components into an image.
    Args:
        track: instance of mus object
        hop_length: hop length/window length of sampling
        sr: sample rate of sampling
        n_fft: number of samples per frame
        song_name: name of the input song
        images_dir: absolute path of the output save image
        vocal_estimates: a numpy array with the vocal estimates
        mixture: a numpy array with the mixture
        accompaniment_estimates: a numpy array with the accompaniment estimates
    Returns:
        None
    """
    input_vocals = audio_resample_stft(vocal_estimates, track, sr, n_fft, hop_length)
    input_accompaniment = audio_resample_stft(accompaniment_estimates, track, sr, n_fft, hop_length)
    input_mix = audio_resample_stft(mixture, track, sr, n_fft, hop_length)

    input_mask = mask_compute(input_vocals, input_accompaniment)
    save_images(input=input_mix, masks=input_mask, vocals=input_vocals, song_name=song_name,
                images_dir=images_dir)
    print('Image Saved!')
    return


def crop_song(mus, mus_id: int, hop_length: int, sr: int, n_samples: int):
    """
    A function that return the input's song duration
    Args:
        mus: a mus object from mubdb
        mus_id: id of the selected song from the testing dataset
        hop_length: hop length/window length of sampling
        sr: sample rate of sampling
        n_samples: number of samples for desired song duration
    Returns:
         track: modified track object
    """
    track = mus[int(mus_id)]
    sliding = round(hop_length / sr, 4)
    track.chunk_start = float(30)
    track.chunk_duration = sliding * n_samples
    return track


def save_open_unmix_stems(audio: list, track_name: str, stems_dir: str):
    """
    A function to save the estimates of the open-unmix pre-trained model to the corresponding directory

    Args:
        audio: A list of json objects that contains the audio and the its label
        track_name: Song name
        stems_dir: An absolute path to the stems directory
    Returns
        None
    """
    for a in audio:
        audio_reshaped = a.get('audio').astype('float64')
        label = a.get('label')
        save_audio_as_wav(audio_reshaped, label, track_name, stems_dir)
    print('Stems Saved!')
    return


def get_estimates_openumnix(track: Track, stems_dir: str, sr: int = None, song_name: str = None) -> tuple:
    """
    A function that extracts the separated vocals and accompaniments from the pre-trained open-unmix model and return
    them as a json object
    Args:
        track: A track object that extends the musdb class
        stems_dir: Absolute path to stems directory
        sr: sample rate of the input song (not musdb track)
        song_name: name of the input song (not musdb track)
    Returns:
        estimates_json: A json object that contains the vocals and the accompaniments with its label and audio as keys
        and values pairs
    """
    audio = track if sr is not None else track.audio
    rate = sr if sr is not None else track.rate
    song_name = song_name if sr is not None else track.name
    estimates = predict.separate(
            torch.as_tensor(audio).float(),
            rate=rate,
            targets=['vocals'],
            residual=True,
            device=device
        )
    for target, estimate in estimates.items():
        if target == 'vocals':
            audio_vocals = estimate.detach().cpu().numpy()[0]
        else:
            audio_accompaniments = estimate.detach().cpu().numpy()[0]
    audio = [{'audio': audio_vocals, 'label': 'vocals'}, {'audio': audio_accompaniments, 'label': 'accompaniment'}]
    save_open_unmix_stems(audio=audio, track_name=song_name, stems_dir=stems_dir)
    vocal_estimates = audio_vocals.T
    accompaniment_estimates = audio_accompaniments.T
    estimates_json = {
        'vocals': vocal_estimates,
        'accompaniment': accompaniment_estimates
    }
    return estimates_json, vocal_estimates.T, accompaniment_estimates.T


def evaluate_open_unmix_estimates(track, estimates: dict, estimates_dir: str) -> dict:
    """
    A function that returns the score for an open-unmix estimate, meaning scores of a vocal/accompaniment separation
    according to ISR, SAR, SDR, SIR metrics

    Args:
        track: A track object that extends the musdb class
        estimates: A json object that contains the vocals and the accompaniments with its label and audio as keys and values pairs
        estimates_dir: An absolute path for json estimates to be saved
    Returns:
        scores: The aggregated scores for each track nicely formatted
    """
    scores = museval.eval_mus_track(
        track=track, user_estimates=estimates, output_dir=estimates_dir
    )

    return scores


def add_track_to_results(results, track_scores):
    for score in track_scores:
        results.add_track(score.df)
    return results


def save_separation_scores(results, track_scores, method: list, estimates_dir: str) -> str:
    """
    Function that creates corresponding directories and saves scores as a .csv file for each component and method

    Args:
        results: An museval object that extends the Evalstore class
        track_scores: An array of museval objects that extends the TrackStore class
        method: An array with the title for each method
        estimates_dir: An absolute path to estimates directory
    Returns:
        estimate_csv: A path of the saved estimates csv
    """
    scores_dir = os.path.join(estimates_dir, 'scores')
    if not os.path.isdir(scores_dir):
        os.mkdir(scores_dir)
    for name, score in zip(method, track_scores):
        results.add_track(score.df)
        estimate_csv = f'{scores_dir}/{name}.csv'
        results.save(f'{scores_dir}/{name}.PANDAS')
        methods = museval.MethodStore()
        methods.add_evalstore(results, name=name)
        methods.df.to_csv(rf'{estimate_csv}')
        comparison = methods.agg_frames_tracks_scores()
        print(comparison)
        logging.info(f'Results: \n{comparison}')

    return estimate_csv
