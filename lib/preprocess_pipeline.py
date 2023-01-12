import librosa
import numpy as np
import os
import librosa.display
import cv2
from numpy import newaxis
import matplotlib.pyplot as plt


# depricated
class LogSpectrogramExtract:
    """
    The LogSpectrogramExtract class to extract the log magnitude (in dB) of an audio time-series signal

    Parameters
    ----------
    frame_size: int
        Sets the length of the windowed signal after padding with zeros.
    hop_length: int
        Sets number of audio samples between adjacent STFT columns.

    Methods
    -------
    transformation()
        Method that takes as a parameter an audio time-series signal and returns the discrete Fourier Transform
    """
    def __init__(self, frame_size, hop_length):
        """
        Class constructor takes as parameters the window size and hop length to compute the Short-time Fourier
        transform of an input's signal.

        Parameters
        ----------
        frame_size: int
            Sets length of the windowed signal after padding with zeros.
        hop_length: int
            Sets number of audio samples between adjacent STFT columns.
        """
        self.frame_size = frame_size
        self.hop_length = hop_length

    def transformation(self, signal_input):
        """
        Function that takes as a parameter an audio time-series signal and applies the FFT algorithm to compute its
        corresponding discrete Fourier Transform.

        Parameters
        ----------
        signal_input: np.ndarray
            Input signal, an audio time-series sequence.
        Returns
        -------
        log_spectrogram: np.ndarray
            A discrete representation of the computed input signal on a logarithmic
            frequency scale.
        spectrogram: np.ndarray
            A complex-valued matrix of short-term Fourier transform coefficients.
        """
        stft = librosa.stft(librosa.to_mono(signal_input), n_fft=self.frame_size, hop_length=self.hop_length)
        # spectrogram = np.abs(stft)
        # log_spectrogram = librosa.power_to_db(spectrogram)

        return stft


class BinaryMaskExtract:
    """
    BinaryMaskExtract class to extract the binary mask of each track's vocal component

    Parameters
    ----------
    eps: float
        Machine epsilon to give an upper bound on the relative approximation error.

    Methods
    -------
    mask_compute()
        Function that takes the fourier transformation of the accompaniments and the vocals signal and computes
        the corresponding binary mask, returns a discrete representation of the computed mask signal on a logarithmic
        frequency scale.
    """
    def __init__(self, eps):
        """
        Class constructor takes as a parameter machine epsilon to give an upper bound on the relative approximation
        error of binary mask computation due to rounding in floating point arithmetic.

        Parameters
        ----------
        eps: float
            Machine epsilon to give an upper bound on the relative approximation error.
        """
        self.eps = eps

    def mask_compute(self, s_accompaniment, s_vocals, separator_flag):
        """
        Function that takes the fourier transformation of the accompaniments and the vocals signal and computes
        the corresponding binary mask.

        Parameters
        ----------
        s_accompaniment: np.ndarray
            Short time fourier transform of the accompaniments signal.
        s_vocals: np.ndarray
            Short time fourier transform of the vocals signal.
        separator_flag: bool
            separator flag whether the tool is run for the separation model.

        Returns
        -------
        log_vocals_mask: np.ndarray
            A discrete representation of the computed mask signal on a logarithmic
            frequency scale.
        """
        # if separator_flag is not None:
        #     s_vocals_center_frame = crop_center(s_vocals, 1, 513)
        #     s_accompaniment_center_frame = crop_center(s_accompaniment, 1, 513)
        #     model = self.eps + np.abs(s_vocals_center_frame) + np.abs(s_accompaniment_center_frame)
        #     vocals_mask = np.divide(np.abs(s_vocals_center_frame), model)
        #     vocal_masks_array.append(vocals_mask)
        #     log_vocals_mask = librosa.amplitude_to_db(np.abs(vocals_mask), ref=np.max)
        # else:
        model = self.eps + np.abs(s_accompaniment) + np.abs(s_vocals)
        vocals_mask = np.divide(np.abs(s_vocals), model)
        log_vocals_mask = librosa.amplitude_to_db(np.abs(vocals_mask), ref=np.max)
        # vocal_mask_img = librosa.display.specshow(log_vocals_mask)

        return log_vocals_mask


class SaveResults:
    """
    SaveResults class to save extracted features.

    Attributes
    ----------
    specs_save_dir: str
        Sets Mixture's signal absolute path
    non_vox_save_dir: str
        Sets accompaniment's signal absolute path
    masks_save_dir: str
        Sets binary mask's signal absolute path

    Methods
    -------
    save_spectrograms()
        Function that saves the discrete representation of mixture signals on a logarithmic frequency scale
        as an image file.
    save_non_vocals()
        Function that saves the discrete representation of non vocal signals on a logarithmic frequency scale
        as an image file.
    save_masks()


    """
    def __init__(self, specs_save_dir, non_vox_save_dir, masks_save_dir, img_width, img_height):
        """
        Class constructor takes as parameters the respective save directories paths.
        Parameters
        ----------
        specs_save_dir: str
            Sets Mixture's signal absolute path
        non_vox_save_dir: str
            Sets accompaniment's signal absolute path
        masks_save_dir: str
            Sets binary mask's signal absolute path
        """
        self.specs_save_dir = specs_save_dir
        self.non_vox_save_dir = non_vox_save_dir
        self.masks_save_dir = masks_save_dir
        self.img_width = img_width
        self.img_height = img_height

    def save_spectrograms(self, feature, song_name):
        """
        Function that saves the discrete representation of mixture signals on a logarithmic frequency scale
        as an image file.
        Parameters
        ----------
        feature: np.ndarray
            Each mixture's signal discrete representation on a logarithmic frequency scale
        song_name: str
            Each mixture's song name
        Returns
        -------
            None
        """
        name = os.path.join(self.specs_save_dir, f'{song_name}.png')
        fig = plt.figure(figsize=(self.img_width / 100, self.img_height / 100))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(feature), ref=np.max), cmap='gray_r', ax=ax)
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.savefig(name, dpi=100)
        plt.close()
        # img = feature[:, :, newaxis]
        # cv2.imwrite(os.path.join(self.specs_save_dir, song_name + '.png'), img)

    def save_non_vocals(self, feature, song_name):
        """
        Function that saves the discrete representation of non vocal signals on a logarithmic frequency scale
        as an image file.

        Parameters
        ----------
        feature: np.ndarray
            Each non vocal's signal discrete representation on a logarithmic frequency scale
        song_name: str
            Each non vocal's song name

        Returns
        -------
            None
        """
        name = os.path.join(self.non_vox_save_dir, f'{song_name}.png')
        fig = plt.figure(figsize=(self.img_width / 100, self.img_height / 100))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(feature), ref=np.max), cmap='gray_r', ax=ax)
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.savefig(name, dpi=100)
        plt.close()
        img = feature[:, :, newaxis]
        cv2.imwrite(os.path.join(self.non_vox_save_dir, song_name + '.png'), img)

    def save_masks(self, feature, song_name):
        """
        Function that saves the discrete representation of binary mask signals on a logarithmic frequency scale
        as an image file.

        Parameters
        ----------
        feature: np.ndarray
            Each binary mask's signal discrete representation on a logarithmic frequency scale
        song_name: str
            Each binary mask's song name
        Returns
        -------
            None
        """
        name = os.path.join(self.masks_save_dir, f'{song_name}.png')
        fig = plt.figure(figsize=(1 / 100, self.img_height / 100))
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        librosa.display.specshow(librosa.amplitude_to_db(np.abs(feature), ref=np.max), cmap='gray_r', ax=ax)
        ax.set_axis_off()
        fig.add_axes(ax)
        plt.savefig(name, dpi=100)
        plt.close()
