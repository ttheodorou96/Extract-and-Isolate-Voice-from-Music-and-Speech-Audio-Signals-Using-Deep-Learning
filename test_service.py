import argparse
import logging
import sys

import librosa
import musdb
import numpy as np
import museval
import os
from lib.separate_helper_funcs import model_predictions, evaluate_SDR_format, eval_mus_track
from lib.helper_funcs import yaml_loader, yaml_deconstruct
from lib.enums import Method
import pandas as pd
from lib.estimates_scores import add_track_to_results, estimate_scores_user_reference

root_dir = os.getcwd()
cf_path = "configuration_files/vocal_separator_tool/init_separator_tool.yaml"
config = yaml_loader(filepath=cf_path)
db_path = os.path.join(root_dir, 'musdb18')
mus = musdb.DB(root=db_path, subsets='test')

_, _, _, n_fft, hop_length, _, sr, _, img_width, _, _, _, _, _ = yaml_deconstruct(config)
SONG_TEST_DURATION = 60


def check_NaN(dataframe):
    """
         Filter by target = `vocals` AND metric = `SDR` to get the score.
    """
    # filter by target = `vocals` AND metric = `SDR` to get the score
    x = dataframe[(dataframe['target'] == 'vocals') & dataframe['metric'].isin(['SDR'])]
    print(x, isinstance(x, pd.DataFrame))
    y = x.at['0', 'score']
    return y


def my_model_test(evaluation_path):
    """
        Get prediction and evaluation results for each song in musdb18 test set.
    """
    n_samples = round(float(SONG_TEST_DURATION) * sr / hop_length)
    methods = museval.MethodStore(frames_agg='median', tracks_agg='median')
    for i, track in enumerate(mus):
        predicted_soft_mask, input_mix, acc, my_track = model_predictions(mus,
                                                                          i,
                                                                          hop_length,
                                                                          sr,
                                                                          n_samples,
                                                                          n_fft)
        S_vocals = np.multiply(input_mix, predicted_soft_mask)
        S_acc = np.multiply(input_mix, acc)
        method = [Method.my_separator.value]
        audio_reference = {
            'vocals': evaluate_SDR_format(my_track['vocals'], sr, my_track['rate']),
            'accompaniment': evaluate_SDR_format(my_track['accompaniment'], sr, my_track['rate'])
        }
        audio_estimates = {
            'vocals': evaluate_SDR_format(S_vocals, sr, my_track['rate']),
            'accompaniment': evaluate_SDR_format(S_acc, sr, my_track['rate'])
        }
        track_scores, _ = eval_mus_track(
            user_reference=audio_reference, user_estimates=audio_estimates, track=my_track, output_dir=evaluation_path
        )
        results = museval.EvalStore()
        track_scores = [track_scores]
        scores_dir = os.path.join(evaluation_path, 'scores')
        if not os.path.isdir(scores_dir):
            os.mkdir(scores_dir)
        for name, score in zip(method, track_scores):
            results.add_track(score.df)
            print(results)
            methods.add_evalstore(results, name='CNN_DEEP_SEPARATOR')

    estimate_csv = f'{scores_dir}/sigsep_comparison_deep_separator.csv'
    results.save(f'{scores_dir}/sigsep_comparison_deep_separator.PANDAS')
    methods.add_sisec18()
    methods.df.to_csv(rf'{estimate_csv}')
    comparison = methods.agg_frames_tracks_scores()
    print(comparison)


def main():
    """
        Testing prediction service, extracts the scores of each song in testing set. Get the evaluation
        results for each frame and estimate mean score. Append our method into the museval object together with all
        sigsep18 evaluation test contestants. Save csv and json files in evaluation folder. Finally the .csv file can be
        used for boxplot comparisons on each metric.
    """
    evaluation_path = os.path.join(root_dir, 'evaluation')
    if not os.path.isdir(evaluation_path):
        os.mkdir(evaluation_path)
    my_model_test(evaluation_path)


if __name__ == '__main__':
    main()
