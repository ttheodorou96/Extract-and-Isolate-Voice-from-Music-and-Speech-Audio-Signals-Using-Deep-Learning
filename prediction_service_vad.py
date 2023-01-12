import argparse
import logging
import os

import tensorflow as tf
import pickle
from numpy import newaxis
import random
from lib.enums import Categories


def argParser():
    """
    Argument parser for the Custom musdb dataset.

    Returns:
    -------
    args: Dict
        A dictionary of all the positional and optional arguments specified.
    """
    parser = argparse.ArgumentParser(description='Select directory of model you want to test')
    parser.add_argument('name',
                        type=str,
                        metavar='',
                        help='Model name'
                        )
    args = parser.parse_args()
    return args


def test(selected_model_name):
    """
    Function that test the prediction accuracy of the pre-trained model provided by the user
    Parameters
    ----------
    selected_model_name: str
        The name of the selected model the user choose
    Returns
    -------
         None
    """
    CATEGORIES = [Categories.vocals.value, Categories.accompaniment.value]
    pickle_in = open(f"vocal_activity_detection/vad_model/{selected_model_name}/X_test.pickle", "rb")
    X_test = pickle.load(pickle_in)

    pickle_in = open(f"vocal_activity_detection/vad_model/{selected_model_name}/y_test.pickle", "rb")
    y_test = pickle.load(pickle_in)
    X_test = X_test / 255.0
    count_success = 0
    samples = 100
    for i in range(samples):
        print(i+1, ')')
        a = random.randint(0, len(y_test)-1)
        print(f'number of track #{a} of the list')
        expected_output = CATEGORIES[y_test[a]]
        print("expected output: " + expected_output)
        test_data = X_test[a]
        test_data = test_data[newaxis, :, :, :]
        model = tf.keras.models.load_model(f"vocal_activity_detection/vad_model/{selected_model_name}")
        prediction = model.predict([test_data])
        prediction_output = CATEGORIES[round(prediction[0][0])]
        print("prediction output: " + prediction_output)
        if prediction_output == expected_output:
            count_success += 1
            logging.info(f'{i + 1}) number of track #{a}. Expected output: {expected_output}. Prediction output: '
                         f'{prediction_output}. Successful prediction!')
            print('successful prediction!')
        else:
            logging.info(f'{i + 1}) number of track #{a}. Expected output: {expected_output}. Prediction output: '
                         f'{prediction_output}. Unsuccessful prediction!')
            print('unsuccessful prediction!')
    logging.info(f'successful predictions: {count_success}. Success rate of the model is: {count_success / samples}')
    print(f'successful predictions: {count_success}. Success rate of the model is: {count_success / samples}')
    return


def main():
    args = argParser()
    model_name = args.name
    root_dir = os.getcwd()
    logger_path = os.path.join(root_dir, 'logger')
    prediction_service_logger = 'prediction_service_vad'
    prediction_service_path = os.path.join(logger_path, prediction_service_logger)
    if not os.path.isdir(prediction_service_path):
        os.mkdir(prediction_service_path)
    logger_file_name = os.path.join(prediction_service_path, model_name + '_logger_6.log')
    logging.basicConfig(level=logging.INFO, filename=logger_file_name, filemode="w",
                        format="%(asctime)s - %(levelname)s - %(message)s")
    test(selected_model_name=model_name)


if __name__ == '__main__':
    main()
