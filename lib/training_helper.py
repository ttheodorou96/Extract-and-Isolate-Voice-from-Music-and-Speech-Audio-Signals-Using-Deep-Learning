import os

from .data_loader import DataLoader
from .enums import Paths, ArchitectureMode
from .helper_funcs import yaml_deconstruct
from .models import Models

root_dir = os.getcwd()


def train_model(my_data: dict) -> None:
    """
    Function gets that initiates the preprocess and training pipeline. Calls corresponding functions and classes to
    create or get the needed paths, load the training data split it into train/validation sets and finally initiates
    the model training process given the parameters the user specified while running

    Args:
        my_data: A dictionary that contains data from the loaded configuration file and the argument parser
    Returns:
        None
    """
    model_name, epochs, batch_size, frame_size, hop_length, win_length, sample_rate, _, img_width, _, \
    dataset_dir, _, _, mode = yaml_deconstruct(my_data)
    img_height = int((frame_size / 2) + 1)
    if mode == ArchitectureMode.svs.value:
        main_directory = Paths.SEPARATOR_ROOT_DIR.value
        model_dir = Paths.SEPARATOR_MODEL.value
    elif mode == ArchitectureMode.dae.value:
        main_directory = Paths.AE.value
        model_dir = Paths.AE_MODEL.value
    else:
        main_directory = Paths.VAD_ROOT_DIR.value
        model_dir = Paths.VAD_MODEL.value

    dataset_path = os.path.join(os.path.join(root_dir, main_directory), dataset_dir)

    mix = Paths.MIXTURE_DIR.value
    accompaniment = Paths.NON_VOCALS_DIR.value
    vocals = Paths.VOCALS_DIR.value
    masks = Paths.VOCAL_MASKS_DIR.value
    model_path = os.path.join(main_directory, model_dir)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    categories = [mix, accompaniment, masks, vocals]
    dataloader = DataLoader(dataset_path=dataset_path,
                            categories=categories,
                            model_path=model_path,
                            model_details=my_data
                            )
    if mode == ArchitectureMode.vad.value:
        dataloader.split_data()
        my_model = Models(height=img_height,
                          width=img_width,
                          model_path=model_path,
                          model_name=model_name,
                          epochs=epochs,
                          batch_size=batch_size
                          )
        my_model.vad_model()
    else:
        X_train, y_train = dataloader.load_training_data(
            0, 2) if mode == ArchitectureMode.svs.value else dataloader.load_training_data(3, 3)
        my_model = Models(height=img_height,
                          width=img_width,
                          model_path=model_path,
                          model_name=model_name,
                          epochs=epochs,
                          batch_size=batch_size,
                          X_train=X_train,
                          y_train=y_train
                          )
        print(y_train.shape[2])
        # check IBM's label shape to build model architecture accordingly
        if y_train.shape[2] == 1:
            # (2049, 1) -> CNN_separator
            my_model.cnn_separator()
        else:
            # (2049, 9) -> CNN_deep_separator
            my_model.deep_cnn_separator() if mode == ArchitectureMode.svs.value else my_model.initiate_svs_as_autoencoder()
    return
