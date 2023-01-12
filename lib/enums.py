from enum import Enum


class Method(Enum):
    open_unmix = 'OPENUNMIX'
    ground_truth = 'GT'
    my_separator = 'CNN_DEEP_SEPARATOR'
    librosa_separator = 'REPET-SIM'
    spleeter = 'SPLEETER'


class Categories(Enum):
    vocals = 'Vocals'
    accompaniment = 'Accompaniments'


class Tools(Enum):
    open_unmix = '1'
    spleeter = '2'
    my_separator = '3'
    librosa_separator = '4'
    all = 'All'


class Modes(Enum):
    acapella = 'acapella'
    respiratory = 'respiratory'
    voice = 'voice'


class Tools_resp(Enum):
    dnn_lstm = '1'
    nn_filtering = '2'


class Paths(Enum):
    VAD_ROOT_DIR = 'vocal_activity_detection'
    SEPARATOR_ROOT_DIR = 'source_separation_model'
    AE = 'auto_encoder'
    MIXTURE_DIR = 'mixture_spectrogram'
    VOCALS_DIR = 'vocals_spectrogram'
    NON_VOCALS_DIR = 'non-vocals_spectrogram'
    VOCAL_MASKS_DIR = 'vocals_masks'
    SEPARATOR_MODEL = 'separator_model'
    VAD_MODEL = 'vad_model'
    AE_MODEL = 'ae_model'
    MY_MODEL_SEPARATOR = 'testModel4'
    MY_VAD_MODEL = 'model_test_1.4_100_128_513x25'


class ArchitectureMode(Enum):
    # SingingVoiceSeparation
    svs = 'SVS'
    # VocalActivityDetection
    vad = 'VAD'
    # AutoEncoder
    dae = 'DAE'