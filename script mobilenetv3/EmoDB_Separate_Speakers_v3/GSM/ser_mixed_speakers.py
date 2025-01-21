import os
import cv2
import time
import json
import joblib
import pandas as pd
import numpy as np

import imageio
import tensorflow as tf
import librosa
import librosa.display
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal.windows import hamming

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle, class_weight
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score

import keras
from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential
from keras.layers import InputLayer, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
# from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.mobilenet_v3 import MobileNetV3Large, preprocess_input

DATASET_DIR = "Datasets/Converted Datasets/EmoDB"
SPECTROGRAM_DIR = "Spectrograms/EmoDB_Separate_Speakers_v3/Converted_Log"
TF_RECORDS_DIR = "TFRecords/EmoDB_Separate_Speakers_v3_MobileNetV3Large"
TF_RECORDS_NAME = "EmoDB_Separate_Speakers_v3_converted_log_MobileNetV3Large.tfrecords"
MODEL_DIR = "Models"
MODEL_NAME = "EmoDB_Separate_Speakers_v3_converted_log_MobileNetV3Large.h5"
NUM_CLASSES = 7
SAMPLE_RATE = 8000
BATCH_SIZE = 32
EPOCHS = 50
RANDOM_SEED = 42


def create_dataframe_emodb():
    EMOTION_DICT_EMODB = {'W': 'anger', 'L': 'boredom', 'E': 'disgust', 'A': 'fear', 'F': 'happiness', 'T': 'sadness',
                          'N': 'neutral'}
    if DATASET_DIR != "Datasets/EmoDB" and DATASET_DIR != "Datasets/Converted Datasets/EmoDB":
        raise Exception(
            "DATASET_DIR must be set to 'Datasets/EmoDB' or 'Datasets/Converted Datasets/EmoDB' for EmoDB dataset")
    file_person, file_gender, file_emotion, file_path = [], [], [], []
    file_list = os.listdir(DATASET_DIR)
    for file in file_list:
        person = int(file[0:2])
        gender = 'male' if person in [3, 10, 11, 12, 15] else 'female'
        emotion = EMOTION_DICT_EMODB[file[5]]
        file_person.append(person)
        file_gender.append(gender)
        file_emotion.append(emotion)
        file_path.append(os.path.join(DATASET_DIR, file))
    file_dict = {'person': file_person, 'gender': file_gender, 'emotion': file_emotion, 'path': file_path}
    emodb_df = pd.DataFrame.from_dict(file_dict)
    return emodb_df


def create_dataframe_emodb_separate_speakers_mixed():
    EMOTION_DICT_EMODB = {'W': 'anger', 'L': 'boredom', 'E': 'disgust', 'A': 'fear', 'F': 'happiness', 'T': 'sadness', 'N': 'neutral'}
    if DATASET_DIR != "Datasets/EmoDB" and DATASET_DIR != "Datasets/Converted Datasets/EmoDB":
        raise Exception("DATASET_DIR must be set to 'Datasets/EmoDB' or 'Datasets/Converted Datasets/EmoDB' for EmoDB dataset")
    file_person, file_gender, file_emotion, file_path = [], [], [], []
    file_list = os.listdir(DATASET_DIR)
    for file in file_list:
        person = int(file[0:2])
        gender = 'male' if person in [3, 10, 11, 12, 15] else 'female'
        emotion = EMOTION_DICT_EMODB[file[5]]
        file_person.append(person)
        file_gender.append(gender)
        file_emotion.append(emotion)
        file_path.append(os.path.join(DATASET_DIR, file))
    file_dict = {'person': file_person, 'gender': file_gender, 'emotion': file_emotion, 'path': file_path}
    emodb_df = pd.DataFrame(file_dict)
    emodb_df_male = emodb_df[emodb_df['gender'] == 'male']
    emodb_df_female = emodb_df[emodb_df['gender'] == 'female']
    male_dfs = [group for _, group in emodb_df_male.groupby('person')]
    female_dfs = [group for _, group in emodb_df_female.groupby('person')]
    from random import shuffle, seed
    seed(RANDOM_SEED)
    shuffle(male_dfs)
    shuffle(female_dfs)
    return male_dfs, female_dfs


def create_dataframe_emoiit():
    if DATASET_DIR != "Datasets/EMO-IIT" and DATASET_DIR != "Datasets/Converted Datasets/EMO-IIT":
        raise Exception(
            "DATASET_DIR must be set to 'Datasets/EMO-IIT' or 'Datasets/Converted Datasets/EMO-IIT' for EMO-IIT dataset")
    file_emotion, file_path = [], []
    emotion_dir_list = os.listdir(DATASET_DIR)
    for emotion_dir in emotion_dir_list:
        file_list = os.listdir(os.path.join(DATASET_DIR, emotion_dir))
        for file in file_list:
            if file.endswith('.wav'):
                file_emotion.append(emotion_dir)
                file_path.append(os.path.join(DATASET_DIR, emotion_dir, file))
    file_dict = {'emotion': file_emotion, 'path': file_path}
    emoiit_df = pd.DataFrame.from_dict(file_dict)
    emoiit_df = pd.DataFrame(shuffle(emoiit_df, random_state=RANDOM_SEED), columns=emoiit_df.columns).reset_index(
        drop=True, inplace=False)
    return emoiit_df


def create_dataframe_emoiit_separate_speakers_mixed():
    genders = ["male", "female"]
    file_emotion, file_speaker_id, file_gender, file_path = [], [], [], []
    for gender in genders:
        for emotion_dir in os.listdir(os.path.join(DATASET_DIR, gender)):
            for file in os.listdir(os.path.join(DATASET_DIR, gender, emotion_dir)):
                if file.endswith(".wav"):
                    speaker_id = file[:5] if file[:4].lower() == "b511" else file[:4].lower()
                    file_emotion.append(emotion_dir)
                    file_speaker_id.append(speaker_id)
                    file_gender.append(gender)
                    file_path.append(os.path.join(DATASET_DIR, gender, emotion_dir, file))
    emoiit_df = pd.DataFrame({'emotion': file_emotion, 'speaker_id': file_speaker_id, 'gender': file_gender, 'path': file_path}).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    emoiit_df_male = emoiit_df[emoiit_df['gender'] == 'male']
    emoiit_df_female = emoiit_df[emoiit_df['gender'] == 'female']

    def process_dataframe_gender_separated(emoiit_df):
        common_speakers = set.intersection(*[set(emoiit_df[emoiit_df['emotion'] == emotion]['speaker_id'].unique()) for emotion in emoiit_df['emotion'].unique() if emotion != 'irritation'])
        common_speakers_list = list(common_speakers)
        emoiit_df_common_speakers = emoiit_df[emoiit_df['speaker_id'].isin(common_speakers_list)]
        emoiit_df_common_speakers_dict = {speaker: emoiit_df_common_speakers[emoiit_df_common_speakers['speaker_id'] == speaker].reset_index(drop=True) for speaker in common_speakers_list}
        emoiit_df_irritation = emoiit_df[emoiit_df['emotion'] == 'irritation'].sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
        num_parts = len(common_speakers_list)
        emoiit_df_irritation_parts = [emoiit_df_irritation[i:i + len(emoiit_df_irritation) // num_parts] for i in range(0, len(emoiit_df_irritation), len(emoiit_df_irritation) // num_parts)]
        return [pd.concat([emoiit_df_common_speakers_dict[common_speakers_list[i]], emoiit_df_irritation_parts[i]], ignore_index=True).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True) for i in range(len(common_speakers_list))]
    
    parts_male = process_dataframe_gender_separated(emoiit_df_male)
    parts_female = process_dataframe_gender_separated(emoiit_df_female)

    min_len_parts_list = min(len(parts_female), len(parts_male))
    # sort by length of parts
    parts_male = sorted(parts_male, key=len, reverse=True)[:min_len_parts_list]
    parts_female = sorted(parts_female, key=len, reverse=True)[:min_len_parts_list]
    
    from random import shuffle, seed
    seed(RANDOM_SEED)
    shuffle(parts_male)
    shuffle(parts_female)

    return parts_male, parts_female

def create_dataframe_ravdess():
    emotion_list = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
    if DATASET_DIR != "Datasets/RAVDESS/audio_speech_actors_01-24" and DATASET_DIR != "Datasets/Converted Datasets/RAVDESS":
        raise Exception(
            "DATASET_DIR must be set to 'Datasets/RAVDESS/audio_speech_actors_01-24' or 'Datasets/Converted Datasets/RAVDESS' for RAVDESS dataset")
    file_person, file_gender, file_emotion, file_intensity, file_path = [], [], [], [], []
    person_dir_list = os.listdir(DATASET_DIR)
    for person_dir in person_dir_list:
        if person_dir.startswith("Actor_"):
            person = int(person_dir.split("_")[1])
            file_list_person = os.listdir(os.path.join(DATASET_DIR, person_dir))
            for file in file_list_person:
                if file.endswith(".wav"):
                    file_person.append(person)
                    file_path.append(os.path.join(DATASET_DIR, person_dir, file))
                    file_gender.append("male" if person % 2 == 1 else "female")
                    file_emotion.append(emotion_list[int(file.split("-")[2]) - 1])
                    file_intensity.append(int(file.split("-")[3].split(".")[0]))
    file_dict = {'person': file_person, 'gender': file_gender, 'emotion': file_emotion, 'intensity': file_intensity,
                 'path': file_path}
    ravdess_df = pd.DataFrame(file_dict)
    ravdess_df = pd.DataFrame(shuffle(ravdess_df, random_state=RANDOM_SEED), columns=ravdess_df.columns).reset_index(
        drop=True, inplace=False)
    return ravdess_df


def preprocess_dataset(ser_df, dataset_type, ohe=None, fold=None):
    audio_block_list = []
    emotion_list = []
    for row in tqdm(ser_df.itertuples(), desc=f"Preprocessing audio files dataset - {dataset_type}", total=len(ser_df)):
        data, _ = librosa.load(row.path, sr=SAMPLE_RATE)
        if data.shape[0] < SAMPLE_RATE:
            data = np.pad(data, (0, SAMPLE_RATE - data.shape[0]), 'constant')
        frames = librosa.util.frame(data, frame_length=SAMPLE_RATE, hop_length=int(SAMPLE_RATE/100)).T
        for frame in frames:
            audio_block_list.append(frame)
            emotion_list.append(row.emotion)
    audio_block_list = np.array(audio_block_list)
    emotion_list = np.array(emotion_list)
    if ohe is None:
        ohe = OneHotEncoder(categories='auto', sparse=False)
        emotion_list = ohe.fit_transform(emotion_list[:, np.newaxis])
        ohe_path = os.path.join(MODEL_DIR, MODEL_NAME.replace(".h5", f"_fold_{fold + 1}_ohe.pkl")) if fold is not None else os.path.join(MODEL_DIR, MODEL_NAME.replace(".h5", "_ohe.pkl"))
        joblib.dump(ohe, ohe_path)
    else:
        emotion_list = ohe.transform(emotion_list[:, np.newaxis])
    return audio_block_list, emotion_list


def create_spectrogram_log(data, sr):
    X = np.abs(librosa.stft(data, window=hamming(int(np.round(sr / 1000) * 32)), n_fft=int(np.round(sr / 1000) * 32),
                            hop_length=int(np.round(sr / 1000) * 4)))
    Xdb = librosa.amplitude_to_db(np.abs(X), ref=np.max)
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=-90, vmax=-7)
    image = cmap(norm(Xdb))
    image = image[:, :, :3]
    image = cv2.normalize(src=np.flip(image, axis=0), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    return image


def create_spectrogram_linear(data, sr):
    X = np.abs(librosa.stft(data, window=hamming(int(np.round(sr / 1000) * 32)), n_fft=int(np.round(sr / 1000) * 32),
                            hop_length=int(np.round(sr / 1000) * 4)))
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=np.amin(X), vmax=np.amax(X))
    image = cmap(norm(X))
    image = image[:, :, :3]
    image = cv2.normalize(src=np.flip(image, axis=0), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    return image


def create_spectrogram_mel(data, sr):
    Xmel = np.abs(
        librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128, window=hamming(int(np.round(sr / 1000) * 32)),
                                       n_fft=int(np.round(sr / 1000) * 32), hop_length=int(np.round(sr / 1000) * 4)))
    X_mel_db = librosa.amplitude_to_db(Xmel, ref=np.max)
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=-90, vmax=-7)
    image = cmap(norm(X_mel_db))
    image = image[:, :, :3]
    image = cv2.normalize(src=np.flip(image, axis=0), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    return image


def create_spectrogram_hpss(data, sr):
    X = librosa.stft(data, window=hamming(int(np.round(sr / 1000) * 32)), n_fft=int(np.round(sr / 1000) * 32),
                     hop_length=int(np.round(sr / 1000) * 4))
    X_harmonic, X_percussive = librosa.decompose.hpss(X)
    X_db = librosa.amplitude_to_db(np.abs(X), ref=np.max)
    X_harmonic_db = librosa.amplitude_to_db(np.abs(X_harmonic), ref=np.max)
    X_percussive_db = librosa.amplitude_to_db(np.abs(X_percussive), ref=np.max)
    image = np.hstack((X_db, X_harmonic_db, X_percussive_db))
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=-90, vmax=-7)
    image = cmap(norm(image))
    image = image[:, :, :3]
    image = cv2.normalize(src=np.flip(image, axis=0), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    return image


def image_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.encode_png(value).numpy()])
    )


def bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_example_train(image, path, emotion_id):
    feature = {
        "image": image_feature(image),
        "path": bytes_feature(path),
        "emotion_id": float_feature(emotion_id),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def create_example_test(image, path, emotion_id, sample_weight):
    feature = {
        "image": image_feature(image),
        "path": bytes_feature(path),
        "emotion_id": float_feature(emotion_id),
        "sample_weight": float_feature(sample_weight),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_tfrecord_train(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "emotion_id": tf.io.FixedLenFeature([NUM_CLASSES], tf.float32),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_png(example["image"], channels=3)
    return example


def parse_tfrecord_test(example):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "path": tf.io.FixedLenFeature([], tf.string),
        "emotion_id": tf.io.FixedLenFeature([NUM_CLASSES], tf.float32),
        "sample_weight": tf.io.FixedLenFeature([1], tf.float32),
    }
    example = tf.io.parse_single_example(example, feature_description)
    example["image"] = tf.io.decode_png(example["image"], channels=3)
    return example


def create_spectrogram_dataset(audio_block_list, emotion_list, tf_records_name, spectrogram_dir, sr, create_spectrogram,
                               sample_weight=None, dataset_type="train"):
    if dataset_type not in ["train", "dev", "test"]:
        raise ValueError("dataset_type must be 'train', 'dev' or 'test'")
    if not os.path.exists(os.path.join(TF_RECORDS_DIR, dataset_type)):
        os.makedirs(os.path.join(TF_RECORDS_DIR, dataset_type))
    with tf.io.TFRecordWriter(os.path.join(TF_RECORDS_DIR, dataset_type, tf_records_name)) as writer:
        for index, block in enumerate(tqdm(audio_block_list, desc=f"Creating Spectrogram Dataset - {dataset_type}",
                                           total=audio_block_list.shape[0])):
            image = create_spectrogram(block, sr)
            if not os.path.exists(os.path.join(spectrogram_dir, dataset_type)):
                os.makedirs(os.path.join(spectrogram_dir, dataset_type))
            image_path = os.path.join(spectrogram_dir, dataset_type, f"{index:05d}.png")
            imageio.imsave(image_path, image)
            image = tf.io.decode_png(tf.io.read_file(image_path))
            os.remove(image_path)
            if dataset_type == "train":
                example = create_example_train(image, image_path, emotion_list[index])
            else:
                if sample_weight is None:
                    raise ValueError("sample_weight must be provided for test dataset")
                else:
                    example = create_example_test(image, image_path, emotion_list[index],
                                                  np.expand_dims(sample_weight[index], axis=0))
            writer.write(example.SerializeToString())


def prepare_sample_train(features):
    image = preprocess_input(tf.cast(features["image"], tf.float32))
    return image, features["emotion_id"]


def prepare_sample_test(features):
    image = preprocess_input(tf.cast(features["image"], tf.float32))
    sample_weight = tf.squeeze(features["sample_weight"])
    return image, features["emotion_id"], sample_weight


def get_dataset(filename, batch_size, dataset_type="train"):
    if dataset_type not in ["train", "dev", "test"]:
        raise ValueError("dataset_type must be 'train', 'dev' or 'test'")
    AUTOTUNE = tf.data.AUTOTUNE
    if dataset_type == "train":
        dataset = (
            tf.data.TFRecordDataset(filename, num_parallel_reads=AUTOTUNE)
            .map(parse_tfrecord_train, num_parallel_calls=AUTOTUNE)
            .map(prepare_sample_train, num_parallel_calls=AUTOTUNE)
            .shuffle(batch_size * 10, seed=RANDOM_SEED)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
        )
    else:
        dataset = (
            tf.data.TFRecordDataset(filename, num_parallel_reads=AUTOTUNE)
            .map(parse_tfrecord_test, num_parallel_calls=AUTOTUNE)
            .map(prepare_sample_test, num_parallel_calls=AUTOTUNE)
            .shuffle(batch_size * 10, seed=RANDOM_SEED)
            .batch(batch_size)
            .prefetch(AUTOTUNE)
        )
    return dataset


def create_ser_model():
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # vgg16 = VGG16(weights="imagenet")
        # model = Model(inputs=vgg16.input, outputs=Dense(NUM_CLASSES, activation="softmax", name="emotion")(vgg16.get_layer("fc2").output))
        mobilenetv3_large = MobileNetV3Large(weights="imagenet")
        model = Model(inputs=mobilenetv3_large.input, outputs=Dense(NUM_CLASSES, activation="softmax", name="emotion")(
            mobilenetv3_large.layers[-2].output))
        optimizer = tf.optimizers.SGD(learning_rate=0.0001, decay=0.0001, momentum=0.9)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'], weighted_metrics=['accuracy'])
        return model


def plot_history(history, model_name):
    fig, axs = plt.subplots(2, 1, figsize=(15, 15))
    fig.suptitle(model_name, size=20)
    axs[0].plot(history['loss'])
    axs[0].title.set_text('Training Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[1].plot(history['accuracy'])
    axs[1].title.set_text('Training Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    plt.show()


def plot_training_curve(model_name):
    history_json = json.load(open(os.path.join(MODEL_DIR, model_name.replace(".h5", ".json")), "r"))
    plt.figure(figsize=(9, 6))
    plt.plot(history_json['accuracy'])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()
    plt.figure(figsize=(9, 6))
    plt.plot(history_json['weighted_accuracy'])
    plt.xlabel("Epochs")
    plt.ylabel("Weighted Accuracy")
    plt.show()
    plt.figure(figsize=(9, 6))
    plt.plot(history_json['loss'])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()


def get_run_logdir(root_logdir):
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


# maxim 4
FOLD_TO_TRAIN = 4
ser_df_male_parts, ser_df_female_parts = create_dataframe_emodb_separate_speakers_mixed()
assert FOLD_TO_TRAIN < len(ser_df_male_parts), f"FOLD_TO_TRAIN must be less than the number of folds, which is {len(ser_df_male_parts)}!"
strategy = tf.distribute.MirroredStrategy()
num_gpus = strategy.num_replicas_in_sync
genders = ["male", "female"]
for fold, (test_df_male, test_df_female) in enumerate(zip(ser_df_male_parts, ser_df_female_parts)):
    print(f"\nFold {fold + 1}:")
    test_df_dict = {"male": test_df_male, "female": test_df_female}
    train_dfs = [pd.concat([df_male, df_female], ignore_index=True) for idx, (df_male, df_female) in enumerate(zip(ser_df_male_parts, ser_df_female_parts)) if idx != fold]
    train_df = pd.concat(train_dfs).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    audio_block_list_train, emotion_list_train = preprocess_dataset(train_df, "train", fold=fold)
    new_model_name = MODEL_NAME.replace(".h5", f"_fold_{fold + 1}.h5")
    ohe = joblib.load(os.path.join(MODEL_DIR, MODEL_NAME.replace(".h5", f"_fold_{fold + 1}_ohe.pkl")))
    cls_weight = class_weight.compute_class_weight(class_weight='balanced', classes=ohe.categories_[0], y=ohe.inverse_transform(emotion_list_train).flatten())
    cls_weight_dict = dict(zip(ohe.categories_[0], cls_weight))
    new_tfrecords_name = TF_RECORDS_NAME.replace(".tfrecords", f"_fold_{fold + 1}.tfrecords")
    new_spectrogram_dir = SPECTROGRAM_DIR + f"_fold_{fold + 1}"
    if not os.path.exists(new_spectrogram_dir):
        os.makedirs(new_spectrogram_dir)
        os.makedirs(os.path.join(new_spectrogram_dir, "train"))

    # de comentat
    # create_spectrogram_dataset(audio_block_list_train, emotion_list_train, new_tfrecords_name, new_spectrogram_dir, sr=SAMPLE_RATE, create_spectrogram=create_spectrogram_log, dataset_type="train")
    # del audio_block_list_train, emotion_list_train

    # de comentat
    if fold == FOLD_TO_TRAIN:
        create_spectrogram_dataset(audio_block_list_train, emotion_list_train, new_tfrecords_name, new_spectrogram_dir, sr=SAMPLE_RATE, create_spectrogram=create_spectrogram_log, dataset_type="train")
        del audio_block_list_train, emotion_list_train

    new_tfrecords_name_test_dict = {}
    for gender in genders:
        audio_block_list_test, emotion_list_test = preprocess_dataset(test_df_dict[gender], "test", ohe=ohe, fold=fold)
        emotion_list_test_unique = test_df_dict[gender]["emotion"].unique()
        cls_weight_dict = {emotion: cls_weight_dict[emotion] for emotion in emotion_list_test_unique}
        test_sample_weight = class_weight.compute_sample_weight(class_weight=cls_weight_dict, y=ohe.inverse_transform(emotion_list_test).flatten())
        new_tfrecords_name_test = TF_RECORDS_NAME.replace(".tfrecords", f"_fold_{fold + 1}_{gender}.tfrecords")
        new_tfrecords_name_test_dict[gender] = new_tfrecords_name_test
        new_spectrogram_dir_test = SPECTROGRAM_DIR + f"_fold_{fold + 1}_{gender}"
        if not os.path.exists(new_spectrogram_dir_test):
            os.makedirs(new_spectrogram_dir_test)
            os.makedirs(os.path.join(new_spectrogram_dir_test, "test"))

        # de comentat
        # create_spectrogram_dataset(audio_block_list_test, emotion_list_test, new_tfrecords_name_test, new_spectrogram_dir_test, sr=SAMPLE_RATE, create_spectrogram=create_spectrogram_log, sample_weight=test_sample_weight, dataset_type="test")
        # del audio_block_list_test, emotion_list_test

        # de comentat
        if fold == FOLD_TO_TRAIN:
            create_spectrogram_dataset(audio_block_list_test, emotion_list_test, new_tfrecords_name_test, new_spectrogram_dir_test, sr=SAMPLE_RATE, create_spectrogram=create_spectrogram_log, sample_weight=test_sample_weight, dataset_type="test")
            del audio_block_list_test, emotion_list_test

    if fold == FOLD_TO_TRAIN:
        train_dataset = get_dataset(os.path.join(TF_RECORDS_DIR, "train", new_tfrecords_name), BATCH_SIZE * num_gpus, dataset_type="train")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)
        model = create_ser_model()
        run_logdir = get_run_logdir(root_logdir=os.path.join(os.curdir, "Logs", new_model_name.replace(".h5", "")))
        tensorboard_cb = tf.keras.callbacks.TensorBoard(run_logdir)
        checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, new_model_name), verbose=1, monitor='accuracy', save_best_only=True, mode='auto')
        history = model.fit(x=train_dataset, class_weight=dict(enumerate(cls_weight)), epochs=EPOCHS, verbose=1, callbacks=[tensorboard_cb, checkpoint], shuffle=False, workers=8, use_multiprocessing=True)
        json.dump(history.history, open(os.path.join(MODEL_DIR, new_model_name.replace(".h5", ".json")), "w"))
        plot_history(history.history, model_name=f"Fold {fold + 1} - {new_model_name.replace('.h5', '')}")
        plot_training_curve(new_model_name)

        model = load_model(os.path.join(MODEL_DIR, new_model_name))
        for gender in genders:
            test_dataset = get_dataset(os.path.join(TF_RECORDS_DIR, "test", new_tfrecords_name_test_dict[gender]), BATCH_SIZE, dataset_type="test")
            results_test = model.evaluate(test_dataset, workers=12, use_multiprocessing=True)
            print(f"FOLD {fold + 1} - {gender}: Test accuracy: {results_test[1]:0.2%}\nTest weighted accuracy: {results_test[2]:0.2%}\nTest loss: {results_test[0]}")
