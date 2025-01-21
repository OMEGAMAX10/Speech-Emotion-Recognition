import cv2
import librosa
import itertools
import numpy as np
import tensorflow as tf
from statistics import mode
from matplotlib import pyplot as plt
from scipy.signal.windows import hamming


EMOTION_LIST = ['anger', 'boredom', 'disgust', 'fear', 'hapiness', 'irritation', 'neutral', 'sadness']
THRESHOLD_AE = 0.25

def bytes_to_sound(data):
    sound_data = np.frombuffer(data, dtype=np.int16, count=len(data) // 2).astype(np.float64) / 32768.0
    return sound_data


def frames(data, sample_rate, hop_length):
    audio_block_list = []
    if data.shape[0] < sample_rate:
        data = np.pad(data, (0, sample_rate - data.shape[0]), 'constant')
    frames = librosa.util.frame(data, frame_length=sample_rate, hop_length=hop_length).T
    for frame in frames:
        audio_block_list.append(frame)
    return np.array(audio_block_list)


def create_spectrogram_log(data, sample_rate):
    X = np.abs(librosa.stft(data, window=hamming(int(np.round(sample_rate / 1000) * 32)), n_fft=int(np.round(sample_rate / 1000) * 32), hop_length=int(np.round(sample_rate / 1000) * 4)))
    Xdb = librosa.amplitude_to_db(np.abs(X), ref=np.max)
    cmap = plt.cm.jet
    norm = plt.Normalize(vmin=-90, vmax=-7)
    image = cmap(norm(Xdb))
    image = image[:, :, :3]
    image = cv2.normalize(src=np.flip(image, axis=0), dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    return image


def preprocess_wav(data, sample_rate, hop_length):
    audio_block_list = frames(data, sample_rate, hop_length)
    spectrogram_list = np.array([create_spectrogram_log(block, sample_rate) for block in audio_block_list])
    return spectrogram_list


def predict(model, spectrogram_list):
    preds = model.predict(spectrogram_list)
    preds = preds.argmax(axis=1)
    preds = [EMOTION_LIST[pred] for pred in preds]
    return preds


def preprocess_input_autoencoder(img):
    img = img / 255.0
    return np.float32(img)


def SSIMLoss_batch(y_true, y_pred):
    return 1 - tf.image.ssim(y_true, y_pred, 1.0)


def predict_autoencoder(model_ae, model_ser, spectrogram_list):
    image_true_ae = preprocess_input_autoencoder(spectrogram_list)
    reconstructed_pred = model_ae.predict(image_true_ae, verbose=0)
    anomaly_score = SSIMLoss_batch(image_true_ae, reconstructed_pred).numpy()
    preds = model_ser.predict(spectrogram_list, verbose=0)
    preds = np.argmax(preds, axis=1)
    anomaly_idx = np.where(anomaly_score < THRESHOLD_AE)[0]
    preds[anomaly_idx] = 6  # neutral index in ohe vector
    preds = [EMOTION_LIST[pred] for pred in preds]
    return preds


def longest_sequence(lst):
    if not lst:
        return []
    longest = max(list(map(lambda x: list(x[1]), itertools.groupby(lst))), key=len)
    return longest


def get_best_emotion(emotion_list, max_length_consecutives):
    # emotion = mode(emotion_list)
    # print(emotion)
    consecutives = longest_sequence(emotion_list)
    if len(consecutives) >= max_length_consecutives:
        return consecutives[0]
    return mode(emotion_list)
