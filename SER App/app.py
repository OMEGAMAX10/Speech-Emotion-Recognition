import keras
import socket
import numpy as np
from time import sleep, time
from threading import Thread
from utils import bytes_to_sound, preprocess_wav, predict, get_best_emotion

HOST = "0.0.0.0"
PORT = 5000
SAMPLE_RATE = 8000
HOP_LENGTH_SECONDS = 0.01
MAX_LENGTH_CONSECUTIVES_PERCENTAGE = 0.1
HOP_LENGTH = int(SAMPLE_RATE * HOP_LENGTH_SECONDS)
WAV_HEADER_SIZE = 44
MODEL_PATH = './Models/EMO-IIT_converted_log_MobileNetV3Large.h5'
model = keras.models.load_model(MODEL_PATH)


class ClientThread(Thread):
    def __init__(self, ip, port, conn):
        Thread.__init__(self)
        self.ip = ip
        self.port = port
        self.conn = conn
        print("[+] New server socket thread started for " + ip + ":" + str(port))

    def run(self):
        with self.conn:
            start_chrono = time()
            print("\nConnection from : " + self.ip + ":" + str(self.port))
            # self.conn.send(str.encode(f'Conection established with {self.ip}:{self.port}\n'))
            audio_content = b''
            second_for_processing = 1
            emotion_list = []
            emotion_list_onesec_length = np.ceil(1 / HOP_LENGTH_SECONDS).astype(int)
            sleep(0.1)
            while True:
                data = self.conn.recv(4096)
                if not data:
                    if len(audio_content) < SAMPLE_RATE * 0.4:
                        self.conn.close()
                        break
                    sound_data = bytes_to_sound(audio_content[WAV_HEADER_SIZE:])
                    spectrogram_list = preprocess_wav(sound_data[SAMPLE_RATE * (second_for_processing - 1):], SAMPLE_RATE, HOP_LENGTH)
                    for idx in range(0, len(spectrogram_list), emotion_list_onesec_length):
                        emotions = predict(model, np.array(spectrogram_list[idx: idx + emotion_list_onesec_length]))
                        emotion_list.extend(emotions)
                        if (len(emotion_list) % emotion_list_onesec_length == 0 and len(emotion_list) // emotion_list_onesec_length > 0) or (idx + emotion_list_onesec_length >= len(spectrogram_list)):
                            response = f"{second_for_processing - 1} {second_for_processing} {get_best_emotion(emotion_list[-emotion_list_onesec_length:], max_length_consecutives=int(MAX_LENGTH_CONSECUTIVES_PERCENTAGE / HOP_LENGTH_SECONDS))}\n"
                            self.conn.sendall(response.encode())
                            print(response, end='')
                            # print(emotion_list[-emotion_list_onesec_length:])
                            stop_chrono = time()
                            print(f"Time elapsed: {stop_chrono - start_chrono : .4f} seconds\n")
                            start_chrono = time()
                            second_for_processing += 1
                    self.conn.close()
                    break
                audio_content += data
                audio_length = SAMPLE_RATE * (second_for_processing + 1) * 2 - HOP_LENGTH + WAV_HEADER_SIZE
                if len(audio_content) >= audio_length:
                    sound_data = bytes_to_sound(audio_content[WAV_HEADER_SIZE:audio_length])
                    spectrogram_list = preprocess_wav(sound_data[SAMPLE_RATE * (second_for_processing - 1): SAMPLE_RATE * (second_for_processing + 1)], SAMPLE_RATE, HOP_LENGTH)
                    emotions = predict(model, spectrogram_list)
                    emotion_list.extend(emotions)
                    response = f"{second_for_processing - 1} {second_for_processing} {get_best_emotion(emotion_list[-emotion_list_onesec_length:], max_length_consecutives=int(MAX_LENGTH_CONSECUTIVES_PERCENTAGE / HOP_LENGTH_SECONDS))}\n"
                    self.conn.sendall(response.encode())
                    print(response, end='')
                    # print(emotion_list[-emotion_list_onesec_length:])
                    stop_chrono = time()
                    print(f"Time elapsed: {stop_chrono - start_chrono : .4f} seconds\n")
                    start_chrono = time()
                    second_for_processing += 1


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen(10)
    while True:
        print("\nListening on port: ", PORT)
        connection, addr = s.accept()
        client_thread = ClientThread(addr[0], addr[1], connection)
        client_thread.start()
