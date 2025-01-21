# Speech Emotion Recognition

Speech emotion recognition system using MobileNetV3 convolutional neural network for classifying 7 basic human emotions (anger, boredom, disgust, fear, happiness, sadness, and neutral) from the spectrograms of the voice signal. The system consists of an API deployed on a Docker container, where the user can send an audio "wav" file through the socket created by the container and get the emotions from each 1-second intervals of the file.

## 1. Container Deployment

### a. CPU-only deployment
The container can be deployed and run using the following commands in case the GPU is not used:

```
sudo docker build --tag ser-docker .
sudo docker run -d --name ser-docker -p 5000:5000 ser-docker
```

### b. GPU deployment
The container can be deployed and run using the following commands in case the GPU is used:

```
sudo docker build -f Dockerfile.gpu --tag ser-docker-gpu .
sudo docker run -d --gpus all --name ser-docker-gpu -p 5000:5000 ser-docker-gpu
```

## 2. API Usage

The API can be used by sending a file through the socket that is opened by the server by using the following netcat (nc) command on Linux:

```
cat test.wav | nc -q1 localhost 5000
```

The API will return, in almost real time, a list of 1-second intervals with the predominant emotion from it with the following structure:

```
0 1 sadness
1 2 sadness
2 3 anger
3 4 happiness
4 5 sadness
5 6 sadness
6 7 happiness
...
```

## 3. Model Used

The model used for speech emotion recognition is a MobileNetV3 convolutional neural network, trained on the [EMO-IIT dataset](http://www.dasconference.ro/dvd2016/data/papers/D36-paper.pdf) which was converted to GSM format (sampling rate of 8kHz), and can be downloaded from [here](https://mega.nz/folder/KhxRTawZ#LNyigs6v10Ij6WMvPyuYHw).
