import pyaudio
import librosa
import numpy as np
import requests

# ring buffer will keep the last 2 seconds worth of audio
ringBuffer = RingBuffer(2 * 22050)

def callback(in_data, frame_count, time_info, flag):
    audio_data = np.fromstring(in_data, dtype=np.float32)
    
    # we trained on audio with a sample rate of 22050 so we need to convert it
    audio_data = librosa.resample(audio_data, 44100, 22050)
    ringBuffer.extend(audio_data)

    # machine learning model takes wavform as input and
    # decides if the last 2 seconds of audio contains a goal
    if model.is_goal(ringBuffer.get()):
        # GOAL!! Trigger light show
        requests.get("http://127.0.0.1:8082/goal")

    return (in_data, pyaudio.paContinue)

# function that finds the index of the Soundflower
# input device and HDMI output device
dev_indexes = findAudioDevices()

stream = pa.open(format = pyaudio.paFloat32,
                 channels = 1,
                 rate = 44100,
                 output = True,
                 input = True,
                 input_device_index = dev_indexes['input'],
                 output_device_index = dev_indexes['output'],
                 stream_callback = callback)

# start the stream
stream.start_stream()

while stream.is_active():
    sleep(0.25)

stream.close()
pa.terminate()