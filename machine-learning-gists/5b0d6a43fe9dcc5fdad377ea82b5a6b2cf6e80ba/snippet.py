import tensorflow as tf # >= 2.0.0-alpha0
import librosa
import numpy as np
import time
from os import listdir
from os.path import isfile, join
from os.path import splitext

tf.config.gpu.set_per_process_memory_fraction(0.9)

# Preprocessing parameters
sr = 44100 # Sampling rate
duration = 5
hop_length = 347 # to make time steps 128
fmin = 20
fmax = sr // 2
n_mels = 128
n_fft = n_mels * 20
samples = sr * duration

def get_melspec(signals=None, sample_rate=44100, n_mels=128, win_length=None, hop_length=512, n_fft=1024, fmax=8000, fmin=80, power=2.0, window=tf.signal.hann_window, pad_end=True):
    """Compute the Mel-spectrum for audio file
    
    Keyword Arguments:        
        signals {tensor} -- input signals as tensor or np.array in float32 type (default: {None})
        sample_rate {int} -- sampling rate (default: {44100})
        win_length {int} -- frame length to compute STFT (default: {1024})
        frame_step {int} -- frame step to compute STFT (default: {512})
        fft_length {int} -- FFT length to compute STFT (default: {1024})
        fmax {int} -- Top edge of the highest frequency band (default: {8000})
        fmin {int} -- Lower bound on the frequencies to be included in the mel spectrum (default: {80})
    
    Returns:
        Tensor -- melspec as tf.Tensor
    """          
            
    # Step 1 : signals->stfts
    # `stfts` is a complex64 Tensor representing the Short-time Fourier Transform of
    # each signal in `signals`. Its shape is [batch_size, ?, fft_unique_bins]
    # where fft_unique_bins = fft_length // 2 + 1 = 513.
    if win_length == None:
        win_length = n_fft
        
    stfts = tf.signal.stft(signals, frame_length=win_length, frame_step=hop_length, fft_length=n_fft, window_fn=window, pad_end=pad_end)
       
    # Step2: stfts->magnitude_spectrograms
    # An energy spectrogram is the magnitude of the complex-valued STFT.
    # A float32 Tensor of shape [batch_size, ?, 513].
    spectrograms = tf.math.abs(stfts)**power
    
    # Step 3: magnitude_spectrograms->mel_spectrograms
    # Warp the linear-scale, magnitude spectrograms into the mel-scale.    
    num_spectrogram_bins = spectrograms.shape[-1]          

    # Step 4: Mel filter
    # < tf mel filter >
    #mel_basis = tf.signal.linear_to_mel_weight_matrix(n_mels, num_spectrogram_bins, sample_rate, fmin, fmax)
    #mel_spectrograms = tf.tensordot(spectrograms, mel_basis, 1)     
    # < librosa mel filter >
    mel_basis = tf.convert_to_tensor(librosa.filters.mel(sample_rate, n_fft, fmax=fmax), dtype=tf.float32)
    
    # Step 5: Matrix multiplication 
    mel_spectrograms = tf.tensordot(spectrograms, tf.transpose(mel_basis), 1)             
     
    return mel_spectrograms

def read_audio(path):
    '''
    Reads in the audio file and returns
    an array that we can turn into a melspectogram
    '''
    y, _ = librosa.core.load(path, sr=44100)
    # trim silence
    if 0 < len(y): # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y)
    if len(y) > samples: # long enough
        y = y[0:0+samples]
    else: # pad blank
        padding = samples - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, samples - len(y) - offset), 'constant')
    return y

def audio_to_melspectrogram_librosa(audio):
    '''
    Convert to melspectrogram after audio is read in
    '''
    spectrogram = librosa.feature.melspectrogram(audio, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft, fmin=fmin, fmax=fmax)       
    
    return librosa.power_to_db(spectrogram).astype(np.float32)
    
def audio_to_melspectrogram_tf(audio):
    '''
    Convert to melspectrogram after audio is read in
    '''
    spectrogram = get_melspec(signals=audio, sample_rate=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft, fmax=fmax, fmin=fmin)    
    
    return librosa.power_to_db(spectrogram).astype(np.float32)    

def read_as_melspectrogram(path, processor='gpu'):
    '''
    Convert audio into a melspectrogram 
    so we can use machine learning
    '''
    if processor == 'gpu':
        mels = audio_to_melspectrogram_tf(read_audio(path))
    else: # 'CPU' processing
        mels = audio_to_melspectrogram_librosa(read_audio(path))
          
    return mels

def convert_wav_to_image(path):
    '''
    Convert audio dataset into a melspectrogram dataset
    '''
    X = []
    audio_files = [f for f in listdir(path) if isfile(join(path, f))]
    for f in audio_files:
        name, ext = splitext(f)
        if ext == '.wav':
            x = read_as_melspectrogram('{}/{}'.format(path, f), 'gpu')
            X.append(x)
    return X

if __name__ == "__main__":
    start_time = time.time()
    X = np.array(convert_wav_to_image('input/'))
    end_time = time.time()
    print("Processing time : {}".format(end_time-start_time))