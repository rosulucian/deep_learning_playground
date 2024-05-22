import librosa

import numpy as np

import matplotlib.pyplot as plt

def plot_energy(data, rate, figsize=(14,5)):
    energy = librosa.feature.rms(y=data)
    
    fig = plt.figure(figsize=figsize)
    
    plt.plot(librosa.samples_like(energy), energy[0] * 10)
    plt.plot(data)

    ax = fig.gca()
    ax.set_xticks(np.arange(0, data.shape[0], rate*5))
    
    plt.xlabel('samples')
    plt.ylabel('Energy')
    plt.title('Energy Plot')
    plt.grid()
    plt.show()

def plot_spectrogram(signal, sr, feature='linear', n_fft=2048, hop_length=512, win_length=None, window='hann', **kwargs):

    if feature == 'linear':
        S = librosa.amplitude_to_db(np.abs(librosa.stft(signal)), ref=np.max)
    elif feature == 'mel':
        S = librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            **kwargs,
        )
        
        S = librosa.power_to_db(S, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S, y_axis=feature)
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"{feature.capitalize()}-scaled spectrogram")
    plt.show()




