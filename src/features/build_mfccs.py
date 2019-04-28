import numpy as np
import librosa
import pandas as pd
import sklearn


DATA_PATH = '../data/procesed/train_curated_features.pkl'
SAVE_PATH = '../data/procesed/mfccs/train_curated_mfccs.npy'

# Set the hop length; at 22050 Hz, 512 samples ~= 23ms
# also N of FFT is defined with this value

def wav2mfcc(file_path, hop_length = 512, max_pad_len=None, normalize=True):

	wave, sr = librosa.load(file_path, mono=True)

	if normalize:
		wave = librosa.util.normalize(wave) # normalizing data before mfcc

	# making mfcc from signal
	mfcc = librosa.feature.mfcc(y=wave, sr=sr, hop_length=hop_length, dct_type=2, n_mfcc=40)

	# scaling
	mfcc = sklearn.preprocessing.scale(mfcc, axis=1)

	if max_pad_len:
		if mfcc.shape[1] > max_pad_len:
    		mfcc = mfcc[:,:max_pad_len]
		else:
	    	pad_width = max_pad_len - mfcc.shape[1]
	    	mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    

    return mfcc


def gen_mfcc(input_path=DATA_PATH, output_path=SAVE_PATH, hop_length = 512, max_pad_len=200):
    wavfiles = pd.read_pickle(input_path)['path']


    # Init mfcc vectors
        mfcc_vectors = []

    for wavfile in wavfiles:
    	mfcc = wav2mfcc(wavfile, hop_length=hop_length, max_pad_len=max_pad_len)
    	mfcc_vectors.append(mfcc)


    np.save(output_path, mfcc_vectors)
