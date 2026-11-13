import librosa
import numpy as np

SAMPLE_RATE = 22050
DURATION = 3.0
N_MFCC = 40
MAX_FRAMES = 130

def load_audio(file_or_path):
    """Load mono audio and normalize its sample rate."""
    y, sr = librosa.load(file_or_path, sr=SAMPLE_RATE, mono=True)
    if y.size == 0:
        raise ValueError("The audio file is empty.")
    return y.astype(np.float32), SAMPLE_RATE

def extract_features(y, sr=SAMPLE_RATE):
    """Return a fixed-size MFCC matrix suitable for the CNN."""
    target_length = int(DURATION * sr)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
    if mfcc.shape[1] < MAX_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_FRAMES - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :MAX_FRAMES]
    return mfcc.astype(np.float32)

def emotion_name(label):
    return str(label).replace("_", " ").title()
