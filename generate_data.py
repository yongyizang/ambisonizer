import os, sys
import numpy as np
import librosa
import soundfile as sf
import audiomentations as augs
from scipy import signal
from tqdm import tqdm

np.random.seed(42)
audio_source_dir = None # Fill this with the path to the audio source directory
ir_source_dir = None # Fill this with the path to the IR source directory
data_output_dir = None # Fill this with the path to the output directory
n_samples = None # Fill this with the number of samples you want to generate

assert audio_source_dir is not None, "Please fill in the audio source directory"
assert ir_source_dir is not None, "Please fill in the IR source directory"
assert data_output_dir is not None, "Please fill in the output directory"
assert n_samples is not None, "Please fill in the number of samples to generate"

sample_length = 10 # seconds
source_max_width = np.pi # in radians

irs = os.listdir(ir_source_dir)
audio_files = os.listdir(audio_source_dir)
print("Found", len(irs), "IRs and", len(audio_files), "audio files")

sample_configs = []
for i in tqdm(range(n_samples), desc="Generating config for each sample"):
    ir = np.random.choice(irs)
    audio = np.random.choice(audio_files)
    sample_configs.append((ir, audio))
print("Generated", len(sample_configs), "sample configs")

def convolve(ir, audio):
    return signal.fftconvolve(audio, ir, mode="full")[:len(audio)]

augment = augs.Compose([
    augs.Gain(min_gain_db=-10, max_gain_db=10, p=0.7),
    augs.AirAbsorption(min_distance=0.1, max_distance=10, p=0.7),
    augs.SevenBandParametricEQ(p=0.7),
    augs.GainTransition(p=0.7),
])

def get_coefficients(azimuth, elevation=0):
    # assuming that azimuth and elevation are in radians.
    w = 1
    x = np.cos(elevation) * np.cos(azimuth) * np.sqrt(3)
    y = np.cos(elevation) * np.sin(azimuth) * np.sqrt(3)
    z = np.sin(elevation) * np.sqrt(3)
    return w, x, y, z

def process_sample_config(sample_config):
    ir, audio = sample_config
    audio_name = audio.split(".")[0]
    ir_path = os.path.join(ir_source_dir, ir)
    w_ir, _ = librosa.load(os.path.join(ir_path, "W.wav"), sr=44100, mono=True)
    x_ir, _ = librosa.load(os.path.join(ir_path, "X.wav"), sr=44100, mono=True)
    y_ir, _ = librosa.load(os.path.join(ir_path, "Y.wav"), sr=44100, mono=True)
    # z_ir, _ = librosa.load(os.path.join(ir_path, "Z.wav"), sr=44100, mono=True)
    
    # randomly chop ir
    ir_length = np.random.uniform(0.3, 1.0)
    ir_length = int(ir_length*44100)
    w_ir = w_ir[:ir_length]
    x_ir = x_ir[:ir_length]
    y_ir = y_ir[:ir_length]
    
    ir_fadeout_length = np.random.uniform(0.05, 0.3)
    ir_fadeout_length = int(ir_fadeout_length*44100)
    ir_fadeout = np.linspace(1, 0, ir_fadeout_length)
    w_ir[-ir_fadeout_length:] *= ir_fadeout
    x_ir[-ir_fadeout_length:] *= ir_fadeout
    y_ir[-ir_fadeout_length:] *= ir_fadeout
    
    audio_path = os.path.join(audio_source_dir, audio)
    
    w_channel = np.zeros((sample_length*44100,))
    x_channel = np.zeros((sample_length*44100,))
    y_channel = np.zeros((sample_length*44100,))
    # z_channel = np.zeros((sample_length*44100,))
    
    audios = os.listdir(audio_path)
    audio_length = librosa.get_duration(path=os.path.join(audio_path, audios[0]))
    start_index = int(np.random.uniform(0, int((audio_length - sample_length) * 44100)))
    
    azimuths = np.random.uniform(-np.pi, np.pi, len(audios))
    
    for i, audio in enumerate(audios):
        curr_azi = azimuths[i]
        y, _ = librosa.load(os.path.join(audio_path, audio), sr=44100, mono=False)
        if len(y.shape) == 1:
            y = y[start_index:start_index+int(sample_length*44100)]
            y = augment(samples=y, sample_rate=44100)
            w_c, x_c, y_c, z_c = get_coefficients(curr_azi)
            w_channel += convolve(w_ir, y) * w_c
            x_channel += convolve(x_ir, y) * x_c
            y_channel += convolve(y_ir, y) * y_c
            # z_channel += convolve(z_ir, y) * z_c
        elif len(y.shape) == 2:
            y = y[:, start_index:start_index+int(sample_length*44100)]
            y = augment(samples=y, sample_rate=44100)
            left_sig = y[0] * 0.5
            right_sig = y[1] * 0.5
            source_width = np.random.uniform(0, source_max_width)
            left_azimuth = curr_azi - source_width/2
            right_azimuth = curr_azi + source_width/2
            left_w_c, left_x_c, left_y_c, left_z_c = get_coefficients(left_azimuth)
            right_w_c, right_x_c, right_y_c, right_z_c = get_coefficients(right_azimuth)
            w_channel += convolve(w_ir, left_sig) * left_w_c + convolve(w_ir, right_sig) * right_w_c
            x_channel += convolve(x_ir, left_sig) * left_x_c + convolve(x_ir, right_sig) * right_x_c
            y_channel += convolve(y_ir, left_sig) * left_y_c + convolve(y_ir, right_sig) * right_y_c
            # z_channel += convolve(z_ir, left_sig) * left_z_c + convolve(z_ir, right_sig) * right_z_c
        else:
            raise ValueError("Audio file is not mono or stereo")
    data = np.array([w_channel, x_channel, y_channel])
    if np.max(np.abs(data)) == 0:
        pass
    elif np.max(np.abs(data)) > 1:
        data = data / np.max(np.abs(data))
    
    data = data[:, :int(sample_length*44100)]

    curr_output_dir = os.path.join(data_output_dir, ir.split(".")[0] + "|" + audio_name.split(".")[0].replace(" ", "_") + "|" + str(start_index/44100).format(".2f"))
    if not os.path.exists(curr_output_dir):
        os.makedirs(curr_output_dir)
    sf.write(os.path.join(curr_output_dir, "W.wav"), data[0], 44100)     
    sf.write(os.path.join(curr_output_dir, "X.wav"), data[1], 44100)
    sf.write(os.path.join(curr_output_dir, "Y.wav"), data[2], 44100)
    # sf.write(os.path.join(curr_output_dir, "Z.wav"), data[3], 44100)
           
# use multiprocessing and tqdm
import multiprocessing as mp
pool = mp.Pool(mp.cpu_count())
with tqdm(total=len(sample_configs), desc="Processing samples") as pbar:
    for _ in pool.imap_unordered(process_sample_config, sample_configs):
        pbar.update()