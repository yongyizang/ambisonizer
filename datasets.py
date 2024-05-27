import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import librosa

class Ambisonizer(Dataset):
    """
    Dataset class for the Ambisonizer dataset.
    """
    def __init__(self, base_dir, partition="train", max_len=120000):
        assert partition in ["train", "val", "test"], "Invalid partition. Must be one of ['train', 'val', 'test']"
        self.base_dir = base_dir
        self.partition = partition
        self.base_dir = os.path.join(base_dir, partition)
        self.max_len = max_len
        self.file_list = os.listdir(self.base_dir)
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):            
        base_path = os.path.join(self.base_dir, self.file_list[index])
        w, _ = librosa.load(os.path.join(base_path, "W.wav"), sr=44100, mono=True)
        x, _ = librosa.load(os.path.join(base_path, "X.wav"), sr=44100, mono=True)
        y, _ = librosa.load(os.path.join(base_path, "Y.wav"), sr=44100, mono=True)
        
        # random crop indx
        crop_idx = np.random.randint(0, len(w)-self.max_len)
        w = w[crop_idx:crop_idx+self.max_len]
        x = x[crop_idx:crop_idx+self.max_len]
        y = y[crop_idx:crop_idx+self.max_len]
        
        azimuth_list = [22.5, 202.5]
        azimuth_list = [np.deg2rad(azimuth) for azimuth in azimuth_list]
        left = w + x * np.cos(azimuth_list[0]) + y * np.sin(azimuth_list[0])
        right = w + x * np.cos(azimuth_list[1]) + y * np.sin(azimuth_list[1])
        sig = np.array([left, right])
        target = np.array([x, y])
        random_gain = np.random.uniform(0.5, 1.0)
        sig *= random_gain
        target *= random_gain
        return sig, target
    
def test(source_dir, partition)
    dataset = Ambisonizer(base_dir=source_dir, partition=partition)
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][0].dtype)
    print(dataset[0][1].dtype)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    for i, (x, y) in enumerate(dataloader):
        print(x.shape)
        print(y.shape)
        break