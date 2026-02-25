import torch
import torchaudio.transforms as T
import numpy as np
import polars as pl
from torch.utils.data import Dataset
from scipy.signal.windows import hann
from tqdm.auto import tqdm

def get_nasa_stats(file_list):
    dfs = []
    for f in tqdm(file_list, desc="Reading files"):
        df = pl.read_csv(
            f, 
            separator='\t', 
            has_header=False, 
            new_columns=['b1', 'b2', 'b3', 'b4']
        )
        dfs.append(df)
    
    full_df = pl.concat(dfs)

    # Обратите внимание на .name.
    stats = full_df.select([
        pl.all().mean().name.suffix("_mean"),
        pl.all().std().name.suffix("_std")
    ])
    
    return stats.to_dict(as_series=False)

class NASAIMSRawDataset(Dataset):
    def __init__(self, file_paths, window_size=2048, overlap=1024, bearing_idx=0, mu=0.0, sigma=1.0):
        self.window_size = window_size
        self.stride = window_size - overlap
        self.bearing_idx = bearing_idx
        self.mu = mu
        self.sigma = sigma
        self.file_paths = sorted(file_paths)

        self.points_per_file = 20480 
        # Вычитаем (window_size + stride), чтобы второе окно всегда помещалось в файл
        self.windows_per_file = (self.points_per_file - (self.window_size + self.stride)) // self.stride + 1
        
        print(f"Raw Pair Dataset: {len(self.file_paths)} files, {self.windows_per_file} pairs per file.")

    def __len__(self):
        return len(self.file_paths) * self.windows_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.windows_per_file
        window_in_file_idx = idx % self.windows_per_file
        
        start_1 = window_in_file_idx * self.stride
        start_2 = start_1 + self.stride
        
        file_path = self.file_paths[file_idx]
        # Читаем сразу весь файл для Bearer, чтобы не делать два запроса к CSV
        data = pl.read_csv(file_path, separator='\t', has_header=False, columns=[self.bearing_idx]).to_numpy().flatten()
        
        sig1 = (data[start_1 : start_1 + self.window_size] - self.mu) / self.sigma
        sig2 = (data[start_2 : start_2 + self.window_size] - self.mu) / self.sigma
        
        return (torch.tensor(sig1, dtype=torch.float32).unsqueeze(0), 
                torch.tensor(sig2, dtype=torch.float32).unsqueeze(0))

class NASAIMSSpectrumDataset(Dataset):
    def __init__(self, file_paths, window_size=2048, overlap=1024, bearing_idx=0, mu=0.0, sigma=1.0, use_windowing=True):
        self.window_size = window_size
        self.stride = window_size - overlap
        self.bearing_idx = bearing_idx
        self.mu = mu
        self.sigma = sigma
        self.file_paths = sorted(file_paths)
        
        self.points_per_file = 20480 
        self.windows_per_file = (self.points_per_file - (self.window_size + self.stride)) // self.stride + 1
        self.window_func = hann(window_size) if use_windowing else np.ones(window_size)

        print(f"Spectrum Pair Dataset: {len(self.file_paths)} files.")

    def __len__(self):
        return len(self.file_paths) * self.windows_per_file

    def _process_signal(self, sig):
        sig = (sig - self.mu) / self.sigma
        sig = sig - np.mean(sig)
        windowed = sig * self.window_func
        spec = np.abs(np.fft.rfft(windowed))
        return torch.tensor(np.log1p(spec), dtype=torch.float32).unsqueeze(0)

    def __getitem__(self, idx):
        file_idx = idx // self.windows_per_file
        window_in_file_idx = idx % self.windows_per_file
        
        start_1 = window_in_file_idx * self.stride
        start_2 = start_1 + self.stride
        
        data = pl.read_csv(self.file_paths[file_idx], separator='\t', has_header=False, columns=[self.bearing_idx]).to_numpy().flatten()
        
        spec1 = self._process_signal(data[start_1 : start_1 + self.window_size])
        spec2 = self._process_signal(data[start_2 : start_2 + self.window_size])
        
        return spec1, spec2

class NASAIMSMelDataset(Dataset):
    def __init__(self, file_paths, bearing_idx=0, sample_rate=20000, window_chunk_size=8192, overlap_size=4096,
                 n_fft=1024, hop_length=256, n_mels=64, f_max=5000, mu=0.0, sigma=1.0):
        
        self.file_paths = sorted(file_paths)
        self.bearing_idx = bearing_idx
        self.window_chunk_size = window_chunk_size
        self.stride = window_chunk_size - overlap_size
        self.mu = mu
        self.sigma = sigma
        
        self.points_per_file = 20480
        self.chunks_per_file = (self.points_per_file - (self.window_chunk_size + self.stride)) // self.stride + 1

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, 
            n_mels=n_mels, f_min=20, f_max=f_max, power=2.0
        )
        self.amplitude_to_db = T.AmplitudeToDB()

        self.last_file_idx = -1
        self.last_data = None

        print(f"Mel Pair Dataset: {len(self.file_paths)} files, {self.chunks_per_file} pairs per file.")

    def __len__(self):
        return len(self.file_paths) * self.chunks_per_file

    def _get_mel(self, signal_np):
        signal_np = signal_np - np.mean(signal_np)
        sig_t = torch.from_numpy(signal_np).float().unsqueeze(0)
        sig_t = (sig_t - self.mu) / self.sigma
        return self.amplitude_to_db(self.mel_transform(sig_t))

    def __getitem__(self, idx):
        file_idx = idx // self.chunks_per_file
        chunk_in_file_idx = idx % self.chunks_per_file
        
        start_1 = chunk_in_file_idx * self.stride
        start_2 = start_1 + self.stride
        
        if file_idx != self.last_file_idx:
            self.last_data = pl.read_csv(
                self.file_paths[file_idx], separator='\t', has_header=False, columns=[self.bearing_idx]
            ).to_numpy().flatten()
            self.last_file_idx = file_idx
        
        mel1 = self._get_mel(self.last_data[start_1 : start_1 + self.window_chunk_size])
        mel2 = self._get_mel(self.last_data[start_2 : start_2 + self.window_chunk_size])
        
        return mel1, mel2