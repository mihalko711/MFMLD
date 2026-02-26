from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import numpy as np
import polars as pl
import torch
import glob
import os
from tqdm.auto import tqdm
from scipy.signal.windows import hann

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
    def __init__(self, 
                 file_paths, 
                 window_size=2048, 
                 overlap=1024, 
                 bearing_idx=0, 
                 mu=None, 
                 sigma=None):
        """
        Args:
            file_paths (list): Список путей к файлам
            window_size (int): Размер окна
            overlap (int): Перекрытие
            bearing_idx (int): Индекс подшипника (0-3)
            mu, sigma (float): Если None, вычисляются автоматически по всему датасету
        """
        self.window_size = window_size
        self.stride = window_size - overlap
        self.points_per_file = 20480
        
        # 1. Загружаем все данные в память сразу
        print(f"Loading {len(file_paths)} files into memory...")
        all_signals = []
        
        for path in tqdm(sorted(file_paths)):
            df = pl.read_csv(path, separator='\t', has_header=False, columns=[bearing_idx])
            all_signals.append(df.to_numpy().flatten())
        
        # Конкатенируем в один гигантский вектор
        self.full_data = np.concatenate(all_signals).astype(np.float32)
        
        # 2. Нормализация
        if mu is None:
            self.mu = self.full_data.mean()
        else:
            self.mu = mu
            
        if sigma is None:
            self.sigma = self.full_data.std()
        else:
            self.sigma = sigma
            
        self.full_data = (self.full_data - self.mu) / self.sigma

        # 3. Расчет индексов
        self.windows_per_file = (self.points_per_file - self.window_size) // self.stride + 1
        self.num_files = len(file_paths)
        
        print(f"Dataset loaded. Total windows: {len(self)}")

    def __len__(self):
        return self.num_files * self.windows_per_file

    def __getitem__(self, idx):
        # Вычисляем глобальный индекс начала окна в self.full_data
        file_idx = idx // self.windows_per_file
        window_in_file_idx = idx % self.windows_per_file
        
        # Начало = (номер файла * точек в файле) + (сдвиг внутри файла)
        start_pos = (file_idx * self.points_per_file) + (window_in_file_idx * self.stride)
        end_pos = start_pos + self.window_size
        
        signal = self.full_data[start_pos:end_pos]
        
        # Возвращаем тензор (C, L), где C=1 (канал)
        return torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

class NASAIMSSpectrumDataset(Dataset):
    def __init__(self, 
                 file_paths, 
                 window_size=2048, 
                 overlap=1024, 
                 bearing_idx=0, 
                 mu=None, 
                 sigma=None,
                 use_windowing=True):
        
        self.window_size = window_size
        self.stride = window_size - overlap
        self.points_per_file = 20480
        self.use_windowing = use_windowing

        print(f"Loading {len(file_paths)} files into memory...")
        all_signals = []
        for path in tqdm(sorted(file_paths)):
            df = pl.read_csv(path, separator='\t', has_header=False, columns=[bearing_idx])
            all_signals.append(df.to_numpy().flatten())

        self.full_data = np.concatenate(all_signals).astype(np.float32)

        if mu is None:
            self.mu = self.full_data.mean()
        else:
            self.mu = mu
            
        if sigma is None:
            self.sigma = self.full_data.std()
        else:
            self.sigma = sigma

        if self.use_windowing:
            self.window_func = hann(window_size).astype(np.float32)
        else:
            self.window_func = np.ones(window_size, dtype=np.float32)

        self.num_files = len(file_paths)
        self.windows_per_file = (self.points_per_file - self.window_size) // self.stride + 1
        
        print(f"Spectrum Dataset ready. Total windows: {len(self)}")

    def __len__(self):
        return self.num_files * self.windows_per_file

    def __getitem__(self, idx):
        # Вычисляем позицию окна в глобальном массиве
        file_idx = idx // self.windows_per_file
        window_in_file_idx = idx % self.windows_per_file
        start_pos = (file_idx * self.points_per_file) + (window_in_file_idx * self.stride)
        end_pos = start_pos + self.window_size
        
        signal = self.full_data[start_pos:end_pos].copy()

        signal = (signal - self.mu) / self.sigma
        signal = signal - np.mean(signal)
        signal = signal * self.window_func
        
        spectrum = np.fft.rfft(signal)
        
        magnitude = np.abs(spectrum)
        magnitude = magnitude[:self.window_size//2] 
        log_magnitude = np.log1p(magnitude)

        return torch.from_numpy(log_magnitude.astype(np.float32)).unsqueeze(0)

class NASAIMSMelDataset(Dataset):
    def __init__(self, 
                 file_paths, 
                 bearing_idx=0, 
                 sample_rate=20000, 
                 window_chunk_size=8192,
                 overlap_size=4096,
                 n_fft=1024, 
                 hop_length=256,        
                 n_mels=64,
                 f_max=5000, 
                 mu=0.0, 
                 sigma=1.0):
        
        self.file_paths = sorted(file_paths)
        self.bearing_idx = bearing_idx
        self.sample_rate = sample_rate
        self.window_chunk_size = window_chunk_size
        self.mu = mu
        self.sigma = sigma
        
        # Шаг скольжения окна по файлу
        self.stride = window_chunk_size - overlap_size
        if self.stride <= 0:
            raise ValueError("Overlap должен быть меньше chunk_size")

        # Расчет количества чанков в одном файле (20480 точек)
        self.points_per_file = 20480
        self.chunks_per_file = (self.points_per_file - window_chunk_size) // self.stride + 1

        # Настройка Мел-трансформации
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=20,
            f_max=f_max,
            power=2.0
        )
        self.amplitude_to_db = T.AmplitudeToDB()

        # Кэш для ускорения чтения
        self.last_file_idx = -1
        self.last_data = None

        print(f"Dataset: {len(self.file_paths)} files.")
        print(f"Chunks per file: {self.chunks_per_file}, Total samples: {len(self)}")

    def __len__(self):
        return len(self.file_paths) * self.chunks_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.chunks_per_file
        chunk_in_file_idx = idx % self.chunks_per_file
        
        start_pos = chunk_in_file_idx * self.stride
        end_pos = start_pos + self.window_chunk_size
        
        # Оптимизированное чтение: не читаем файл с диска, если он уже в памяти
        if file_idx != self.last_file_idx:
            file_path = self.file_paths[file_idx]
            # Читаем сразу в numpy для скорости
            self.last_data = pl.read_csv(
                file_path, separator='\t', has_header=False, columns=[self.bearing_idx]
            ).to_numpy().flatten()
            self.last_file_idx = file_idx
        
        signal_np = self.last_data[start_pos:end_pos]
        
        # Препроцессинг
        signal_np = signal_np - np.mean(signal_np) # Удаляем DC
        signal_tensor = torch.from_numpy(signal_np).float().unsqueeze(0) # [1, chunk_size]
        signal_tensor = (signal_tensor - self.mu) / self.sigma

        
        # Спектрограмма
        with torch.no_grad():
            mel_spec = self.mel_transform(signal_tensor)
            mel_db = self.amplitude_to_db(mel_spec)
        
        mel_db = mel_db[:, :, :32] 
        
        return mel_db # [1, n_mels, time_steps]