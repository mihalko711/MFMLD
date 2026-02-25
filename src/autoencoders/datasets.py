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
                 mu=0.0, 
                 sigma=1.0):
        """
        Args:
            data_dir (str): Путь к папке с текстовыми файлами (напр. '2nd_test')
            window_size (int): Размер окна (количество точек)
            overlap (int): Размер перекрытия в точках
            bearing_idx (int): Индекс подшипника (0, 1, 2 или 3)
            mu (float): Среднее для Z-нормализации
            sigma (float): Стандартное отклонение для Z-нормализации
        """
        self.window_size = window_size
        self.overlap = overlap
        self.bearing_idx = bearing_idx
        self.mu = mu
        self.sigma = sigma
        
        self.file_paths = sorted(file_paths)
        if not self.file_paths:
            raise FileNotFoundError(f"Файлы в {data_dir} не найдены.")

        self.stride = window_size - overlap
        if self.stride <= 0:
            raise ValueError("Overlap должен быть меньше, чем window_size")

        self.points_per_file = 20480 
        self.windows_per_file = (self.points_per_file - self.window_size) // self.stride + 1
        
        print(f"Dataset initialized: {len(self.file_paths)} files, "
              f"{self.windows_per_file} windows per file. "
              f"Total samples: {len(self)}")

    def __len__(self):
        return len(self.file_paths) * self.windows_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.windows_per_file
        
        window_in_file_idx = idx % self.windows_per_file
        start_pos = window_in_file_idx * self.stride
        end_pos = start_pos + self.window_size
        
        file_path = self.file_paths[file_idx]
        
        data = pl.read_csv(file_path, separator='\t', has_header=False, columns=[self.bearing_idx])
        signal = data[start_pos:end_pos, 0].to_numpy()
        
        signal = (signal - self.mu) / self.sigma
        
        return torch.tensor(signal, dtype=torch.float32).unsqueeze(0)

class NASAIMSSpectrumDataset(Dataset):
    def __init__(self, 
                 file_paths, 
                 window_size=2048, 
                 overlap=1024, 
                 bearing_idx=0, 
                 mu=0.0, 
                 sigma=1.0,
                 use_windowing=True):
        """
        Args:
            file_paths (list): Список путей к файлам.
            window_size (int): Размер окна для БПФ.
            overlap (int): Перекрытие окон.
            bearing_idx (int): Индекс колонки подшипника (0-3).
            mu (float): Среднее для нормализации исходного сигнала.
            sigma (float): СКО для нормализации исходного сигнала.
            use_windowing (bool): Применять ли окно Ханна перед БПФ.
        """
        self.window_size = window_size
        self.overlap = overlap
        self.bearing_idx = bearing_idx
        self.mu = mu
        self.sigma = sigma
        self.use_windowing = use_windowing
        
        self.file_paths = sorted(file_paths)
        if not self.file_paths:
            raise FileNotFoundError("Файлы не найдены.")

        self.stride = window_size - overlap
        self.points_per_file = 20480 
        self.windows_per_file = (self.points_per_file - self.window_size) // self.stride + 1

        if self.use_windowing:
            self.window_func = hann(window_size)
        else:
            self.window_func = np.ones(window_size)

        print(f"Spectrum Dataset initialized: {len(self.file_paths)} files.")
        print(f"Output spectrum size: {window_size // 2 + 1} bins.")

    def __len__(self):
        return len(self.file_paths) * self.windows_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.windows_per_file
        window_in_file_idx = idx % self.windows_per_file
        start_pos = window_in_file_idx * self.stride
        end_pos = start_pos + self.window_size
        
        file_path = self.file_paths[file_idx]
        
        data = pl.read_csv(file_path, separator='\t', has_header=False, columns=[self.bearing_idx])
        signal = data[start_pos:end_pos, 0].to_numpy()

        signal = (signal - self.mu) / self.sigma
        signal = signal - np.mean(signal)
        windowed_signal = signal * self.window_func
        
        spectrum = np.fft.rfft(windowed_signal)
        
        magnitude_spectrum = np.abs(spectrum)
        magnitude_spectrum = np.log1p(magnitude_spectrum)

        return torch.tensor(magnitude_spectrum, dtype=torch.float32).unsqueeze(0)

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
        mel_spec = self.mel_transform(signal_tensor)
        mel_db = self.amplitude_to_db(mel_spec)
        
        return mel_db # [1, n_mels, time_steps]