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

        all_signals = []
        for path in tqdm(self.file_paths, desc="Loading data to RAM"):
            data = pl.read_csv(path, separator='\t', has_header=False, columns=[bearing_idx])
            all_signals.append(data.to_numpy().flatten())
        
        self.full_data = np.concatenate(all_signals).astype(np.float32)
        self.windows_per_file = (self.points_per_file - (self.window_size + self.stride)) // self.stride + 1

    def __len__(self):
        return len(self.file_paths) * self.windows_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.windows_per_file
        window_in_file_idx = idx % self.windows_per_file
        
        base_offset = file_idx * self.points_per_file
        start_1 = base_offset + (window_in_file_idx * self.stride)
        start_2 = start_1 + self.stride
        
        sig1 = (self.full_data[start_1 : start_1 + self.window_size] - self.mu) / self.sigma
        sig2 = (self.full_data[start_2 : start_2 + self.window_size] - self.mu) / self.sigma
        
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

        all_signals = []
        for path in tqdm(self.file_paths, desc="Loading spectrum data"):
            data = pl.read_csv(path, separator='\t', has_header=False, columns=[bearing_idx])
            all_signals.append(data.to_numpy().flatten())
        
        self.full_data = np.concatenate(all_signals).astype(np.float32)
        self.windows_per_file = (self.points_per_file - (self.window_size + self.stride)) // self.stride + 1
        self.window_func = hann(window_size).astype(np.float32) if use_windowing else np.ones(window_size, dtype=np.float32)

    def __len__(self):
        return len(self.file_paths) * self.windows_per_file

    def _process_signal(self, sig):
        sig = (sig - self.mu) / self.sigma
        sig = sig - np.mean(sig)
        windowed = sig * self.window_func
        spec = np.abs(np.fft.rfft(windowed))
        # Срез до 1024 для корректной работы UNet (чтобы размер был кратен степени 2)
        spec = spec[:self.window_size // 2] 
        return torch.from_numpy(np.log1p(spec).astype(np.float32)).unsqueeze(0)

    def __getitem__(self, idx):
        file_idx = idx // self.windows_per_file
        window_in_file_idx = idx % self.windows_per_file
        
        base_offset = file_idx * self.points_per_file
        start_1 = base_offset + (window_in_file_idx * self.stride)
        start_2 = start_1 + self.stride
        
        spec1 = self._process_signal(self.full_data[start_1 : start_1 + self.window_size])
        spec2 = self._process_signal(self.full_data[start_2 : start_2 + self.window_size])
        
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

        all_signals = []
        for path in tqdm(self.file_paths, desc="Loading mel data"):
            data = pl.read_csv(path, separator='\t', has_header=False, columns=[bearing_idx])
            all_signals.append(data.to_numpy().flatten())
        
        self.full_data = np.concatenate(all_signals).astype(np.float32)
        self.chunks_per_file = (self.points_per_file - (self.window_chunk_size + self.stride)) // self.stride + 1

        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, 
            n_mels=n_mels, f_min=20, f_max=f_max, power=2.0
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def __len__(self):
        return len(self.file_paths) * self.chunks_per_file

    def _get_mel(self, signal_np):
        signal_np = signal_np - np.mean(signal_np)
        sig_t = torch.from_numpy(signal_np).unsqueeze(0)
        sig_t = (sig_t - self.mu) / self.sigma
        
        with torch.no_grad():
            mel = self.amplitude_to_db(self.mel_transform(sig_t))
        
        # Срез до 32 временных отсчетов для корректной работы UNet
        return mel[:, :, :32].to(torch.float32)

    def __getitem__(self, idx):
        file_idx = idx // self.chunks_per_file
        chunk_in_file_idx = idx % self.chunks_per_file
        
        base_offset = file_idx * self.points_per_file
        start_1 = base_offset + (chunk_in_file_idx * self.stride)
        start_2 = start_1 + self.stride
        
        mel1 = self._get_mel(self.full_data[start_1 : start_1 + self.window_chunk_size])
        mel2 = self._get_mel(self.full_data[start_2 : start_2 + self.window_chunk_size])
        
        return mel1, mel2

class NASAIMSRawForecastDataset(Dataset):
    def __init__(self, file_paths, pre_horizon=2048, horizon=1024, overlap=1024, bearing_idx=0, mu=0.0, sigma=1.0):
        self.pre_horizon = pre_horizon
        self.horizon = horizon
        self.total_len = pre_horizon + horizon
        self.stride = self.total_len - overlap
        self.bearing_idx = bearing_idx
        
        # Принудительно делаем константы float32
        self.mu = np.float32(mu)
        self.sigma = np.float32(sigma)
        
        self.file_paths = sorted(file_paths)
        self.points_per_file = 20480

        all_signals = []
        for path in tqdm(self.file_paths, desc="Loading data to RAM"):
            data = pl.read_csv(path, separator='\t', has_header=False, columns=[bearing_idx])
            all_signals.append(data.to_numpy().flatten())
        
        self.full_data = np.concatenate(all_signals).astype(np.float32)
        self.windows_per_file = (self.points_per_file - self.total_len) // self.stride + 1

    def __len__(self):
        return len(self.file_paths) * self.windows_per_file

    def __getitem__(self, idx):
        file_idx = idx // self.windows_per_file
        window_in_file_idx = idx % self.windows_per_file
        
        base_offset = file_idx * self.points_per_file
        start_pos = base_offset + (window_in_file_idx * self.stride)
        mid_pos = start_pos + self.pre_horizon
        end_pos = start_pos + self.total_len
        
        # Операции происходят в float32, так как и массив, и mu/sigma имеют этот тип
        sig_pre = (self.full_data[start_pos : mid_pos] - self.mu) / self.sigma
        sig_hor = (self.full_data[mid_pos : end_pos] - self.mu) / self.sigma
        
        # .float() в конце гарантирует dtype=torch.float32
        return (torch.from_numpy(sig_pre).float().unsqueeze(0), 
                torch.from_numpy(sig_hor).float().unsqueeze(0))

class NASAIMSLongTermSpectrumDataset(Dataset):
    def __init__(self, 
                 file_paths, 
                 window_size=2048, 
                 stride=1024, 
                 horizon_files=5, 
                 bearing_idx=0, 
                 mu=None, 
                 sigma=None,
                 use_windowing=True):
        """
        Args:
            file_paths (list): Список путей к файлам (отсортированный).
            window_size (int): Размер окна для БПФ.
            stride (int): Шаг окна внутри одного файла.
            horizon_files (int): На сколько файлов вперед смотрим (прогноз).
                                Если файлы через 10 мин, то horizon_files=6 — это прогноз на 1 час.
            bearing_idx (int): Индекс подшипника (0-3).
            mu, sigma (float): Параметры нормализации.
        """
        self.window_size = window_size
        self.stride = stride
        self.horizon_files = horizon_files
        self.points_per_file = 20480
        
        # 1. Загрузка всех данных в RAM (как мы и решили, это быстрее всего)
        self.file_paths = sorted(file_paths)
        print(f"Loading {len(self.file_paths)} files into RAM...")
        all_signals = []
        for path in tqdm(self.file_paths):
            data = pl.read_csv(path, separator='\t', has_header=False, columns=[bearing_idx])
            all_signals.append(data.to_numpy().flatten())
        
        self.full_data = np.concatenate(all_signals).astype(np.float32)
        
        # 2. Статистика для нормализации
        self.mu = mu if mu is not None else self.full_data.mean()
        self.sigma = sigma if sigma is not None else self.full_data.std()
        
        # 3. Настройка окна
        self.window_func = hann(window_size).astype(np.float32) if use_windowing else np.ones(window_size, dtype=np.float32)
        
        # 4. Расчет индексов
        self.windows_per_file = (self.points_per_file - self.window_size) // self.stride + 1
        # Количество доступных "стартовых" файлов (последние K файлов не могут быть входными)
        self.available_files = len(self.file_paths) - self.horizon_files
        
        if self.available_files <= 0:
            raise ValueError("horizon_files слишком велик для такого количества файлов!")

        print(f"Dataset initialized: {len(self)} pairs available.")
        print(f"Prediction Horizon: {self.horizon_files} files ahead.")

    def __len__(self):
        # Общее количество пар: (кол-во файлов - горизонт) * окон в файле
        return self.available_files * self.windows_per_file

    def _process_signal(self, sig):
        """Превращает сырой отрезок сигнала в Log-Spectrum"""
        # Нормализация
        sig = (sig - self.mu) / self.sigma
        # Удаление DC-составляющей
        sig = sig - np.mean(sig)
        # Оконная функция
        sig = sig * self.window_func
        # БПФ
        spec = np.abs(np.fft.rfft(sig))
        # Срез до степени 2 (1024 точки) для UNet/MLP
        spec = spec[:self.window_size // 2]
        # Логарифмирование
        return np.log1p(spec).astype(np.float32)

    def __getitem__(self, idx):
        # 1. Определяем индексы файлов
        file_idx_current = idx // self.windows_per_file
        file_idx_future = file_idx_current + self.horizon_files
        
        # 2. Определяем позицию окна внутри файлов (используем ту же позицию для обоих)
        window_in_file_idx = idx % self.windows_per_file
        start_in_file = window_in_file_idx * self.stride
        
        # 3. Вычисляем глобальные смещения в self.full_data
        offset_current = (file_idx_current * self.points_per_file) + start_in_file
        offset_future = (file_idx_future * self.points_per_file) + start_in_file
        
        # 4. Извлекаем сырые сигналы
        sig_current = self.full_data[offset_current : offset_current + self.window_size]
        sig_future = self.full_data[offset_future : offset_future + self.window_size]
        
        # 5. Превращаем в спектры
        spec_current = self._process_signal(sig_current)
        spec_future = self._process_signal(sig_future)
        
        # 6. Возвращаем тензоры [1, 1024]
        return (torch.from_numpy(spec_current).unsqueeze(0), 
                torch.from_numpy(spec_future).unsqueeze(0))