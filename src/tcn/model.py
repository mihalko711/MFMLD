import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # Используем .clone() или .contiguous() для безопасности после среза
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, groups=8):
        super(TemporalBlock, self).__init__()
        
        # Первая свертка
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.gn1 = nn.GroupNorm(groups, n_outputs) # Групповая нормализация
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Вторая свертка
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.gn2 = nn.GroupNorm(groups, n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.gn1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.gn2, self.relu2, self.dropout2)
        
        # Линейная подстройка размерности для Skip-connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, groups=8):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Важно: num_channels[i] должен делиться на groups нацело!
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 
                                     stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, 
                                     dropout=dropout, groups=groups)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNForecaster(nn.Module):
    def __init__(self, input_dim, output_horizon, channels, kernel_size=3, dropout=0.2):
        super().__init__()
        # Сохраняем параметры для чекпоинта
        self.input_dim = input_dim
        self.output_horizon = output_horizon
        self.channels = channels
        self.kernel_size = kernel_size
        self.dropout = dropout

        self.tcn = TemporalConvNet(input_dim, channels, kernel_size, dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], output_horizon)

    def forward(self, x):
        # x: [Batch, 1, 1024]
        out = self.tcn(x)         # [Batch, Last_Channel, 1024]
        out = self.pool(out)      # [Batch, Last_Channel, 1]
        out = out.squeeze(-1)     # [Batch, Last_Channel]
        return self.fc(out)       # [Batch, 256]