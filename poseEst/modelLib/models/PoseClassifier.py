import torch
import torch.nn as nn
from .. import config

"""
    The imp keypoints afaik
    11,12,13,14,15,16,17,18,23,24,25,26,27,28
"""


class TimeDistributed(nn.Module):
    def __init__(self, layer, time_steps, **args):
        super(TimeDistributed, self).__init__()
        self.layers = nn.ModuleList([layer(**args) for i in range(time_steps)])

    def forward(self, x):
        batch_size, time_steps, _, num_keypoints, dim = x.size()
        # print(x.size())
        output = torch.tensor([]).to(config.DEVICE)
        for i in range(time_steps):
            output_t = self.layers[i](x[:, i, :, :, :])
            output_t = output_t.unsqueeze(1)
            output = torch.cat((output, output_t), 1)
        return output


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=(2, 2))
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv1(x))


class LSTMModel(nn.Module):
    def __init__(self, num_classes, lstm_layers, hidden_size, fc_size):
        super(LSTMModel, self).__init__()
        self.num_classes = num_classes
        self.lstm_layers = lstm_layers
        self.hidden_size = hidden_size
        self.fc_size = fc_size

        self.lstm = nn.LSTM(input_size=self.fc_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.lstm_layers)

        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        outputs, hidden = self.lstm(x)
        outputs = self.fc(outputs)
        return outputs


class PoseClassifier(nn.Module):

    def __init__(self):
        super(PoseClassifier, self).__init__()
        self.model = nn.Sequential(
            TimeDistributed(CNNBlock, time_steps=45,
                            in_channels=1, out_channels=16),
            TimeDistributed(nn.BatchNorm2d, time_steps=45, num_features=16),
            TimeDistributed(nn.Dropout, time_steps=45, p=0.5),
            # nn.BatchNorm2d(16),
            TimeDistributed(nn.Flatten, time_steps=45),
            LSTMModel(6, 20, 120, 224)
        )

    def forward(self, x):
        return self.model(x)


# if __name__ == "__main__":
#     model = PoseClassifier().to(config.DEVICE)
#     temp = torch.rand(20, 45, 1, 15, 2).to(config.DEVICE)
#     output = model(temp)
#     # print(output)
#     print(output.shape)
