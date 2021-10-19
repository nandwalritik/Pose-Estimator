# import torch
# import torch.nn as nn

# class TimeDistributed(nn.Module):
#     def __init__(self, layer, time_steps, **args):
#         super(TimeDistributed, self).__init__()

#         self.layers = nn.ModuleList([layer(**args) for i in range(time_steps)])

#     def forward(self, x):

#         batch_size, time_steps, C, H, W = x.size()
#         output = torch.tensor([])
#         for i in range(time_steps):
#           output_t = self.layers[i](x[:, i, :, :, :])
#           output_t  = output_t.unsqueeze(1)
#           output = torch.cat((output, output_t ), 1)
#         return output

# x = torch.rand(20,45,1,12,2)
# model = TimeDistributed(nn.Conv1d,time_steps=45,in_channels = 1,out_channels=8,kernel_size=(2,2))
# output = model(x)
# print(output.shape)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers import Conv1D, MaxPooling1D, LSTM
from keras.layers.wrappers import TimeDistributed
import numpy as np
# from keras.optimizers import Adam 

model = Sequential([
        TimeDistributed(Conv1D(16,3, activation='relu', padding = "same"),input_shape=(45,12,2)),
        TimeDistributed(BatchNormalization()),
        #TimeDistributed(MaxPooling1D()),
        TimeDistributed(Dropout(0.5)),
        #TimeDistributed(Conv1D(64,3, activation='relu',padding = "same")),
        BatchNormalization(),
        #TimeDistributed(Dropout(0.8)),
        TimeDistributed(Flatten()),
        #TimeDistributed(Dense(30,activation='softmax')),  
        LSTM(20,unit_forget_bias = 0.5, return_sequences = True),
        TimeDistributed(Dense(6,activation='softmax'))        
    ])


x = np.random.rand(20,45,12,2)
# model = model(x)
output = model(x)
print(model.weights.size())
print(output.shape)