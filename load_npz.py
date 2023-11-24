import torch
import numpy as np

load_data = np.load('output.npz', allow_pickle=True)
imgs = load_data['data']
keys = load_data['keys']
data_dict = dict(zip(keys,imgs))

print("hello")