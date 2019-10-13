import torch
import torch.nn as nn

class Net(nn.Module):
	def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(,16)
        self.fc2 = nn.Linear(16,16)
        self.fc3 = nn.Linear(16,16)
        self.fc4 = nn.Linear(16,4)
