from model_utils import *
import torch
import torch.nn as nn
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self, input_dim, n_actions):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Conv2d(input_dim[0], 16, kernel_size=8, stride=4),
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2),
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1),
        
        output_size = self._get_conv_output(input_dim)

        self.fc1 = nn.Linear(output_size, 256),
        self.fc2 = nn.Linear(256, n_actions),
        self.activation = nn.ReLU()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _get_conv_output(self, shape):
        o = torch.zeros(1, *shape) 
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size())) 

    def forward(self, state):
        x = self.activation(self.conv1(state))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = x.view(x.size()[0], -1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x
    
def build_dqn(input_dim, n_actions, lr=0.0001):
    model = DeepQNetwork(input_dim, n_actions)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss = nn.MSELoss()
    return model, optimizer, loss