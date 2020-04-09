import numpy as np
import torch
from torch import nn

# costumary linear layer which is applied to transposed activations of incoming layer

class Lin_Tr(torch.nn.Module):
    def __init__(self, D_in, H):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(Lin_Tr, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H, bias=None)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """

        out = self.linear1(x.transpose_(1, 2))

        return out

class Indicator(nn.Module):
    
    def __init__(
        self,
        in_channels=1,
        out_channels=2,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
        a=-0.0312,
        b=-3/2,
        init_mode='optimal',
        pert_1_w=torch.Tensor([0.]),
        pert_1_b=torch.Tensor([0.]),
        pert_2_w=torch.Tensor([0.]),
        pert_2_b=torch.Tensor([0.]),
        pert_3_w=torch.Tensor([0.]),
        pert_3_b=torch.Tensor([0.]),
    ):
        super().__init__()
        self.layers = nn.Sequential(
                      nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias), # convolution necessary to represent boundary conditions for vector of points of periodic shift
                      nn.ReLU(),
                      Lin_Tr(2, 2), #want -Id, no bias for simplification
                      nn.Threshold(a, 1), # results in 2-dim binary representation of lower and upper bound for interval
                      nn.Linear(2, 1, bias=None), #want (-1,-1), no bias, combines together the binary information
                      nn.Threshold(b, 1), # 1 if and only if both binary inputs are 1
                      nn.ReLU(), # zero out all other values
                     )
        self.init_weights(init_mode, pert_1_w, pert_1_b, pert_2_w, pert_2_b, pert_3_w, pert_3_b)

    def init_weights(self, mode, pert_1_w, pert_1_b, pert_2_w, pert_2_b, pert_3_w, pert_3_b):

        '''
        Different initialization can be chosen, including the optimal initialization. Allows perturbations of the parameters of the optimal initialization.

        Parameters:
            mode : string - choose initialization
            pert_#layer_w : torch-Tensor - define perturbation of weight
            pert_#layer_b : torch.Tensor - define perturbation of bias
        '''


        if mode=='optimal':

            # manual initialization to the optimal weights

            self.layers[0].weight = torch.nn.Parameter(torch.Tensor([[[1]], [[-1]]]) + pert_1_w)
            self.layers[0].bias = torch.nn.Parameter(torch.Tensor([0., 0.5]) + pert_1_b)

            self.layers[2].linear1.weight = torch.nn.Parameter(-torch.eye(2) + pert_2_w)
            #self.layers[2].linear1.bias = torch.nn.Parameter(torch.Tensor([0., 0.]) + pert_2_b) # bias not used for simplification

            self.layers[4].weight = torch.nn.Parameter(torch.Tensor([[-1., -1.]]) + pert_3_w)
            #self.layers[4].bias = torch.nn.Parameter(torch.Tensor([1.]) + pert_3_b)




    def forward(self, x):

        return self.layers(x)



def toy_data(N=10000, n=3):
    
    '''
    Creates the toy dataset consisting of periodic shifts of a discrete representation of the interval [0,1).

    Parameters:
        N : integer - number of training samples
        n : integer - controls number of points representing the periodic shift via the defintion of m
    '''

    m = 2 ** (n + 1)
    base = np.arange(2 ** (n + 1)) / 2 ** (n + 1)
    base_target = (base >= 0) * (base < 1 / 2)
    base_feat = base + 1 / 2 ** (n + 2)
    shifts = np.random.randint(0, m - 1, N)

    data = []

    for i in range(N):

        data.append([np.roll(base_feat, shift=shifts[i], axis=0).reshape(1, -1), np.roll(base_target, shift=shifts[i], axis=0).reshape(1, -1)])

    return data


def train(data, model, criterion=nn.MSELoss(), lr=0.05, epochs=10, batch_size=64):

    trainloader = torch.utils.data.DataLoader(data, shuffle=True, batch_size=batch_size)

    optimizer = torch.optim.SGD(model.parameters(), lr)

    # training

    for ep in range(epochs):
        running_loss = 0
        for feature, target in trainloader:

            # Training pass
            optimizer.zero_grad()

            output = model(feature.float())
            loss = criterion(output.transpose_(1, 2), target.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            print(f"Training loss: {running_loss/len(trainloader)}")