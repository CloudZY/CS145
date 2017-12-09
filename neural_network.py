import torch
import torch.nn.functional as F

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):

        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        self.dropout = torch.nn.Dropout(p=0.1)

    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        h_relu = F.relu(self.linear1(x))
        h_relu = self.dropout(h_relu)
        o_origin = self.linear2(h_relu)
        return o_origin
