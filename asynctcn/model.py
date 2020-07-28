from torch import nn

from asynctcn.tcn import TemporalConvNet


class TCN(nn.Module):
    def __init__(self, input_size, outputs, num_channels, kernel_size=3, dropout=0.2):
        """ Irregular output TCN.

        Parameters
        ----------
        input_size : int
            Size of input
        outputs : list[tuple]
            List of tuples containing (key, output_size)
        num_channels : list
            List of # channels for each TCN block
        kernel_size : int, optional
            Kernel size (1d conv), by default 2
        dropout : float, optional
            Dropout, by default 0.2
        """
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)

        self.linear_layers = {}
        for key, output_size in outputs:
            self.linear_layers[key] = nn.Linear(num_channels[-1], output_size)

    def forward(self, x, key):
        y1 = self.tcn(x)
        out = self.linear_layers[key](y1[:, :, -1])
        return out
