import bisect
import math
import os

import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class DassaultDataset(Dataset):
    def __init__(self, train_dir, seq_window=400):
        """
        seq_window should (could) eventually be a list with the window for each input channel
        """
        super().__init__()

        self.seq_window = seq_window
        self.sequences = []

        for fp in os.listdir(train_dir):
            if 'flight' not in fp:
                continue
            
            flight_path = os.path.join(train_dir,
                                       fp)
            # read Summary csv
            df_summary = pd.read_csv(os.path.join(flight_path,
                                                  'Summary.csv'),
                                                  sep=';')

            # remove disabled sensors
            df_summary = df_summary[df_summary['ENABLED'] == 1]

            X = []
            y = [] # tuple (label, arr)

            # Grab sensor time series
            for lab, typ in df_summary[['LABEL', 'TYPE']].astype(str).values:
                npa = np.load(os.path.join(flight_path,
                                           lab,
                                           'Data.npy'), allow_pickle=True)
                npa[:, 0] = pd.to_datetime(npa[:, 0], format='%H:%M:%S.%f')
                # in case the ts is not sorted
                npa = npa[npa[:, 0].argsort()]

                if typ == 'INPUT':
                    X.append(npa)
                else:
                    y.append((lab, npa))

            self.sequences.append({'X': X, 'y': y, 'len': sum([len(y_[1]) for y_ in y])})

    def __len__(self):
        return sum([seq['len'] for seq in self.sequences])

    def __getitem__(self, idx):
        seq_idx = 0
        
        # scales with number of sequences we have (should be relatively small)
        while idx >= self.sequences[seq_idx]['len']:
            idx -= self.sequences[seq_idx]['len']
            seq_idx += 1

        # could be made constant time at the cost of memory
        for lab, y in self.sequences[seq_idx]['y']:
            if idx < len(y):
                break
            else: 
                idx -= len(y)

        ydt, yval = y[idx]

        X = np.zeros((len(self.sequences[seq_idx]['X']), self.seq_window))

        for i, seq in enumerate(self.sequences[seq_idx]['X']):
            idx = bisect.bisect_left(seq[:, 0], ydt)
            li = idx - math.ceil(self.seq_window / 2)
            ri = idx + math.floor(self.seq_window / 2)

            lo = 0
            if li < 0:
                lo = 0 - li
                li = 0

            ro = self.seq_window
            if ri >= len(seq):
                ro = self.seq_window - (ri - len(seq))
                ri = len(seq)

            X[i, lo:ro] = seq[li:ri, 1]

        return X, yval, lab


if __name__ == '__main__':
    # debug
    dataset = DassaultDataset('data/Dassault_Cache/train', 400)
    import IPython; IPython.embed(); exit(1)
