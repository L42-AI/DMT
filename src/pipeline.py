from analyser import Analyser
import pandas as pd

import torch

class Pipeline:
    def __init__(self, analyser: Analyser):
        self.analyser = analyser
        self.embedding_dim = 16

    def prepare(self):
        wide_format = pd.pivot_table(self.analyser.data, index=('id', 'time'), columns='variable', values='value')
        wide_format['id_col'] = wide_format.index.get_level_values(0)
        wide_format['id_col'] = wide_format['id_col'].apply(lambda x: int(x[-2:]))
        wide_format = wide_format.droplevel(0)

        id_col = wide_format.pop('id_col')
        self.X = wide_format.drop(columns=['mood', 'circumplex.valence', 'circumplex.arousal'])
        self.y = wide_format['mood']

        return torch.from_numpy(id_col.values), torch.from_numpy(self.X.values), torch.from_numpy(self.y.values)

    def train(self):
        pass
