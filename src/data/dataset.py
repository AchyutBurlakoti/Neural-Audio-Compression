from torch.utils.data import Dataset
import glob

import torchaudio
from torch import Tensor

class AudioDataset(Dataset):

    def __init__(self, file_path, dataset_name=None) :
        super().__init__()

        self.filenames = glob.glob(file_path)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index) -> Tensor:
        wf, sr = torchaudio.load(self.filenames[index])
        assert sr == 16000
        return wf.T