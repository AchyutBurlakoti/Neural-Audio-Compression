import torch
import torchaudio

import torchaudio.functional as F
from torch import Tensor
from typing import Tuple

def resample(audio_file, audio_sr) -> Tuple[Tensor, int]:
    wf, sr = torchaudio.load(audio_file)
    return F.resample(
        wf, 
        sr, 
        audio_sr,
        lowpass_filter_width=16,
        rolloff=0.85,
        resampling_method="kaiser_window",
        beta=8.555504641634386
    ), audio_sr
