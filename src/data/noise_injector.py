import torch
import torchaudio

import glob

import torchaudio.functional as F
from torch.nn.utils.rnn import pad_sequence

class Noiseinjector :

    def __init__(self, file_path) :

        self.filenames = glob.glob(file_path + '.wav')
        self.noise_data = None
        self.rq_sr = 16000

        self.noise_dim = []

    def _resample_file(self, wf, sr) :

        # Kaiser Fast
        return F.resample(
            wf,
            sr,
            self.rq_sr,
            lowpass_filter_width=16,
            rolloff=0.85,
            resampling_method="kaiser_window",
            beta=8.555504641634386
        )
        
    def _load_noise_data(self):

        noise_data = []
        for files in self.filenames :

            wf, sr = torchaudio.load(files)
            if sr != self.rq_sr :
                wf = self._resample_file(wf, sr)

            self.noise_dim.append(wf.shape[-1])
            data = wf[0].view(1, wf.shape[-1])
            noise_data.append(data.T)
        
        self.noise_data = pad_sequence(noise_data)
        self.noise_data = self.noise_data.T

    def save_audio(self):
        pass

    def add_noise(self, data, snr=20):
        
        data = self._resample_file(data, 22050)
        data_len = data.shape[-1]
        index = 0
        noise_len = self.noise_dim[index]
        sampled_noise_data = self.noise_data[0][index][:noise_len].view(1, noise_len)

        if noise_len > data_len:
            new_sampled_noise_data = sampled_noise_data[:, :data_len]

        else :
            difference = data_len - noise_len

            if difference > noise_len:
                times = data_len / noise_len
                print(times)

                temp_tensor = sampled_noise_data

                if int(times) == times:
                    
                    temp_tensor = [sampled_noise_data.T] * times
                    new_sampled_noise_data = pad_sequence(temp_tensor).T
                    new_sampled_noise_data = new_sampled_noise_data[0].view(1, noise_len * times)

                else :
                    count = int(times)
                    remaining = data_len - count * noise_len
                    rem_data = sampled_noise_data[:, :remaining]
                    
                    temp_tensor = [sampled_noise_data.T] * count
                    temp_tensor.append(rem_data.T)
                    new_sampled_noise_data = pad_sequence(temp_tensor).T
                    new_sampled_noise_data = torch.reshape(new_sampled_noise_data[0], (-1,)).view(1, noise_len * (count + 1))[:, :data_len]

            else :
                rem_data = sampled_noise_data[:, :difference]
                new_sampled_noise_data = pad_sequence([new_sampled_noise_data.T, rem_data.T]).T
                new_sampled_noise_data = new_sampled_noise_data[0][0][:noise_len+difference].view(1, noise_len+difference)

        noise_rms = new_sampled_noise_data.norm(p=2)
        speech_rms = data.norm(p=2)
        snr = 10 ** (snr / 20)
        scale = snr * noise_rms / speech_rms
        
        return (scale * data + new_sampled_noise_data) / 2