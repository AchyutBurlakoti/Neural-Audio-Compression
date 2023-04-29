from torch import nn
import math

from .quantization.vq import ResidualVectorQuantizer
from .encoder import Encoder
from .decoder import Decoder

from src.config_file import configuration

class SwarModel(nn.Module):

    def __init__(self, encoder, decoder, quantizer, target_bandwidth=6, sample_rate=16000, channels=1):
        super().__init__()
        self.bandwidth = None
        self.target_bandwidth = target_bandwidth
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_rate = math.ceil(self.sample_rate / configuration["vector_quantization"]["striding_factor"])

        self.bits_per_codebook = int(math.log2(self.quantizer.bins))

    def set_target_bandwidth(self, bw = configuration["model"]["inference"]["target_bandwidth"]) :
      self.bandwidth = bw

    def encode(self, x):
        return self._encode_frame(x)

    def _encode_frame(self, x):
        emb = self.encoder(x)
        codes = self.quantizer.encode(emb, self.frame_rate, self.bandwidth)
        codes = codes.transpose(0, 1)
        return codes

    def decode(self, encoded_frames):
        return self._decode_frame(encoded_frames)

    def _decode_frame(self, encoded_frame):
        codes = encoded_frame
        codes = codes.transpose(0, 1)
        emb = self.quantizer.decode(codes)
        out = self.decoder(emb)
        return out

    def forward(self, x):
        emb = self.encoder(x)
        codes, _, _, loss = self.quantizer(emb, self.frame_rate, self.bandwidth)
        recon = self.decoder(codes)
        return recon, loss

    @staticmethod
    def build_model(target_bandwidth, sample_rate=configuration["sampling_rate"]):
        encoder = Encoder()
        decoder = Decoder()

        n_q = int(1000 * target_bandwidth // (math.ceil(sample_rate / configuration["vector_quantization"]["striding_factor"]) * 10))

        quantizer = ResidualVectorQuantizer(dimension=configuration["vector_quantization"]["dimension"], n_q=n_q, bins=1024)
        model = SwarModel(encoder, decoder, quantizer)
        return model