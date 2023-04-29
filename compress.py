import argparse
import glob
import torchaudio
import torch

from src.models.model import SwarModel
from src.file_format.fileformat import BitPacker, write_nac_header
import io
import time


parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", default="./data/input/", help="Provide directory path where uncompressed wav file exist")
parser.add_argument("--output", "-o", default="./data/output/", help="Provide directory path where compressed nac file should go")
parser.add_argument("--model", "-m", default="./models/neural-audio-codec.pt", help="Model path")
parser.add_argument("--target", "-t", type=int, default=24, help="Target bitrate")

args = parser.parse_args()

print(args.input)

filenames = glob.glob(args.input + "*.wav")
model = SwarModel.build_model(24).to(torch.device('cuda'))

checkpoint = torch.load(args.model)
model.load_state_dict(checkpoint["model_state_dict"])

for file in filenames:
    wf, sr = torchaudio.load(file)

    model.set_target_bandwidth(args.target)
    wf = wf.view(1, 1, wf.shape[-1])
    wf = wf.to(torch.device('cuda'))

    if wf.shape[-1] > 32000:
        audio_wav = torch.empty(0).to(torch.device('cuda')).to(torch.int32)

        for i in range(0, 26):
            offset = i * 32000
            audio = wf[:, :, offset:offset+32000]
            output = model._encode_frame(audio)[0]
            audio_wav = torch.cat([audio_wav, torch.flatten(output)], dim=0)
        
        output = audio_wav.tolist()
    else:

        output = model._encode_frame(wf).to(torch.device('cuda'))[0]

        output = torch.flatten(output).tolist()

    fo = io.BytesIO()
    write_nac_header(fo, args.target, None)
    packer = BitPacker(10, fo)

    for token in output:
        packer.push(token)

    packer.flush()

    new_file_name = args.output + "/" + file.split("\\")[-1].split('.')[0] + ".nac"

    print(file + "  ==>  ", new_file_name)

    with open(new_file_name, 'wb') as outputfile:
        outputfile.write(fo.getbuffer())