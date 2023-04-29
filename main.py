from src.models.model import SwarModel
from src.file_format.fileformat import BitPacker, BitUnpacker
import torch
import io

model = SwarModel.build_model(24).to(torch.device('cuda'))

path = "./models/" + <model_name>

checkpoint = torch.load(path)
model.load_state_dict(checkpoint["model_state_dict"])

import torchaudio

wf, sr = torchaudio.load('<path-to-audio-file>')

model.set_target_bandwidth(24) # 24 represent that the target bandwidth is 24kbps

output = model._encode_frame(wf.view(1, 1, wf.shape[-1]).to(torch.device('cuda')))[0]

output = torch.flatten(output).tolist()

fo = io.BytesIO()
packer = BitPacker(10, fo)

for token in output:
    packer.push(token)

packer.flush()
fo.seek()

with open('<file-name>.nac', 'wb') as outputfile:  # saving compressed audio to custom file format
    outputfile.write(fo.getbuffer())

