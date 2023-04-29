from fastapi import UploadFile, File, FastAPI
from api.endpoints.models import OutputData
from src.file_format.fileformat import BitUnpacker, read_nac_header
import io
import torch
import math

from decompress import *
from src.models.model import SwarModel
from starlette.middleware.cors import CORSMiddleware as CORSMiddleware  # noqa

path = "./models/neural-audio-codec.pt"

checkpoint = torch.load(path)
model = SwarModel.build_model(24).to(torch.device('cuda'))
model.load_state_dict(checkpoint['model_state_dict'])

api_router = FastAPI()

origins = ["*"]

api_router.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@api_router.post("/", response_model=OutputData)
async def root(input: UploadFile=File(...)):
    content = await input.read()
    buf = io.BytesIO(content)
    buf.seek(0)

    try:
        bit_rate, _ = read_nac_header(buf)
        n_q = int(1000 * bit_rate // (math.ceil(16000 / 320) * 10))
        model.set_target_bandwidth(bit_rate)
    except ValueError():
        print("Given file is not .nac")

    unpacker = BitUnpacker(10, buf)
    rebuilt = []
    while True:
        value = unpacker.pull()
        if value is None:
            break
        rebuilt.append(value)
    c = len(rebuilt) / 100
    rebuilt_tensor = torch.Tensor(rebuilt).int()

    if rebuilt_tensor.shape[-1] > 4800:
        wav = torch.empty(1, 0).to(torch.device('cuda'))
        for i in range(0, 26): # 26 for 2 * 26 s clip
            offset = i * n_q * 100
            data = rebuilt_tensor[offset:offset+4800].to(torch.device('cuda'))
            output = model._decode_frame(data.view(1, n_q, 100).long())
            wav = torch.cat([wav, output.view(1, 32000)], dim=1)
        decompressed = wav.tolist()
        print("decompressed")
    else:
        rebuilt_tensor = rebuilt_tensor.view(1, int(c), 100)
        rebuilt_tensor = rebuilt_tensor.to(torch.device('cuda'))
        decompressed = model._decode_frame(rebuilt_tensor)
        decompressed = decompressed.view(1, decompressed.shape[-1])
        decompressed = decompressed.tolist()
    return OutputData(waveform=decompressed, message=input.filename)