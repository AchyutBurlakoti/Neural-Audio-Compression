import io
import json
import struct
import typing as tp

_AUDIO_CODEC_MAGIC = b'NAC'

_codec_header_struct = struct.Struct('!3sBIB')

def write_nac_header(fo: tp.IO[bytes], bit_rate: int, metadata: tp.Any):
    meta_dumped = json.dumps(metadata).encode('utf-8')
    version = 0
    if not bit_rate:
        bit_rate = 3
    header = _codec_header_struct.pack(_AUDIO_CODEC_MAGIC, version, len(meta_dumped), bit_rate)
    fo.write(header)
    fo.write(meta_dumped)
    fo.flush()

def _read_exactly(fo: tp.IO[bytes], size: int) -> bytes:
    buf = b""
    while len(buf) < size:
        new_buf = fo.read(size)
        if not new_buf:
            raise EOFError("Impossible to read enough data from the stream, ",
                           f"{size} bytes remaining.")
        buf += new_buf
        size -= len(new_buf)
    return buf

def read_nac_header(fo: tp.IO[bytes]):
    header_bytes = _read_exactly(fo, _codec_header_struct.size)
    magic, version, meta_size, bit_rate = _codec_header_struct.unpack(header_bytes)
    if magic != _AUDIO_CODEC_MAGIC:
        raise ValueError()
    if version != 0:
        raise ValueError()
    meta_bytes = _read_exactly(fo, meta_size)
    return bit_rate, json.loads(meta_bytes.decode('utf-8'))

class BitPacker:
    """Simple bit packer to handle ints with a non standard width, e.g. 10 bits.
    Note that for some bandwidth (1.5, 3), the codebook representation
    will not cover an integer number of bytes.
    Args:
        bits (int): number of bits per value that will be pushed.
        fo (IO[bytes]): file-object to push the bytes to.
    """
    def __init__(self, bits: int, fo: tp.IO[bytes]):
        self._current_value = 0
        self._current_bits = 0
        self.bits = bits
        self.fo = fo

    def push(self, value: int):
        """Push a new value to the stream. This will immediately
        write as many uint8 as possible to the underlying file-object."""
        self._current_value += (value << self._current_bits)
        self._current_bits += self.bits
        while self._current_bits >= 8:
            lower_8bits = self._current_value & 0xff
            self._current_bits -= 8
            self._current_value >>= 8
            self.fo.write(bytes([lower_8bits]))

    def flush(self):
        """Flushes the remaining partial uint8, call this at the end
        of the stream to encode."""
        if self._current_bits:
            self.fo.write(bytes([self._current_value]))
            self._current_value = 0
            self._current_bits = 0
        self.fo.flush()

class BitUnpacker:
    """BitUnpacker does the opposite of `BitPacker`.
    Args:
        bits (int): number of bits of the values to decode.
        fo (IO[bytes]): file-object to push the bytes to.
        """
    def __init__(self, bits: int, fo: tp.IO[bytes]):
        self.bits = bits
        self.fo = fo
        self._mask = (1 << bits) - 1
        self._current_value = 0
        self._current_bits = 0

    def pull(self) -> tp.Optional[int]:
        """
        Pull a single value from the stream, potentially reading some
        extra bytes from the underlying file-object.
        Returns `None` when reaching the end of the stream.
        """
        while self._current_bits < self.bits:
            buf = self.fo.read(1)
            if not buf:
                return None
            character = buf[0]
            self._current_value += character << self._current_bits
            self._current_bits += 8

        out = self._current_value & self._mask
        self._current_value >>= self.bits
        self._current_bits -= self.bits
        return out

if __name__ == "__main__":
    import torch
    data = torch.randint(2 ** 10, (2, 10))
    print(data)
    fo = io.BytesIO()
    packer = BitPacker(10, fo)
    tokens = data.view(20).tolist()
    for token in tokens:
        packer.push(token)
    packer.flush()
    fo.seek(0)
    unpacker = BitUnpacker(10, fo)
    rebuilt = []
    while True:
        value = unpacker.pull()
        if value is None:
            break
        rebuilt.append(value)
    final_data = torch.Tensor(rebuilt).int()
    print(final_data.view(2, 10))
