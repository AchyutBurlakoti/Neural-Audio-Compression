import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from torch import nn

from src.data.dataset import AudioDataset
import auraloss

from src.models.model import SwarModel
from src.models.discriminators import MelMultiScaleDiscriminator
from src.utils.loss import *

# put all the data (i.e. wav files) in the /data/input/ folder
dataset = AudioDataset("./data/input/*.wav", "")
stft_loss = auraloss.freq.STFTLoss()

def collate_fn(batch):
    lengths = torch.tensor([elem.shape[-1] for elem in batch])
    data = nn.utils.rnn.pad_sequence(batch, batch_first=True)
    return data.transpose(1, 2)

dataloader = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate_fn, num_workers=0)

# here 24 represent the models max. bitrate capacity is 24 kbps but during inference we can set bitrate to any lower value other than 24 as per the need
model = SwarModel.build_model(24).to(torch.device('cuda'))

disc = MelMultiScaleDiscriminator().to(torch.device('cuda'))

generator_optim = optim.Adam(model.parameters(), lr=3 * 1e-4, betas=(0.5, 0.9))
discriminator_optim = optim.Adam(disc.parameters(), lr=1e-6, betas=(0.5, 0.9))

loss = MelGanDiscriminatorLosses()

path = "<model-name>.pt"

def save_model(model, disc, gen_optimizer, disc_optimizer):
    torch.save({
        'model_state_dict': model.state_dict(),
        'disc_state_dict': disc.state_dict(),
        'genoptimizer_state_dict': gen_optimizer.state_dict(),
        'discoptimizer_state_dict': disc_optimizer.state_dict()
    },path)

EPOCH = 800

for epoch in range(0, EPOCH):
  print("EPOCH number = %d" % epoch)
  for i, x in enumerate(iter(dataloader)):
    with torch.autograd.set_detect_anomaly(True):

      x = x.to(torch.device("cuda"))

      fake_audio, vq_loss = model(x)

      # Generator
      disc_fake = disc(fake_audio)
      disc_real = disc(x)

      gen_loss1 = stft_loss(x, fake_audio) + vq_loss
      gen_loss2 = loss.generator_loss(disc_real, disc_fake)
      gen_loss3 = loss.feature_matching_loss(disc_real, disc_fake)

      gen_loss = gen_loss1 + gen_loss2 + 100 * gen_loss3

      generator_optim.zero_grad()
      gen_loss.backward()
      generator_optim.step()

      # Discriminator
      disc_fake = disc(fake_audio.detach())
      disc_real = disc(x)
      
      disc_loss = loss.discriminator_loss(disc_real, disc_fake)

      discriminator_optim.zero_grad()
      disc_loss.backward()
      discriminator_optim.step()

      if i % 5 == 0:
        print("Training generator loss = ", gen_loss, " discriminator loss = ", disc_loss) 

      save_model(model, disc, generator_optim, discriminator_optim)