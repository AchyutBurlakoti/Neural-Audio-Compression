## Neural Audio Codec
Neural audio codecs that use end-to-end approaches have gained popularity due to their
ability to learn efficient audio representations through data-driven methods, without relying
on handcrafted signal processing components. These codecs utilize autoencoder networks
with quantization of hidden features, and have been applied in early works for speech coding
, as well as in more recent studies, where a deep convolutional network was used for
speech compression. While most of these works target speech coding at low bitrates,
several studies have demonstrated the efficient compression of audio using neural networks.

### Architecture
![image](https://user-images.githubusercontent.com/52134359/234917869-ae54487f-12fb-491a-8a14-48cec4d00aee.png)

### Quantization
Quantization is a fundamental process in data compression, and its main job is to discretize a continuous latent space by preparing a codebook. In audio and image compression,
quantization is commonly used to represent high-dimensional data with lower-dimensional
embeddings. The quantizer prepares the codebook for these 
embeddings, allowing us to store
the index of their nearest neighbor in the codebook. This process is called vector quantization, and it involves grouping similar embeddings together into clusters. The codebook
consists of the centroid of each cluster, which is represented by a discrete symbol.

There exist the limitation in the uses of Vector Quantization and in order to eliminate those issues we decided to use Residual Vector Quantization. The RVQ comes with the additional feature of adaptive bitrate. It means that
instead of using all quantizer codebook you can use the any number of codebook but with the trade of quality.

https://user-images.githubusercontent.com/52134359/234917299-9f92508f-e82b-4987-a461-1f894a9f5eff.mp4

### Traning Model with your data from scratch
* Put all your .wav files in the /data/input/
* run following commands in the root dir :
```
pip install .
python train.py
```

All other functionality related to the model's uses can be found the root dir and /src/ folder

### Sources that are used for the completion of the projects are :
* [SoundStream An End to End Neural Audio Codec.pdf](https://github.com/AchyutBurlakoti/Neural-Audio-Compression/files/11345797/SoundStream.An.End.to.End.Neural.Audio.Codec.pdf)
* [High Fidelity Neural Audio Compression.pdf](https://github.com/AchyutBurlakoti/Neural-Audio-Compression/files/11345808/High.Fidelity.Neural.Audio.Compression.pdf)
* [Unsupervised speech representation learning.pdf](https://github.com/AchyutBurlakoti/Neural-Audio-Compression/files/11345811/Unsupervised.speech.representation.learning.pdf)

### Result and Discussion

The model was compared with standard audio codecs such as OPUS, and EVS with two
standard metrics ViSQOL and MUSHRA.
On average from all 4 bit-rates (3, 6, 12, 24 kbps) we obtain the MUSHRA score of 47.47±0.6
out of 100. For 12 and 24 kbps we obtain the MUSHRA score above 50 which is acceptable
but not that great but for 3 and 6 kbps we obtain above 40 and at 6 kbps our model beats
the OPUS. As the MUSHRA score is a subjective evaluation and needs proper guidelines to
be followed we decide to also calculate the objective score by using ViSQOL. If the ViSQOL
score of the audio is above 3 then it is acceptable otherwise audio quality is considered bad
quality. Our model obtains an above 3 score only for 24 kbps which shows that further
improvement needs to be done to increase the score.

### Future Uses

The developed model is a self-associative network which learns the representation of the data through the compression of those high dimensional data in the discrete latent space (i.e. through vector quantization) so model knows
the audio data representation very well and can be further used in other underlying downstream tasks such as text-to-speech, audio generation and other form of audio modeling.

### Further Improvement that can be done
* Unable to try the model with the discriminator of Hifi-GAN due to lack of memory capacity so anyone can try it out as the code for Hifi-GAN traning is also provided in the /src/ folder
* The calculated MUSHRA scores still doesn't represent the model efficiency well due to lack of experimentation setup for MUSHRA score calculation.

