## Neural Audio Codec
Neural audio codecs that use end-to-end approaches have gained popularity due to their
ability to learn efficient audio representations through data-driven methods, without relying
on handcrafted signal processing components. These codecs utilize autoencoder networks
with quantization of hidden features, and have been applied in early works for speech coding
, as well as in more recent studies, where a deep convolutional network was used for
speech compression. While most of these works target speech coding at low bitrates,
several studies have demonstrated the efficient compression of audio using neural networks.

Download model from : https://drive.google.com/file/d/1xc7-heD1JIf2BOA02Ta5YUpJguDSmGZY/view?usp=sharing
For more information please read the report.pdf from /reports/

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

### Future Uses

The developed model is a self-associative network which learns the representation of the data through the compression of those high dimensional data in the discrete latent space (i.e. through vector quantization) so model knows
the audio data representation very well and can be further used in other underlying downstream tasks such as text-to-speech, audio generation and other form of audio modeling.

### Further Improvement that can be done
* Unable to try the model with the discriminator of Hifi-GAN due to lack of memory capacity so anyone can try it out as the code for Hifi-GAN traning is also provided in the /src/ folder
* The calculated MUSHRA scores still doesn't represent the model efficiency well due to lack of experimentation setup for MUSHRA score calculation.


## Custom File Format (.nac)

As with the other audio codecs like mp3, flacc which requires their own file format, our neural audio codec also have it's own file format called .nac (neural audio codec).

* Byte order : network big endian
* Header format (9 bytes) :
  * 3 bytes: magic string
  * 1 byte : version number
  * 4 bytes: metadata length
  * 1 byte : bit rate
  
 ![image](https://user-images.githubusercontent.com/52134359/234928959-726e0d3d-93cd-4bf7-b0fa-709781ce96b6.png)
 
 ## Result
 
 The following results are the reconstruction of the audio when they are compressed at 24 kbps bitrate i.e. only 24000 bits are need to represent 1s audio clip which is in total 2.9 KB for 16000Hz audio waveform.
 
### speech audio
  * original audio (1.63 MB in .wav format)

https://user-images.githubusercontent.com/52134359/234932740-34b00361-937d-4d56-b47a-a4bf9522c9bd.mp4

  * reconstructed 24000 kbps audio (153 KB in .nac format)
  
https://user-images.githubusercontent.com/52134359/234932104-245b4e9a-638b-4527-98f8-4475c65b3426.mp4

### piano audio
  * original audio 
 

https://user-images.githubusercontent.com/52134359/234934838-479839db-8f6c-48c7-9302-e4634f75be26.mp4


  * reconstructed audio
 
https://user-images.githubusercontent.com/52134359/234933083-298adb7d-e650-4548-825c-179c9d380635.mp4
