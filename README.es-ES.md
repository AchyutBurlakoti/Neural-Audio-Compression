

## Neural Audio Codec
Los códecs de audio neural que utilizan enfoques de extremo a extremo han ganado popularidad debido a su capacidad para aprender representaciones de audio eficientes mediante métodos basados en datos, sin depender de componentes de procesamiento de señales creados manualmente. Estos códecs utilizan redes autoencoder con cuantización de características ocultas y se han aplicado en trabajos iniciales para codificación de voz, así como en estudios más recientes, donde se utilizó una red convolucional profunda para la compresión de voz. Aunque la mayoría de estos trabajos se centran en la codificación de voz a bajas tasas de bits, varios estudios han demostrado la compresión eficiente de audio utilizando redes neuronales.

Descargue el modelo desde: https://drive.google.com/file/d/1xc7-heD1JIf2BOA02Ta5YUpJguDSmGZY/view?usp=sharing
Para más información, por favor lea el report.pdf en /reports/

### Arquitectura
![image](https://user-images.githubusercontent.com/52134359/234917869-ae54487f-12fb-491a-8a14-48cec4d00aee.png)

### Cuantización
La cuantización es un proceso fundamental en la compresión de datos, y su función principal es discretizar un espacio latente continuo mediante la preparación de un codebook. En la compresión de audio e imágenes, la cuantización se utiliza comúnmente para representar datos de alta dimensión con embeddings de menor dimensión. El cuantizador prepara el codebook para estos embeddings, permitiéndonos almacenar el índice de su vecino más cercano en el codebook. Este proceso se llama cuantización vectorial y consiste en agrupar embeddings similares en clústeres. El codebook consiste en el centroide de cada clúster, representado por un símbolo discreto.

Existen limitaciones en el uso de la Cuantización Vectorial y, para eliminar esos problemas, decidimos utilizar la Cuantización Vectorial Residual (RVQ). La RVQ cuenta con la característica adicional de tasa de bits adaptable. Esto significa que, en lugar de utilizar todo el codebook del cuantizador, puedes usar cualquier número de codebooks, pero con un trade-off en la calidad.

https://user-images.githubusercontent.com/52134359/234917299-9f92508f-e82b-4987-a461-1f894a9f5eff.mp4

### Entrenamiento del modelo con tus datos desde cero
* Coloque todos sus archivos .wav en /data/input/
* Ejecute los siguientes comandos en el directorio raíz:
```
pip install .
python train.py
```

Toda otra funcionalidad relacionada con el uso del modelo se puede encontrar en el directorio raíz y en la carpeta /src/

### Las fuentes utilizadas para la finalización del proyecto son:
* [SoundStream An End to End Neural Audio Codec.pdf](https://github.com/AchyutBurlakoti/Neural-Audio-Compression/files/11345797/SoundStream.An.End.to.End.Neural.Audio.Codec.pdf)
* [High Fidelity Neural Audio Compression.pdf](https://github.com/AchyutBurlakoti/Neural-Audio-Compression/files/11345808/High.Fidelity.Neural.Audio.Compression.pdf)
* [Unsupervised speech representation learning.pdf](https://github.com/AchyutBurlakoti/Neural-Audio-Compression/files/11345811/Unsupervised.speech.representation.learning.pdf)

### Usos Futuros

El modelo desarrollado es una red autoasociativa que aprende la representación de los datos a través de la compresión de esos datos de alta dimensión en un espacio latente discreto (es decir, mediante cuantización vectorial), por lo que el modelo conoce muy bien la representación de los datos de audio y puede utilizarse posteriormente en otras tareas posteriores (downstream) como texto a voz, generación de audio y otras formas de modelado de audio.

### Mejoras Adicionales que se pueden realizar
* No fue posible probar el modelo con el discriminador de Hifi-GAN debido a la falta de capacidad de memoria, por lo que cualquiera puede intentar hacerlo, ya que el código para el entrenamiento de Hifi-GAN también se proporciona en la carpeta /src/
* Las puntuaciones MUSHRA calculadas aún no representan bien la eficiencia del modelo debido a la falta de una configuración experimental para el cálculo de las puntuaciones MUSHRA.


## Formato de Archivo Personalizado (.nac)

Al igual que otros códecs de audio como mp3 o flac, que requieren su propio formato de archivo, nuestro códec de audio neural también tiene su propio formato llamado .nac (neural audio codec).

* Orden de bytes: network big endian
* Formato de encabezado (9 bytes):
  * 3 bytes: cadena mágica
  * 1 byte: número de versión
  * 4 bytes: longitud de metadatos
  * 1 byte: tasa de bits
  
 ![image](https://user-images.githubusercontent.com/52134359/234928959-726e0d3d-93cd-4bf7-b0fa-709781ce96b6.png)
 
 ## Resultados
 
 Los siguientes resultados muestran la reconstrucción del audio cuando se comprime a una tasa de bits de 24 kbps, es decir, solo se necesitan 24000 bits para representar un clip de audio de 1 segundo, lo que equivale a un total de 2.9 KB para una forma de onda de audio a 16000 Hz.
 
### Audio de voz
  * Audio original (1.63 MB en formato .wav)

https://user-images.githubusercontent.com/52134359/234932740-34b00361-937d-4d56-b47a-a4bf9522c9bd.mp4

  * Audio reconstruido a 24000 kbps (153 KB en formato .nac)
  
https://user-images.githubusercontent.com/52134359/234932104-245b4e9a-638b-4527-98f8-4475c65b3426.mp4

### Audio de piano
  * Audio original 
 

https://user-images.githubusercontent.com/52134359/234934838-479839db-8f6c-48c7-9302-e4634f75be26.mp4


  * Audio reconstruido
 
https://user-images.githubusercontent.com/52134359/234933083-298adb7d-e650-4548-825c-179c9d380635.mp4
