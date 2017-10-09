# ContourGAN
Tensorflow for ContourGAN: Image Contour Detection with Generative Adversarial Encoder-Decoder Networks

### Introduction
In this paper, we developed an encoder-decoder convolutional framework to extract images contour and generative adversarial models are also intriduced to produce high-quality edge information. Since traditional contour detection methods directly upsampled feature maps to images scale and the computed intermediate related informations are ignored. In addition, most of image-to-image models only take the loss between predicted results and ground truth into consideration. Based on these observations, we utilze all computed feature maps in enoder stage. For the more, generative adversarial models were introduced to further improve detection performance.In conclusion, our proposed contour detection methods (ContourGAN) contains two stages: the first stage is an encoder-decoder model whose weights updated from a binary cross entropy loss (BCE), fine-tuning from VGG16. In the second stage, we introduce a discriminator network which employ the corresponding ground truth and predicted contour results as input to discriminate them. Our The experiments results  achieve state-of-the-art performance on BSD500 datasets achieving ODS F-measure of \textbf{0.823}, demonstrated that our proposed model based on adversarial and BCE loss extremely outperform others model.

## Framework

![alt tag](images/frame.jpg)

## Results on BSDS500

![alt tag](images/results.jpg)


## Prerequisites

- Python 2.7
- [TensorFlow==0.12.1+](https://www.tensorflow.org/)
- [TensorLayer==1.4+](https://github.com/zsdonghao/tensorlayer)
