# CNN based spoof speech detection
Test different Convolution Neural Networks for spoof speech detection task on [ASV spoof 2019](https://www.asvspoof.org/index2019.html) dataset.

This repository contains the code for:

- feature computation (melspectrogram and logmelspectrogram) </li>
-  train and testing for 3 different CNN based architectures:
    - fine tuning of VGGish [1], i.e. weights are initialized using VGGish pretrained model;
    - CNN architecture proposed in [2] ;
    - modification of CNN architecture proposed in [2] changing the input (different feature computation parameters)
- train and testing is done both for binary case (bonafide VS spoof) and for multiclass case (identify which TTS algorithm has been used)

    

 
 [1] [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish)
 
 [2]: ["Hello? Who Am I Talking to?" A Shallow CNN Approach for Human vs. Bot Speech Classification", <em> A. Lieto; D. Moro; F. Devoti; C. Parera; V. Lipari; P. Bestagini; S. Tubaro </em> ICASSP 2019 - 2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP),](https://ieeexplore.ieee.org/abstract/document/8682743)