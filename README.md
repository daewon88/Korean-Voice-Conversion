# Korean-Voice-Conversion

Korean Voice Conversion using Cycle-GAN.  
It is tensorflow 2.0 implementation of the paper : ["CycleGAN-VC Parallel-Data-Free Voice Conversion Using Cycle-Consistent Adversarial Networks"](https://arxiv.org/abs/1711.112930) (I changed some hyper-parameters to apply to Korean-dataset)  
You can listen converted voices in *Sample directory*

# Dependancies

* librosa
* pyworld
* tensorflow 2.2
* tensorflow addons

# Usage

## Dataset

Download Korean dataset here.(https://ithub.korean.go.kr/user/total/referenceManager.do)  
I used Man's voice and Woman's voice in 20s.

## Train
Locate train dataset and validation dataset to designated folders.(train dataset : train_A, train_B / validation dataset : validation_A, validation_B). Before starting training, preprocessing is carried out. (It may take quite a long time)  
Trainging takes at least 100 epochs to achieve good quality results.

    !python main.py
  
## Test
Locate test dataset to designated folders.(test_A, test_B)

    !python test.py
    
# Reference

* https://arxiv.org/abs/1711.112930
* https://github.com/pritishyuvraj/Voice-Conversion-GAN : pytorch implemantation. I referenced the preprocessing code.

    
    
 
    
 








