# captcha_recognition_CNN

In this project, I use PIL to generate verification code, and then build a CNN model to recognize the code. 

## Introduction

In this project, I use opencv to generate verification code, and then build a CNN model to recognize the code. Captcha is widely used in websites login process, to verify the one who's attempting to login is a real user rather than a robot. Therefore, this project can be used to build a automatic login system later.

## Methodology

1. Generate verification code using PIL (random numbers and chars, random points, random lines)
2. Peprare data to be input to CNN model (char to index, normalize, reshape)
3. Build CNN model and train (train a 10 number captcha recognition needs 6 hours using my own laptop, I also tried to train a 26 alphabets+10 numbers captcha recognition for one day, individual char accuracy improve from 10% to 50%, I think another 1 or 2 days are needed)
4. Recognize captcha using the model.

## Result

#### Verification code: 
![captcha](/code/39839.png) </br>

#### Model accuracy:


## References:
https://github.com/sajal2692/data-science-portfolio/blob/master/digit_recognition-mnist-sequence.ipynb </br>
https://groups.google.com/forum/#!topic/keras-users/UIhlW423YFs
