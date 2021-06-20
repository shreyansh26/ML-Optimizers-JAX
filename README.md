## ML Optimizers from scratch using JAX

Implementations of some popular optimizers from scratch for a simple model i.e., Linear Regression on a dataset of 5 features. The goal of this project was to understand how these optimizers work under the hood and try to to do a toy implementation myself. I also use a bit of JAX magic to perform the differentiation of the loss function w.r.t to the weights and the bias without explicitly writing their derivatives as a separate function. This can help to generalize this notebook for other types of loss functions as well.

![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1ZF-DWGZocky0omUz92dlHhgTsTPNavKt/view?usp=sharing]


The optimizers I have implemented are - 
* Batch Gradient Descent
* Batch Gradient Descent + Momentum
* Nesterov Accelerated Momentum
* Adagrad
* RMSprop
* Adam
* Adamax
* Nadam
* Adabelief

References -
* https://ruder.io/optimizing-gradient-descent/
* https://theaisummer.com/optimization/