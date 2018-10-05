# Variants of RMSProp and Adagrad

Keras implementation of SC-Adagrad, SC-RMSProp and RMSProp Algorithms proposed in <a href="https://arxiv.org/abs/1706.05507" target="_blank">here</a>

Short version accepted at ICML, 17 can be found <a href="http://www.ml.uni-saarland.de/Publications/MukHei-VariantsRMSPropAdagradLogRegret.pdf" target="_blank">here</a>

I wrote a blog/tutorial <a href="https://mmahesh.github.io/articles/2017-07/tutorial-on-sc-adagrad-a-new-stochastic-gradient-method-for-deep-learning" target="_blank">here</a>,  describing Adagrad, RMSProp, Adam, SC-Adagrad and SC-RMSProp in simple terms, so that it is easy to understand the gist of the algorithms. 

# Usage

So, you created a deep network using keras, now you want to train with above algorithms. Copy the file  "new_optimizers.py" into your repository. Then in the file where the model is created (also to be compiled) add the following

```python
from new_optimizers import *

# lets for example you want to use SC-Adagrad then
# create optimizer object as follows.

sc_adagrad = SC_Adagrad()

# similarly for SC-RMSProp and RMSProp (Ours)

sc_rmsprop = SC_RMSProp()
rmsprop_variant = RMSProp_variant() 

```

Then in the code where you compile your keras model you must set ```optimizer=sc_adagrad```. You can do the same for SC-RMSProp and RMSProp algorithms.

# Overview of Algorithms 
<div align="center">
  <a href='https://raw.githubusercontent.com/mmahesh/variants_of_rmsprop_and_adagrad/master/poster_image.jpg' target='_blank'><img src="https://raw.githubusercontent.com/mmahesh/variants_of_rmsprop_and_adagrad/master/poster_image.jpg" target='_blank'></a><br><br>
</div>

-----------------

