"""
This file contains the keras implementations of all the optimizers
proposed in 

Variants of RMSProp and Adagrad with Logarithmic Regret Bounds
(http://arxiv.org/abs/1706.05507), M.C. Mukkamala and M. Hein

I used the format used in keras optimizers.py file.
I appreciate all the authors who contributed to keras.

"""

from keras import backend as K
from keras.optimizers import *

class SC_Adagrad(Optimizer):
    """SC-Adagrad optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        xi_1: float, 0 < xi_1 < 1. Generally close to 1.
        xi_2: float, 0 < xi_2 < 1. Generally close to 1.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [Variants of RMSProp and Adagrad with Logarithmic Regret Bounds](http://arxiv.org/abs/1706.05507)
    """

    def __init__(self, lr=0.01, xi_1=0.1, xi_2=0.1,
                 decay=0., **kwargs):
        super(SC_Adagrad, self).__init__(**kwargs)
        self.iterations = K.variable(0, name='iterations')
        self.lr = K.variable(lr, name='lr')
        self.xi_1 = K.variable(xi_1, name='xi_1')
        self.xi_2 = K.variable(xi_2, name='xi_2')
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))


        vs = [K.zeros(K.get_variable_shape(p)) for p in params]
        self.weights = [self.iterations]+ vs

        for p, g, v in zip(params, grads, vs):
            v_t = v + K.square(g)
            p_t = p - self.lr * g / (v_t + self.xi_2*K.exp(-self.xi_1*v_t) )

            self.updates.append(K.update(v, v_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'xi_1': float(K.get_value(self.xi_1)),
                  'xi_2': float(K.get_value(self.xi_2)),
                  'decay': float(K.get_value(self.decay))}
        base_config = super(SC_Adagrad, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class SC_RMSProp(Optimizer):
    """SC-RMSProp optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        xi_1: float, 0 < xi_1 . Fuzz factor params. 
        xi_2: float, 0 < xi_2 . Fuzz factor params.
        gamma: float >= 0. Weighing factor.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [Variants of RMSProp and Adagrad with Logarithmic Regret Bounds](http://arxiv.org/abs/1706.05507)
    """

    def __init__(self, lr=0.01, xi_1=0.1, xi_2=0.1,
                 gamma=0.9, decay=0., **kwargs):
        super(SC_RMSProp, self).__init__(**kwargs)
        self.iterations = K.variable(0, name='iterations')
        self.lr = K.variable(lr, name='lr')
        self.xi_1 = K.variable(xi_1, name='xi_1')
        self.xi_2 = K.variable(xi_2, name='xi_2')
        self.gamma = K.variable(gamma, name='gamma')
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1

        vs = [K.zeros(K.get_variable_shape(p)) for p in params]
        self.weights = [self.iterations]+ vs

        for p, g, v in zip(params, grads, vs):
            v_t = (1-(self.gamma/t))*v + (self.gamma/t)*K.square(g)
            p_t = p - self.lr * g / (t*v_t + self.xi_2*K.exp(-self.xi_1*t*v_t) )

            self.updates.append(K.update(v, v_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'xi_1': float(K.get_value(self.xi_1)),
                  'xi_2': float(K.get_value(self.xi_2)),
                  'gamma': float(K.get_value(self.gamma)),
                  'decay': float(K.get_value(self.decay))}
        base_config = super(SC_RMSProp, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RMSProp_variant(Optimizer):
    """RMSProp optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        gamma: 0 < gamma <= 1, Weighing factor.
        delta: float >= 0. Fuzz factor.
        decay: float >= 0. Learning rate decay over each update.

    # References
        - [Variants of RMSProp and Adagrad with Logarithmic Regret Bounds](http://arxiv.org/abs/1706.05507)
    """

    def __init__(self, lr=0.01, delta=1e-8, 
                 gamma=0.9, decay=0., **kwargs):
        super(RMSProp_variant, self).__init__(**kwargs)
        self.iterations = K.variable(0, name='iterations')
        self.lr = K.variable(lr, name='lr')
        self.delta = K.variable(delta, name='delta')
        self.gamma = K.variable(gamma, name='gamma')
        self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay

    def get_updates(self, params, constraints, loss):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * self.iterations))

        t = self.iterations + 1

        vs = [K.zeros(K.get_variable_shape(p)) for p in params]
        self.weights = [self.iterations]+ vs

        for p, g, v in zip(params, grads, vs):
            v_t = (1-(self.gamma/t))*v + (self.gamma/t)*K.square(g)
            p_t = p - self.lr * g / (K.sqrt(t*v_t) + self.delta )

            self.updates.append(K.update(v, v_t))

            new_p = p_t
            # apply constraints
            if p in constraints:
                c = constraints[p]
                new_p = c(new_p)
            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'delta': float(K.get_value(self.delta)),
                  'gamma': float(K.get_value(self.gamma)),
                  'decay': float(K.get_value(self.decay))}
        base_config = super(RMSProp_variant, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))