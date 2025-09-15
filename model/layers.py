import tensorflow as tf
from tensorflow.keras import backend as K
import keras.random as krandom
import keras.ops as kops
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.layers.merge import _Merge
#from tensorflow.python.framework.ops import disable_eager_execution

#disable_eager_execution()

def gradient_penalty(self, batch_size, real_images, fake_images):
    """Calculates the gradient penalty.

    This loss is calculated on an interpolated image
    and added to the discriminator loss.
    """
    # Get the interpolated image
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        # 1. Get the discriminator output for this interpolated image.
        pred = self.discriminator(interpolated, training=True)

    # 2. Calculate the gradients w.r.t to this interpolated image.
    grads = gp_tape.gradient(pred, [interpolated], unconnected_gradients="zero")[0]
    # 3. Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list, unconnected_gradients=tf.UnconnectedGradients.ZERO)
    return [
        grad if grad is not None else tf.zeros_like(var)
        for var, grad in zip(var_list, grads)
    ]


class GradientPenalty(Layer):
    def __init__(self, **kwargs):
        super(GradientPenalty, self).__init__(**kwargs)

    def call(self, inputs):
        
        target, wrt = inputs
        grad = _compute_gradients(target, [wrt])[0]

        return K.sqrt(K.sum(K.batch_flatten(K.square(grad)), axis=1, keepdims=True)) - 1
        
    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], 1)


class RandomWeightedAverage(_Merge):
    def build(self, input_shape):
        super(RandomWeightedAverage, self).build(input_shape)
        if len(input_shape) != 2:
            raise ValueError(
                "A `RandomWeightedAverage` layer should be "
                "called on exactly 2 inputs"
            )

    def _merge_function(self, inputs):
        if len(inputs) != 2:
            raise ValueError(
                "A `RandomWeightedAverage` layer should be "
                "called on exactly 2 inputs"
            )

        x, y = inputs
        shape = K.shape(x)
        weights = K.random_uniform(shape[:1], 0, 1)
        for i in range(len(K.int_shape(x)) - 1):
            weights = K.expand_dims(weights, -1)
        return x * weights + y * (1 - weights)


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (
            s[0],
            None if s[1] is None else s[1] + 2 * self.padding[0],
            None if s[2] is None else s[2] + 2 * self.padding[1],
            s[3],
        )

    def call(self, x):
        i_pad, j_pad = self.padding
        return tf.pad(x, [[0, 0], [i_pad, i_pad], [j_pad, j_pad], [0, 0]], "REFLECT")


class SymmetricPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(SymmetricPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (
            s[0],
            None if s[1] is None else s[1] + 2 * self.padding[0],
            None if s[2] is None else s[2] + 2 * self.padding[1],
            s[3],
        )

    def call(self, x):
        i_pad, j_pad = self.padding
        return tf.pad(x, [[0, 0], [i_pad, i_pad], [j_pad, j_pad], [0, 0]], "SYMMETRIC")
