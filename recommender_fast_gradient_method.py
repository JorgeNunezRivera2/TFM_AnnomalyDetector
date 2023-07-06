"""The Fast Gradient Method attack."""

import numpy as np
import tensorflow as tf

from cleverhans.tf2.utils import optimize_linear, compute_gradient
# Due to performance reasons, this function is wrapped inside of tf.function decorator.
# Not using the decorator here, or letting the user wrap the attack in tf.function is way
# slower on Tensorflow 2.0.0-alpha0.
#@tf.function
def recommender_compute_gradient(model_fn, loss_fn, users,deep_features, y, targeted):
    """
    Computes the gradient of the loss with respect to the input tensor.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param loss_fn: loss function that takes (labels, logits) as arguments and returns loss.
    :param x: input tensor
    :param y: Tensor with true labels. If targeted is true, then provide the target label.
    :param targeted:  bool. Is the attack targeted or untargeted? Untargeted, the default, will
                      try to make the label incorrect. Targeted will instead try to move in the
                      direction of being more like y.
    :return: A tensor containing the gradient of the loss with respect to the input tensor.
    """

    with tf.GradientTape() as g:
        g.watch(deep_features)
        # Compute loss
        loss = loss_fn(y, model_fn([users,deep_features]))
        if (
            targeted
        ):  # attack is targeted, minimize loss of target label rather than maximize loss of correct label
            loss = -loss

    # Define gradient of loss wrt input
    grad = g.gradient(loss, deep_features)
    return grad

def recommender_fast_gradient_method(
    model_fn,
    users,
    deep_features,
    eps,
    norm,
    loss_fn=None,
    clip_min=None,
    clip_max=None,
    y=None,
    targeted=False,
    sanity_checks=False,
):
    """
    Tensorflow 2.0 implementation of the Fast Gradient Method.
    :param model_fn: a callable that takes an input tensor and returns the model logits.
    :param x: input tensor.
    :param eps: epsilon (input variation parameter); see https://arxiv.org/abs/1412.6572.
    :param norm: Order of the norm (mimics NumPy). Possible values: np.inf, 1 or 2.
    :param loss_fn: (optional) callable. Loss function that takes (labels, logits) as arguments and returns loss.
                    default function is 'tf.nn.sparse_softmax_cross_entropy_with_logits'
    :param clip_min: (optional) float. Minimum float value for adversarial example components.
    :param clip_max: (optional) float. Maximum float value for adversarial example components.
    :param y: (optional) Tensor with true labels. If targeted is true, then provide the
              target label. Otherwise, only provide this parameter if you'd like to use true
              labels when crafting adversarial samples. Otherwise, model predictions are used
              as labels to avoid the "label leaking" effect (explained in this paper:
              https://arxiv.org/abs/1611.01236). Default is None.
    :param targeted: (optional) bool. Is the attack targeted or untargeted?
              Untargeted, the default, will try to make the label incorrect.
              Targeted will instead try to move in the direction of being more like y.
    :param sanity_checks: bool, if True, include asserts (Turn them off to use less runtime /
              memory or for unit tests that intentionally pass strange input)
    :return: a tensor for the adversarial example
    """
    if norm not in [np.inf, 1, 2]:
        raise ValueError("Norm order must be either np.inf, 1, or 2.")

    if loss_fn is None:
        loss_fn = tf.losses.MeanSquaredError()

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
        asserts.append(tf.math.greater_equal(deep_features, clip_min))

    if clip_max is not None:
        asserts.append(tf.math.less_equal(deep_features, clip_max))

    # cast to tensor if provided as numpy array
    users = tf.cast(users, tf.float32)
    deep_features = tf.cast(deep_features, tf.float32)

    if y is None:
        # Using model predictions as ground truth to avoid label leaking
        y = tf.argmax(model_fn([users,deep_features]), 1)

    grad = recommender_compute_gradient(model_fn, loss_fn, users,deep_features, y, targeted)

    optimal_perturbation = optimize_linear(grad, eps, norm)
    # Add perturbation to original example to obtain adversarial example
    adv_x = deep_features + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
        # We don't currently support one-sided clipping
        assert clip_min is not None and clip_max is not None
        adv_x = tf.clip_by_value(adv_x, clip_min, clip_max)

    if sanity_checks:
        assert np.all(asserts)
    return adv_x
