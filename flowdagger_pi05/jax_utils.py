"""JAX helpers for the steering policy: type aliases, jitted action
eval/sample, image augmentations, and add_batch_dim.
"""

from functools import partial
import functools
from typing import Any, Callable, Tuple

import distrax
import flax
import jax
import jax.numpy as jnp
import numpy as np

from buffer import DatasetDict

PRNGKey = Any
Params = flax.core.FrozenDict[str, Any]


# --- action eval / sample -------------------------------------------------

@partial(jax.jit, static_argnames='actor_apply_fn')
def eval_actions_jit(actor_apply_fn: Callable[..., distrax.Distribution],
                     actor_params: Params,
                     observations: np.ndarray,
                     actor_batch_stats: Any) -> jnp.ndarray:
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    dist = actor_apply_fn(input_collections, observations, training=False,
                          mutable=False)
    return dist.mode()


@partial(jax.jit, static_argnames='actor_apply_fn')
def sample_actions_jit(
        rng: PRNGKey, actor_apply_fn: Callable[..., distrax.Distribution],
        actor_params: Params,
        observations: np.ndarray,
        actor_batch_stats: Any) -> Tuple[PRNGKey, jnp.ndarray]:
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    dist = actor_apply_fn(input_collections, observations)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)


# --- image augmentations --------------------------------------------------

def random_crop(key, img, padding):
    crop_from = jax.random.randint(key, (2, ), 0, 2 * padding + 1)
    crop_from = jnp.concatenate([crop_from, jnp.zeros((2, ), dtype=jnp.int32)])
    padded_img = jnp.pad(img, ((padding, padding), (padding, padding), (0, 0),
                               (0, 0)),
                         mode='edge')
    return jax.lax.dynamic_slice(padded_img, crop_from, img.shape)


def batched_random_crop(key, imgs, padding=4):
    keys = jax.random.split(key, imgs.shape[0])
    return jax.vmap(random_crop, (0, 0, None))(keys, imgs, padding)


def _maybe_apply(apply_fn, inputs, rng, apply_prob):
    should_apply = jax.random.uniform(rng, shape=()) <= apply_prob
    return jax.lax.cond(should_apply, inputs, apply_fn, inputs, lambda x: x)


def rgb_to_hsv(r, g, b):
    """Converts R, G, B values to H, S, V values."""
    vv = jnp.maximum(jnp.maximum(r, g), b)
    range_ = vv - jnp.minimum(jnp.minimum(r, g), b)
    sat = jnp.where(vv > 0, range_ / vv, 0.)
    norm = jnp.where(range_ != 0, 1. / (6. * range_), 1e9)

    hr = norm * (g - b)
    hg = norm * (b - r) + 2. / 6.
    hb = norm * (r - g) + 4. / 6.

    hue = jnp.where(r == vv, hr, jnp.where(g == vv, hg, hb))
    hue = hue * (range_ > 0)
    hue = hue + (hue < 0)

    return hue, sat, vv


def hsv_to_rgb(h, s, v):
    """Converts H, S, V values to an R, G, B tuple."""
    c = s * v
    m = v - c
    dh = (h % 1.) * 6.
    fmodu = dh % 2.
    x = c * (1 - jnp.abs(fmodu - 1))
    hcat = jnp.floor(dh).astype(jnp.int32)
    rr = jnp.where(
        (hcat == 0) | (hcat == 5), c, jnp.where(
            (hcat == 1) | (hcat == 4), x, 0)) + m
    gg = jnp.where(
        (hcat == 1) | (hcat == 2), c, jnp.where(
            (hcat == 0) | (hcat == 3), x, 0)) + m
    bb = jnp.where(
        (hcat == 3) | (hcat == 4), c, jnp.where(
            (hcat == 2) | (hcat == 5), x, 0)) + m
    return rr, gg, bb


def adjust_brightness(rgb_tuple, delta):
    return jax.tree_util.tree_map(lambda x: x + delta, rgb_tuple)


def adjust_contrast(image, factor):
    def _adjust_contrast_channel(channel):
        mean = jnp.mean(channel, axis=(-2, -1), keepdims=True)
        return factor * (channel - mean) + mean
    return jax.tree_util.tree_map(_adjust_contrast_channel, image)


def adjust_saturation(h, s, v, factor):
    return h, jnp.clip(s * factor, 0., 1.), v


def adjust_hue(h, s, v, delta):
    return (h + delta) % 1.0, s, v


def _random_brightness(rgb_tuple, rng, max_delta):
    delta = jax.random.uniform(rng, shape=(), minval=-max_delta, maxval=max_delta)
    return adjust_brightness(rgb_tuple, delta)


def _random_contrast(rgb_tuple, rng, max_delta):
    factor = jax.random.uniform(
        rng, shape=(), minval=1 - max_delta, maxval=1 + max_delta)
    return adjust_contrast(rgb_tuple, factor)


def _random_saturation(rgb_tuple, rng, max_delta):
    h, s, v = rgb_to_hsv(*rgb_tuple)
    factor = jax.random.uniform(
        rng, shape=(), minval=1 - max_delta, maxval=1 + max_delta)
    return hsv_to_rgb(*adjust_saturation(h, s, v, factor))


def _random_hue(rgb_tuple, rng, max_delta):
    h, s, v = rgb_to_hsv(*rgb_tuple)
    delta = jax.random.uniform(rng, shape=(), minval=-max_delta, maxval=max_delta)
    return hsv_to_rgb(*adjust_hue(h, s, v, delta))


def _to_grayscale(image):
    rgb_weights = jnp.array([0.2989, 0.5870, 0.1140])
    grayscale = jnp.tensordot(image, rgb_weights, axes=(-1, -1))[..., jnp.newaxis]
    return jnp.tile(grayscale, (1, 1, 3))  # Back to 3 channels.


def _color_transform_single_image(image, rng, brightness, contrast, saturation,
                                  hue, to_grayscale_prob, color_jitter_prob,
                                  apply_prob, shuffle):
    """Applies color jittering to a single image."""
    apply_rng, transform_rng = jax.random.split(rng)
    perm_rng, b_rng, c_rng, s_rng, h_rng, cj_rng, gs_rng = jax.random.split(
        transform_rng, 7)

    # Whether the transform should be applied at all.
    should_apply = jax.random.uniform(apply_rng, shape=()) <= apply_prob
    # Whether to apply grayscale transform.
    should_apply_gs = jax.random.uniform(gs_rng, shape=()) <= to_grayscale_prob
    # Whether to apply color jittering.
    should_apply_color = jax.random.uniform(cj_rng, shape=()) <= color_jitter_prob

    # Decorator to conditionally apply fn based on an index.
    def _make_cond(fn, idx):

        def identity_fn(x, unused_rng, unused_param):
            return x

        def cond_fn(args, i):
            def clip(args):
                return jax.tree_util.tree_map(lambda arg: jnp.clip(arg, 0., 1.), args)
            out = jax.lax.cond(should_apply & should_apply_color & (i == idx), args,
                         lambda a: clip(fn(*a)), args,
                         lambda a: identity_fn(*a))
            return jax.lax.stop_gradient(out)

        return cond_fn

    random_brightness_cond = _make_cond(_random_brightness, idx=0)
    random_contrast_cond = _make_cond(_random_contrast, idx=1)
    random_saturation_cond = _make_cond(_random_saturation, idx=2)
    random_hue_cond = _make_cond(_random_hue, idx=3)

    def _color_jitter(x):
        rgb_tuple = tuple(jax.tree_util.tree_map(jnp.squeeze, jnp.split(x, 3, axis=-1)))
        if shuffle:
            order = jax.random.permutation(perm_rng, jnp.arange(4, dtype=jnp.int32))
        else:
            order = range(4)
        for idx in order:
            if brightness > 0:
                rgb_tuple = random_brightness_cond((rgb_tuple, b_rng, brightness), idx)
            if contrast > 0:
                rgb_tuple = random_contrast_cond((rgb_tuple, c_rng, contrast), idx)
            if saturation > 0:
                rgb_tuple = random_saturation_cond((rgb_tuple, s_rng, saturation), idx)
            if hue > 0:
                rgb_tuple = random_hue_cond((rgb_tuple, h_rng, hue), idx)
        return jnp.stack(rgb_tuple, axis=-1)

    out_apply = _color_jitter(image)
    out_apply = jax.lax.cond(should_apply & should_apply_gs, out_apply,
                           _to_grayscale, out_apply, lambda x: x)
    return jnp.clip(out_apply, 0., 1.)


def color_transform(rng,
                    images,
                    brightness=0.2,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.03,
                    color_jitter_prob=0.8,
                    to_grayscale_prob=0.0,
                    apply_prob=1.0,
                    shuffle=True):
    """Applies color jittering and/or grayscaling to a batch of images.

    Args:
        images: an NHWC tensor, with C=3.
        rng: a single PRNGKey.
        brightness: the range of jitter on brightness.
        contrast: the range of jitter on contrast.
        saturation: the range of jitter on saturation.
        hue: the range of jitter on hue.
        color_jitter_prob: the probability of applying color jittering.
        to_grayscale_prob: the probability of converting the image to grayscale.
        apply_prob: the probability of applying the transform to a batch element.
        shuffle: whether to apply the transforms in a random order.
    Returns:
        A NHWC tensor of the transformed images.
    """
    images = images[:, :, :, :, 0]
    rngs = jax.random.split(rng, images.shape[0])
    jitter_fn = functools.partial(
        _color_transform_single_image,
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
        color_jitter_prob=color_jitter_prob,
        to_grayscale_prob=to_grayscale_prob,
        apply_prob=apply_prob,
        shuffle=shuffle)
    augmented_images = jax.vmap(jitter_fn)(images, rngs)
    return augmented_images[..., jnp.newaxis]


# --- misc -----------------------------------------------------------------

def add_batch_dim(input):
    if isinstance(input, dict):
        for k, v in input.items():
            input[k] = v[None]
    else:
        input = input[None]
    return input
