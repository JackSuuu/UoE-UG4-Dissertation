"""Official LIBERO policy-image preprocessing, shared across actor backends."""


def policy_image(raw):
    import numpy as np
    import tensorflow as tf
    from PIL import Image
    # Image preprocessing must not reserve GPU memory needed by frozen OpenVLA.
    tf.config.set_visible_devices([], 'GPU')
    # Rotate, JPEG round trip, antialiased Lanczos resize and center crop.
    image = np.ascontiguousarray(raw[::-1, ::-1])
    image = tf.io.decode_image(tf.image.encode_jpeg(image), expand_animations=False, dtype=tf.uint8)
    image = tf.image.resize(image, (224, 224), method='lanczos3', antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    image = tf.image.convert_image_dtype(image, tf.float32)
    side = tf.sqrt(tf.constant(0.9, tf.float32))
    lo = (1 - side) / 2
    boxes = tf.reshape(tf.stack([lo, lo, lo + side, lo + side]), (1, 4))
    image = tf.image.crop_and_resize(image[None], boxes, [0], (224, 224))[0]
    image = tf.image.convert_image_dtype(tf.clip_by_value(image, 0, 1), tf.uint8, saturate=True)
    return Image.fromarray(image.numpy()).convert('RGB')
