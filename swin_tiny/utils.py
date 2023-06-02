def window_partition(x: tf.Tensor, window_size: int):
    """
        this function is used to create a local window, with the windo_size.
        Params:
            x: (B, H, W, C)
            window_size (int): window size
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
    """
    # batch height width and channles
    B, H, W, C = tf.shape(x).numpy()
    
    x = tf.reshape(
        x, (B, H // window_size, window_size, W // window_size, window_size, C)
    )
    x = tf.transpose(x, perm=[0, 1, 3, 2, 4, 5])
    windows = tf.reshape(x, shape=[-1, window_size, window_size, C])
    return windows


def window_reverse(windows: tf.Tensor, window_size: int, H: int, W: int):
    """
    Params:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = tf.shape(windows)[0] // tf.cast(H * W / window_size / window_size, dtype="int32")
    x = tf.reshape(windows, (B, H // window_size, W // window_size,
                                             window_size, window_size, -1))
    x = tf.transpose(x, [0, 1, 3, 2, 4, 5])
    return tf.reshape(x, (B, H, W, -1))
