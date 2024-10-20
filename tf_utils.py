import tensorflow as tf


def dense_layer(inputs, output_units, bias=True, activation=None, batch_norm=None,
                dropout=None, scope='dense-layer'):
    """
    Applies a dense layer to a 2D tensor of shape [batch_size, input_units]
    to produce a tensor of shape [batch_size, output_units].
    
    Args:
        inputs: Tensor of shape [batch size, input_units].
        output_units: Number of output units.
        activation: activation function.
        dropout: dropout keep prob.
        
    Returns:
        Tensor of shape [batch size, output_units].
    """
    # Variable initializer
    initializer = tf.keras.initializers.VarianceScaling()

    W = tf.Variable(
        initializer(shape=[shape(inputs, -1), output_units]),
        name='weights'
    )
    
    z = tf.matmul(inputs, W)

    if bias:
        b = tf.Variable(
            tf.zeros([output_units]),
            name='biases'
        )
        z = z + b

    if batch_norm is not None:
        # Using keras batch normalization
        bn_layer = tf.keras.layers.BatchNormalization()
        z = bn_layer(z, training=batch_norm)

    z = activation(z) if activation else z
    z = tf.nn.dropout(z, rate=1 - dropout) if dropout is not None else z
    return z


def time_distributed_dense_layer(
        inputs, output_units, bias=True, activation=None, batch_norm=None,
        dropout=None, scope='time-distributed-dense-layer'):
    """
    Applies a shared dense layer to each timestep of a tensor of shape
    [batch_size, max_seq_len, input_units] to produce a tensor of shape
    [batch_size, max_seq_len, output_units].

    Args:
        inputs: Tensor of shape [batch size, max sequence length, ...].
        output_units: Number of output units.
        activation: activation function.
        dropout: dropout keep prob.

    Returns:
        Tensor of shape [batch size, max sequence length, output_units].
    """
    # Variable initializer
    initializer = tf.keras.initializers.VarianceScaling()

    W = tf.Variable(
        initializer(shape=[shape(inputs, -1), output_units]),
        name='weights'
    )

    z = tf.einsum('ijk,kl->ijl', inputs, W)

    if bias:
        b = tf.Variable(
            tf.zeros([output_units]),
            name='biases'
        )
        z = z + b

    if batch_norm is not None:
        # Using keras batch normalization
        bn_layer = tf.keras.layers.BatchNormalization()
        z = bn_layer(z, training=batch_norm)

    z = activation(z) if activation else z
    z = tf.nn.dropout(z, rate=1 - dropout) if dropout is not None else z
    return z


def shape(tensor, dim=None):
    """Get tensor shape/dimension as list/int"""
    if dim is None:
        return tensor.shape.as_list()
    else:
        return tensor.shape.as_list()[dim]


def rank(tensor):
    """Get tensor rank as python list"""
    return len(tensor.shape.as_list())
