import keras
from keras import layers, models, regularizers

def conv_block(x, filters, kernel_size, activation, padding, reg, batch_norm, spatial_dropout=None):
    x = layers.Conv2D(filters, kernel_size, padding=padding, activation=activation,
                      kernel_regularizer=reg)(x)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    if spatial_dropout:
        x = layers.SpatialDropout2D(spatial_dropout)(x)
    return x

def create_cnn_classifier_network(image_size,
                                  nchannels,
                                  conv_layers,
                                  dense_layers,
                                  p_dropout=None,
                                  p_spatial_dropout=None,
                                  lambda_l2=None,
                                  lrate=0.001,
                                  n_classes=7,
                                  loss=keras.losses.SparseCategoricalCrossentropy(),
                                  metrics=[keras.metrics.SparseCategoricalAccuracy()],
                                  padding='same',
                                  conv_activation='relu',
                                  dense_activation='relu',
                                  use_unet=True):

    inputs = keras.Input(shape=(image_size[0], image_size[1], nchannels))
    x = inputs

    reg = regularizers.l2(lambda_l2) if lambda_l2 else None
    skip_connections = []

    # Encoder
    for layer in conv_layers:
        x = conv_block(x,
                       filters=layer['filters'],
                       kernel_size=layer['kernel_size'],
                       activation=conv_activation,
                       padding=padding,
                       reg=reg,
                       batch_norm=layer.get('batch_normalization', False),
                       spatial_dropout=p_spatial_dropout)
        if use_unet:
            skip_connections.append(x)
        if layer.get('pool_size'):
            x = layers.MaxPooling2D(pool_size=layer['pool_size'])(x)

    # Bottleneck
    for layer in dense_layers:
        x = conv_block(x,
                       filters=layer['units'],
                       kernel_size=(3, 3),
                       activation=dense_activation,
                       padding=padding,
                       reg=reg,
                       batch_norm=layer.get('batch_normalization', False),
                       spatial_dropout=p_spatial_dropout)

    if p_dropout:
        x = layers.Dropout(p_dropout)(x)

    # Decoder (U-Net only)
    if use_unet:
        for i, layer in reversed(list(enumerate(conv_layers))):
            if layer.get('pool_size'):
                x = layers.UpSampling2D(size=layer['pool_size'])(x)
            if i < len(skip_connections):
                x = layers.Concatenate()([x, skip_connections[i]])
            x = conv_block(x,
                           filters=layer['filters'],
                           kernel_size=layer['kernel_size'],
                           activation=conv_activation,
                           padding=padding,
                           reg=reg,
                           batch_norm=layer.get('batch_normalization', False),
                           spatial_dropout=p_spatial_dropout)

    # Final class prediction
    outputs = layers.Conv2D(n_classes, (1, 1), activation='softmax', padding='same')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lrate),
                  loss=loss,
                  metrics=metrics)

    return model
