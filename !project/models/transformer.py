from tensorflow.keras import layers, models

def build_libs_transformer(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    # Positional Encoding
    x = layers.Dense(64)(inputs)
    x = layers.LayerNormalization()(x)
    
    # Transformer Block
    x = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Dropout(0.1)(x)
    x = layers.LayerNormalization()(x)
    
    # Feed Forward
    x = layers.Dense(128, activation='gelu')(x)
    x = layers.Dense(64)(x)
    
    # Classification Head
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)