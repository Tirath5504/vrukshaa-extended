import tensorflow as tf

# Load the Keras model in .keras format
keras_model = tf.keras.models.load_model('corn_model.keras')

# Save the model in .h5 format
keras_model.save('corn_model.h5')

# Load the Keras model in .keras format
keras_model = tf.keras.models.load_model('rice_model.keras')

# Save the model in .h5 format
keras_model.save('rice_model.h5')

# Load the Keras model in .keras format
keras_model = tf.keras.models.load_model('wheat_model.keras')

# Save the model in .h5 format
keras_model.save('wheat_model.h5')

# Load the Keras model in .keras format
keras_model = tf.keras.models.load_model('crop_model.keras')

# Save the model in .h5 format
keras_model.save('crop_model.h5')

# Load the Keras model in .keras format
keras_model = tf.keras.models.load_model('potato_model.keras')

# Save the model in .h5 format
keras_model.save('potato_model.h5')