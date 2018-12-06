import keras

def cnn_model():
  # Create a sequential model:
  model = keras.models.Sequential()
  # 1st convolutional layer:
  model.add(keras.layers.Conv2D(
      input_shape=(48,48,3), # Notice input_shape!
      filters=24,
      kernel_size=(3,3),
      activation='relu'))
  # Convolutional layer:
  model.add(keras.layers.Conv2D(
      filters=24,
      kernel_size=(3,3),
      activation='relu'))
  # Max pooling
  model.add(keras.layers.MaxPooling2D(
      pool_size=(2,2)))
  # Convolutional layers:
  model.add(keras.layers.Conv2D(
      filters=48,
      kernel_size=(3,3),
      activation='relu'))
  model.add(keras.layers.Conv2D(
      filters=48,
      kernel_size=(3,3),
      activation='relu'))
  # Max pooling
  model.add(keras.layers.MaxPooling2D(
      pool_size=(2,2)))
  # Flatten the last image
  model.add(keras.layers.Flatten())
  # 3 dense layers:
  for i in range(3):
    model.add(keras.layers.Dense(
        units=250,
        activation='relu'))
  # Last layer
  model.add(keras.layers.Dense(
      units=5,
      activation='softmax'))
  # return model
  return model

model = cnn_model()
model.summary()
keras.utils.plot_model(model,"./cnn_model.png",
  show_shapes=True, show_layer_names=False)
