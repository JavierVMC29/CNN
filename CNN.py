from datetime import datetime
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import my_dataset

datos, metadatos = tfds.load('my_dataset', as_supervised=True, with_info=True)

plt.figure(figsize=(20, 20))

datos_entrenamiento = []
datos_test = []

TAMANO_IMG = 50

for i, (imagen, etiqueta) in enumerate(datos['train']):
    imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG))
    # imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # Cambiar tamano a 100,100,1
    #imagen = imagen.reshape(TAMANO_IMG, TAMANO_IMG)
    datos_entrenamiento.append([imagen, etiqueta])

for i, (imagen, etiqueta) in enumerate(datos['test']):
    imagen = cv2.resize(imagen.numpy(), (TAMANO_IMG, TAMANO_IMG))
    # imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # Cambiar tamano a 100,100,1
    #imagen = imagen.reshape(TAMANO_IMG, TAMANO_IMG)
    datos_test.append([imagen, etiqueta])

train_images = []  # imagenes de entrada (pixeles)
train_labels = []  # etiquetas (descanso o movimiento)

test_images = []
test_labels = []

for imagen, etiqueta in datos_entrenamiento:
    train_images.append(imagen)
    train_labels.append(etiqueta)

for imagen, etiqueta in datos_test:
    test_images.append(imagen)
    test_labels.append(etiqueta)

# Normalizar los datos de las imagenes. Se pasan a numero flotante y dividen entre 255 para quedar de 0-1 en lugar de 0-255
train_images = np.array(train_images).astype(float) / 255
test_images = np.array(test_images).astype(float) / 255

# Convertir etiquetas en arreglo simple
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# CNN
# Topologia
modeloCNN = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu',
                           input_shape=(TAMANO_IMG, TAMANO_IMG, 3)),
    tf.keras.layers.MaxPooling2D(3, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(3, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(3, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')

])

# Compilacion
modeloCNN.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Entrenamiento
logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

checkpointer = tf.keras.callbacks.ModelCheckpoint(
    filepath='mnist.model.best.hdf5', verbose=1, save_best_only=True)

modeloCNN.fit(train_images,
              train_labels,
              batch_size=10,
              epochs=30,
              validation_split=0.2,
              callbacks=[checkpointer, tensorboard_callback])

modeloCNN.load_weights('mnist.model.best.hdf5')

scores = modeloCNN.evaluate(test_images, test_labels)

accuracy = scores[1]*100
loss = scores[0]*100

print("Accuracy: {:.2f}%".format(accuracy))
print("Loss: {:.2f}%".format(loss))
# predictions = modeloCNN.predict([])
