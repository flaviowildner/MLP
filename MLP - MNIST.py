import mnist_library
import keras
import os
import json

from mnist_library import MLP
from mnist_library import aplicar_ruido
from keras.datasets import mnist
from keras.layers import Dense, Dropout
from keras import backend as K
from keras.models import model_from_json
from keras.models import load_model

from matplotlib import pyplot as plt
from numpy import argmax
import numpy as np
np.set_printoptions(suppress=True)


#MNIST DATASET
(train_x, train_y), (test_x, test_y) = mnist.load_data()
train_x = train_x.astype('float32') / 255
test_x = test_x.astype('float32') / 255
train_x = train_x.reshape(60000, 784)
test_x = test_x.reshape(10000, 784)
train_y = keras.utils.to_categorical(train_y, 10)
test_y = keras.utils.to_categorical(test_y, 10)


ruidos = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9]
models = []
history = []
num_classes = 10
optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=0.000001, decay=0.0)

i = 0
while(1):
    try:
        os.remove('model_' + str(i) + '.h5')
        os.remove('weights_' + str(i) + '.h5')
        os.remove('history_' + str(i) + '.json')
    except OSError:
        break
    i = i + 1

for i, taxa_ruido in enumerate(ruidos):
    #Aplica ruído no dataset mnist
    train_ruido_y = aplicar_ruido(train_y, num_classes, taxa_ruido)
    if(train_ruido_y is None):
        break

    #Modelo MLP
    model = MLP(ninput=784, nhidden=2, nneurons=128, outputs=num_classes, dropout=0.0)

    print('Iniciado treinamento com', taxa_ruido, 'de ruido')
    tmp_history = model.train(x=train_x, y=train_ruido_y, epochs=30, batch_size=64, validation_split=0.1)
    print('Terminado treinamento')

    model.model.save('model_' + str(i) + '.h5')
    model.model.save('weights_' + str(i) + '.h5')
    with open('history_' + str(i) + '.json', 'w') as file:
        file.write(json.dumps(tmp_history.history))

    history.append(tmp_history)
    models.append(model)
