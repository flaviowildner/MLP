import MLPlibrary
import keras
import os
import json

from MLPlibrary import MLP
from MLPlibrary import aplicar_ruido
from keras.datasets import mnist
from keras.layers import Dense, Dropout
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


ruidos = [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.82, 0.84, 0.86, 0.88, 0.90, 1.0]
dropout = [0.0, 0.1, 0.2, 0.3, 0.4, 0.7]
score = []
models = []
history = []
num_classes = 10
optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=0.000001, decay=0.0)


try:
    os.mkdir('files')
except OSError:
    pass
os.chdir('files')


data = {}
data['ruidos'] = ruidos
data['dropout'] = dropout

listFiles = os.listdir('./')
if(len(listFiles) != 0):
    newFolder = str(int(listFiles[len(listFiles) - 1]) + 1)
else:
    newFolder = '1'
os.mkdir(newFolder)
os.chdir(newFolder)

report_json = json.dumps(data)
with open('report_configuration.json', 'w') as file:
    file.write(report_json)
file.close()

for d, taxa_dropout in enumerate(dropout):
    score.append([])
    for i, taxa_ruido in enumerate(ruidos):
        #Aplica ruído no dataset mnist
        train_ruido_y = aplicar_ruido(train_y, num_classes, taxa_ruido)
        if(train_ruido_y is None):
            break

        #Modelo MLP
        mlp = MLP(ninput=784, nhidden=2, nneurons=128, outputs=num_classes, dropout=taxa_dropout)

        print('Iniciado treinamento com', taxa_ruido, 'de ruido e', taxa_dropout, 'de dropout')
        tmp_history = mlp.train(x=train_x, y=train_ruido_y, epochs=20, batch_size=128, validation_split=0.1, validation_data=(test_x, test_y))
        print('Terminado treinamento')

        mlp.model.save('model_noise_' + str(taxa_ruido) + '_dropout_' + str(d) + '.h5')
        mlp.model.save('weights_noise' + str(taxa_ruido) + '_dropout_' + str(d) + '.h5')
        with open('history_' + str(taxa_ruido) + '_dropout_' + str(d) + '.json', 'w') as file:
            file.write(json.dumps(tmp_history.history))
        score[d].append(mlp.model.evaluate(test_x, test_y, verbose=0)[1])
        tofile_json = json.dumps(score)
        with open('score.json', 'w') as file:
            file.write(tofile_json)
        file.close()

        history.append(tmp_history)
        models.append(mlp)
