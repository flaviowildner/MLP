from __future__ import print_function
import keras
import random
from matplotlib import pyplot as plt
from numpy import argmax

from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import initializers
from keras import backend as K


class MLP(Model):
    def __init__(self, ninput, nhidden, nneurons, outputs, dropout=0.0):
        super(MLP, self).__init__()
        self.model = Sequential()

        for i in range(nhidden):
            self.model.add(Dense(128, activation='relu', input_shape=(ninput,) if i == 0 else []))
            self.model.add(Dropout(dropout))

        self.model.add(Dense(outputs, activation='softmax'))

    def train(self, x, y, loss='categorical_crossentropy', opt='Adagrad', batch_size=128, epochs=5, validation_split=0.1, validation_data=None, verbose=1):
        self.model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
        return self.model.fit(x=x, y=y, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_split=validation_split)

    def rate_classifier(self, x, y):
        return self.model.evaluate(x, y)

    def predict_proba(self, x):
        return self.predict(x)

    def predict_class(self, x):
        return self.predict_class(x)


def aplicar_ruido(dados, n_classes, taxa_ruido):
    if(taxa_ruido < 0 or taxa_ruido > 1.0):
        print("Taxa de ruído inválida. Deve variar de 0.0 a 1.0")
        return None
    dados_retorno = dados.copy()
    opcoes_sorteio = list(range(n_classes))
    for i in range(len(dados_retorno)):
        if random.random() < taxa_ruido:
            tmp = argmax(dados_retorno[i])
            opcoes_sorteio.remove(tmp)
            elemento_sorteado = random.choice(opcoes_sorteio)
            dados_retorno[i] = keras.utils.to_categorical(elemento_sorteado, n_classes)
            opcoes_sorteio.insert(tmp, tmp)
    return dados_retorno
