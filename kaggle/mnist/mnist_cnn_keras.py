"""
"""
import argparse
import os

import keras

from keras.datasets import mnist
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Input, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

# get settings
parser = argparse.ArgumentParser(description='Keras MNIST example')

parser.add_argument('--gpu', type=str, default='0', metavar='N',
                    help='gpu id for the training to run (default: 0)')

parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate (default: 1e-3)')

parser.add_argument('--dropout', type=float, default=0., metavar='N',
                    help='dropout rate (default: 0)')

parser.add_argument('--epoch', type=int, default=1, metavar='N',
                    help='number of epochs')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


# import data
batch_size = 128
num_classes = 10
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)


# build model
i = Input(shape=input_shape)
x = BatchNormalization()(i)

x = Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
x = BatchNormalization()(x)

x = Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
x = BatchNormalization()(x)

x = Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(args.dropout)(x)

x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(args.dropout)(x)

x = Dense(10, activation='softmax')(x)

model = Model(i, x)

# train the model
model.compile(optimizer=keras.optimizers.Adam(), 
              loss=keras.losses.categorical_crossentropy, 
              metrics=['accuracy'])

model.fit(X_train, y_train, 
          epochs=args.epoch, 
          validation_data=(X_val, y_val), 
          shuffle=True,
          verbose=1,
          batch_size=batch_size)


# test the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print('test loss = {}'.format(test_loss))
print('test accuracy = {}'.format(test_accuracy))


# python mnist_cnn_keras.py --gpu 0 --lr 1e-4 --dropout 0.2 --epoch 10