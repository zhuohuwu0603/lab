"""
"""
import argparse
import os

from keras.datasets import mnist

from keras.utils import to_categorical
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam

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

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


# import data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

# build model
i = Input(shape=(28, 28, 1))
x = BatchNormalization()(i)

x = Convolution2D(32, 3, 3, activation='relu',
                  init='he_normal', border_mode='same')(x)
x = BatchNormalization()(x)

x = Convolution2D(32, 3, 3, activation='relu',
                  init='he_normal', border_mode='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D()(x)

x = Convolution2D(32, 3, 3, activation='relu',
                  init='he_normal', border_mode='same')(x)
x = BatchNormalization()(x)

x = Convolution2D(32, 3, 3, activation='relu',
                  init='he_normal', border_mode='same')(x)
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
model.compile(optimizer=Adam(lr=args.lr),
              loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train,
          nb_epoch=args.epoch,
          validation_data=(X_val, y_val),
          shuffle=True,
          verbose=1,
          batch_size=128)


# test the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print('test loss = {}'.format(test_loss))
print('test accuracy = {}'.format(test_accuracy))


# save model
model.save('model_acc({:0.5f}).h5'.format(test_accuracy))

# python keras_mnist.py --gpu 0 --lr 1e-4 --dropout 0.2 --epoch 10
# python keras_mnist.py --gpu 1 --lr 1e-4 --dropout 0.3 --epoch 10
# python keras_mnist.py --gpu 2 --lr 1e-4 --dropout 0.4 --epoch 10
# python keras_mnist.py --gpu 2 --lr 1e-4 --dropout 0.5 --epoch 10
