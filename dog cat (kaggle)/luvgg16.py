import numpy as np

from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Lambda, Dropout
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator


TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels.h5'


def vgg_preprocess(x):
    vgg_mean = np.array([123.68, 116.779, 103.939],
                        dtype=np.float32).reshape((3, 1, 1))

    x = x - vgg_mean
    return x[:, ::-1]  # reverse axis rgb->bgr


class VGG16(object):

    def __init__(self, pretrained_weight_path=None):
        self._get_original_vgg16()

        if pretrained_weight_path:
            self.fine_tune()
            self.model.load_weights(pretrained_weight_path)
            self.compile()
            print('model loaded from path={}'.format(pretrained_weight_path))

    def _conv_block(self, nb_filter, nb_conv):
        for _ in range(nb_conv):
            self.model.add(Convolution2D(nb_filter, 3, 3,
                                         activation='relu', border_mode='same'))
        self.model.add(MaxPooling2D((2, 2)))

    def _fc_block(self):
        self.model.add(Dense(4096, activation='relu'))

    def _get_original_vgg16(self):
        self.model = Sequential()

        self.model.add(Lambda(vgg_preprocess, input_shape=(
            3, 224, 224), output_shape=(3, 224, 224)))

        self._conv_block(64, 2)
        self._conv_block(128, 2)
        self._conv_block(256, 3)
        self._conv_block(512, 3)
        self._conv_block(512, 3)

        self.model.add(Flatten())

        self._fc_block()
        self._fc_block()

        self.model.add(Dense(1000, activation='relu'))

        weights = get_file('vgg16_weights',
                           TH_WEIGHTS_PATH,
                           cache_subdir='models')

        self.model.load_weights(weights)

    def fine_tune(self):
        self.model.pop()  # take the final dense(1000) out and attach your output layer

        for layer in self.model.layers:
            layer.trainable = False

        self.model.add(Dense(2, activation='softmax'))

    def compile(self, lr=0.01):
        self.model.compile(optimizer=Adam(lr=lr),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def get_batch_generator(self, path, batch_size=32, shuffle=True, class_mode='categorical', augmentation=False):
        if augmentation:
            gen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True)

        else:
            gen = ImageDataGenerator()

        return gen.flow_from_directory(
            path,
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode=class_mode,
            shuffle=shuffle)

    def fit(self, train, valid, nb_epoch=1):
        self.model.fit_generator(
            train,
            samples_per_epoch=train.nb_sample,
            nb_epoch=nb_epoch,
            validation_data=valid,
            nb_val_samples=valid.nb_sample,
            verbose=2)

    def save_weight(self, path):
        self.model.save_weights(path)

    def predict_prob_generator(self, path):
        """
        return the probability of dog
            array([[  1.0000e+00,   9.4158e-18],
                   [  1.0000e+00,   9.7884e-16])
        """
        gen = self.get_batch_generator(path, shuffle=False)
        prob = self.model.predict_generator(
            gen,
            val_samples=gen.nb_sample)

        return prob[:, 1]

    def predict_label_generator(self, prob):
        return np.around(prob)
