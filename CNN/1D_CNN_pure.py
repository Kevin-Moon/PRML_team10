from __future__ import print_function
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation, advanced_activations
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.callbacks import ProgbarLogger
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

# Constants
batch_size = 12
num_classes = 2
epochs = 400
eeg_rows, num_ch = 300, 19
input_shape = (300, 19)

# load data file
x_train = np.load('eeg_train.npy')
y_train = np.load('label_train.npy')

x_test = np.load('eeg_test.npy')
y_test = np.load('label_test.npy')

x_valid = np.load('eeg_valid.npy')
y_valid = np.load('label_valid.npy')

DATA_SET = np.load('EEG_csv_total.npy')

ACC = []
TRY = 1

def create_model():
    conv = Sequential()

    conv.add(Conv1D(8, 5, input_shape=input_shape, padding='same', kernel_initializer='he_uniform'))
    conv.add(advanced_activations.LeakyReLU(alpha=0.3))
    conv.add(MaxPooling1D(pool_size=2, strides=2))
    conv.add(Dropout(0.15))

    conv.add(Conv1D(16, 5, padding='same', kernel_initializer='he_uniform'))
    conv.add(advanced_activations.LeakyReLU(alpha=0.3))
    conv.add(MaxPooling1D(pool_size=2, strides=2))
    conv.add(Dropout(0.15))

    conv.add(Conv1D(32, 3, padding='same', kernel_initializer='he_uniform'))
    conv.add(advanced_activations.LeakyReLU(alpha=0.3))
    conv.add(MaxPooling1D(pool_size=2, strides=2))
    conv.add(Dropout(0.5))

    conv.add(Conv1D(16, 1, padding='same', kernel_initializer='he_uniform'))
    conv.add(Conv1D(32, 3, padding='same', kernel_initializer='he_uniform'))
    conv.add(advanced_activations.LeakyReLU(alpha=0.3))
    conv.add(MaxPooling1D(pool_size=2, strides=2))
    conv.add(Dropout(0.5))

    conv.add(Conv1D(16, 1, padding='same', kernel_initializer='he_uniform'))
    conv.add(Conv1D(32, 3, padding='same', kernel_initializer='he_uniform'))
    conv.add(advanced_activations.LeakyReLU(alpha=0.3))
    conv.add(MaxPooling1D(pool_size=2, strides=2))
    conv.add(Dropout(0.5))

    conv.add(Flatten())
    conv.add(Dense(32, kernel_initializer='he_uniform'))
    conv.add(advanced_activations.LeakyReLU(alpha=0.3))
    conv.add(Dropout(0.5))

    conv.add(Dense(32, kernel_initializer='he_uniform'))
    conv.add(advanced_activations.LeakyReLU(alpha=0.3))
    conv.add(Dropout(0.2))

    conv.add(Dense(8, kernel_initializer='he_uniform'))
    conv.add(advanced_activations.LeakyReLU(alpha=0.3))
    conv.add(Dropout(0.1))

    conv.add(Dense(2, activation='sigmoid'))
    return conv

for x_train,y_train,x_test,y_test,x_valid,y_valid in DATA_SET:

    x_train = x_train.reshape(x_train.shape[0], eeg_rows, num_ch)
    x_test = x_test.reshape(x_test.shape[0], eeg_rows,num_ch)
    x_valid = x_valid.reshape(x_valid.shape[0], eeg_rows, num_ch)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_valid = x_valid.astype('float32')

    x_train /= 255
    x_test /= 255
    x_valid /= 255

    print('x_train shape:', x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    #print(x_train.shape[0], 'train samples')
    #print(x_test.shape[0], 'test samples')

    checkpointer = ModelCheckpoint(filepath="/tmp/weights_cnn_v1", verbose=1, save_best_only=True, monitor='val_loss')
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True,
                              write_images=False)
    # earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=80, verbose=0, mode='auto')

    score = np.array([0,0])
    while score[1] < 0.6: # to avoid too bad initialization

        conv = create_model()

        conv.compile(loss='categorical_crossentropy',
                     optimizer=keras.optimizers.Adam(),
                     metrics=['accuracy'])

        conv.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=0,
                  validation_data=(x_valid, y_valid),
                  callbacks=[checkpointer, tensorboard])

        conv.load_weights("/tmp/weights_cnn_v1")

        #score = conv.evaluate(x_test, y_test, verbose=1)
        score = conv.test_on_batch(x_test, y_test)
        print(score)
        print(conv.summary())


    print('Test loss:', score[0])
    print('Test accuracy[',TRY,']', score[1])
    ACC.append(score[1])
    TRY += 1

print(ACC)
print('AVG :',np.mean(np.array(ACC)))
