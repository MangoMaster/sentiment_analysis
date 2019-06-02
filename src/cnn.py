import os
import functools
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from utilnn import Fscore, coef


def load_data(labels_prefix):
    """
    @param labels_prefix: 'classfication' or 'regression'
    @return: (inputs_train, outputs_train, inputs_test, outputs_test)
    """
    # Generate file path
    inputs_train_file_path = os.path.join(
        os.path.pardir, "data", "word-embedding" + "_train")
    outputs_train_file_path = os.path.join(
        os.path.pardir, "data", labels_prefix + "_train")
    inputs_test_file_path = os.path.join(
        os.path.pardir, "data", "word-embedding" + "_test")
    outputs_test_file_path = os.path.join(
        os.path.pardir, "data", "regression" + "_test")  # test label always use regression
    # Get data
    with open(inputs_train_file_path, 'rb') as inputs_train_file:
        inputs_train = np.load(inputs_train_file)
    with open(outputs_train_file_path, 'rb') as outputs_train_file:
        outputs_train = np.load(outputs_train_file)
    with open(inputs_test_file_path, 'rb') as inputs_test_file:
        inputs_test = np.load(inputs_test_file)
    with open(outputs_test_file_path, 'rb') as outputs_test_file:
        outputs_test = np.load(outputs_test_file)
    # Return data
    return (inputs_train, outputs_train, inputs_test, outputs_test)


def cnn(inputs_train, outputs_train, inputs_test, outputs_test, loss, train_embedding):
    """
    Convolutional neural network.
    @param loss: 'categorical_crossentropy' or 'mean_squared_error'
    @param train_embedding: 0 - initialize with word_embedding_matrix, trainable=False
                            1 - initialize with word_embedding_matrix, trainable=True
                            2 - initialize with random matrix, trainable=True
    """
    word_embedding_matrix_file_path = os.path.join(
        os.path.pardir, "data", "word-embedding_matrix")
    with open(word_embedding_matrix_file_path, 'rb') as word_embedding_matrix_file:
        word_embedding_matrix = np.load(word_embedding_matrix_file)
    # Build CNN model
    if train_embedding == 0:
        embedding_layer = Embedding(word_embedding_matrix.shape[0], word_embedding_matrix.shape[1], weights=[word_embedding_matrix],
                                    input_length=inputs_train.shape[1], trainable=False)
    elif train_embedding == 1:
        embedding_layer = Embedding(word_embedding_matrix.shape[0], word_embedding_matrix.shape[1], weights=[word_embedding_matrix],
                                    input_length=inputs_train.shape[1], trainable=True)
    elif train_embedding == 2:
        embedding_layer = Embedding(word_embedding_matrix.shape[0], word_embedding_matrix.shape[1],
                                    input_length=inputs_train.shape[1], trainable=True)
    else:
        raise ValueError("train_embedding should be 0 or 1 or 2.")
    model = Sequential()
    model.add(embedding_layer)
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.25))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.25))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(outputs_train.shape[1], activation='softmax'))
    print(model.summary())
    # compile
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy', coef])
    # train
    fscore = Fscore()
    model.fit(inputs_train, outputs_train, epochs=15, batch_size=128,
              validation_data=(inputs_test, outputs_test), callbacks=[fscore])
    # evaluate
    score = model.evaluate(inputs_test, outputs_test, batch_size=128)
    print("Eval: loss: %.4f - acc - %.4f - fscore: %.4f - coef: %.4f" %
          (score[0], score[1], fscore.get_data(), score[2]))


cnn_static = functools.partial(cnn, train_embedding=0)
cnn_non_static = functools.partial(cnn, train_embedding=1)
cnn_rand = functools.partial(cnn, train_embedding=2)

if __name__ == "__main__":
    inputs_train, outputs_train, inputs_test, outputs_test = \
        load_data("classification")
    cnn_static(inputs_train, outputs_train, inputs_test, outputs_test,
               loss='categorical_crossentropy')
    cnn_non_static(inputs_train, outputs_train, inputs_test, outputs_test,
                   loss='categorical_crossentropy')
    cnn_rand(inputs_train, outputs_train, inputs_test, outputs_test,
             loss='categorical_crossentropy')

    inputs_train, outputs_train, inputs_test, outputs_test = \
        load_data("regression")
    cnn_static(inputs_train, outputs_train, inputs_test, outputs_test,
               loss='mean_squared_error')
    cnn_non_static(inputs_train, outputs_train, inputs_test, outputs_test,
                   loss='mean_squared_error')
    cnn_rand(inputs_train, outputs_train, inputs_test, outputs_test,
             loss='mean_squared_error')
