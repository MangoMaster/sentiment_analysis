import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from utilnn import Fscore, coef


def load_data(texts_prefix, labels_prefix):
    """
    @param texts_prefix: 'bags-of-words' or 'tf-idf'
    @param labels_prefix: 'classfication' or 'regression'
    @return: (inputs_train, outputs_train, inputs_test, outputs_test)
    """
    # Generate file path
    inputs_train_file_path = os.path.join(
        os.path.pardir, "data", texts_prefix + "_train")
    outputs_train_file_path = os.path.join(
        os.path.pardir, "data", labels_prefix + "_train")
    inputs_test_file_path = os.path.join(
        os.path.pardir, "data", texts_prefix + "_test")
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
    return (inputs_train, outputs_train, inputs_test, outputs_test)


def dnn(inputs_train, outputs_train, inputs_test, outputs_test, loss):
    """
    Fully connected neural network.
    @param loss: 'categorical_crossentropy' or 'mean_squared_error'
    """
    # Build DNN model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=inputs_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(outputs_train.shape[1], activation='softmax'))
    # compile
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy', coef])
    # train
    fscore = Fscore()
    model.fit(inputs_train, outputs_train, epochs=60, batch_size=256,
              validation_data=(inputs_test, outputs_test), callbacks=[fscore])
    # evaluate
    score = model.evaluate(inputs_test, outputs_test, batch_size=256)
    print("Eval: loss: %.4f - acc - %.4f - fscore: %.4f - coef: %.4f" %
          (score[0], score[1], fscore.get_data(), score[2]))


if __name__ == "__main__":
    inputs_train, outputs_train, inputs_test, outputs_test = \
        load_data("bags-of-words", "classification")
    dnn(inputs_train, outputs_train, inputs_test, outputs_test,
        loss='categorical_crossentropy')

    inputs_train, outputs_train, inputs_test, outputs_test = \
        load_data("bags-of-words", "regression")
    dnn(inputs_train, outputs_train, inputs_test, outputs_test,
        loss='mean_squared_error')

    inputs_train, outputs_train, inputs_test, outputs_test = \
        load_data("tf-idf", "classification")
    dnn(inputs_train, outputs_train, inputs_test, outputs_test,
        loss='categorical_crossentropy')

    inputs_train, outputs_train, inputs_test, outputs_test = \
        load_data("tf-idf", "regression")
    dnn(inputs_train, outputs_train, inputs_test, outputs_test,
        loss='mean_squared_error')
