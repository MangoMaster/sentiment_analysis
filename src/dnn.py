import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from utilnn import Fscore, coef


def dnn(texts_prefix, labels_prefix):
    """
    Fully connected neural network.
    """
    # Generate file path
    texts_train_file_path = os.path.join(
        os.path.pardir, "data", texts_prefix + "_train")
    labels_train_file_path = os.path.join(
        os.path.pardir, "data", labels_prefix + "_train")
    texts_test_file_path = os.path.join(
        os.path.pardir, "data", texts_prefix + "_test")
    labels_test_file_path = os.path.join(
        os.path.pardir, "data", "regression" + "_test")  # test label always use regression
    # Get data
    with open(texts_train_file_path, 'rb') as texts_train_file:
        texts_train = np.load(texts_train_file)
    with open(labels_train_file_path, 'rb') as labels_train_file:
        labels_train = np.load(labels_train_file)
    with open(texts_test_file_path, 'rb') as texts_test_file:
        texts_test = np.load(texts_test_file)
    with open(labels_test_file_path, 'rb') as labels_test_file:
        labels_test = np.load(labels_test_file)
    # Build DNN model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=texts_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(labels_train.shape[1], activation='softmax'))
    # loss
    if labels_prefix == "classification":
        loss = 'categorical_crossentropy'
    elif labels_prefix == "regression":
        loss = 'mean_squared_error'
    else:
        raise ValueError("labels_prefix doesn't exist.")
    # optimizer
    optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # compile
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', coef])
    # train
    fscore = Fscore()
    model.fit(texts_train, labels_train, epochs=20, batch_size=128,
              validation_data=(texts_test, labels_test), callbacks=[fscore])
    # evaluate
    score = model.evaluate(texts_test, labels_test, batch_size=128)
    print("Eval: loss: %.4f - acc - %.4f - fscore: %.4f - coef: %.4f" %
          (score[0], score[1], fscore.get_data(), score[2]))


if __name__ == "__main__":
    dnn("bags-of-words", "classification")
    dnn("bags-of-words", "regression")
    dnn("tf-idf", "classification")
    dnn("tf-idf", "regression")
