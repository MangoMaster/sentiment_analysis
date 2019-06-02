import os
import functools
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from utilnn import accuracy, fscore, coef


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


def save_model(model, model_file_name):
    model_file_path = os.path.join(
        os.path.pardir, "models", model_file_name + ".h5")
    model.save(model_file_path)


def lstm(inputs_train, outputs_train, inputs_test, outputs_test, loss, train_embedding):
    """
    LSTM neural network.
    @param loss: 'classification' or 'regression'
    @param train_embedding: 0 - initialize with word_embedding_matrix, trainable=False
                            1 - initialize with word_embedding_matrix, trainable=True
                            2 - initialize with random matrix, trainable=True
    """
    # Load word-embedding matrix
    word_embedding_matrix_file_path = os.path.join(
        os.path.pardir, "data", "word-embedding_matrix")
    with open(word_embedding_matrix_file_path, 'rb') as word_embedding_matrix_file:
        word_embedding_matrix = np.load(word_embedding_matrix_file)
    # Split to train-set and validation-set
    split_at = len(inputs_train) - len(inputs_train) * 2 // 10
    (inputs_train, inputs_validation) = \
        (inputs_train[:split_at], inputs_train[split_at:])
    (outputs_train, outputs_validation) = \
        (outputs_train[:split_at], outputs_train[split_at:])
    # Build LSTM model
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
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(outputs_train.shape[1], activation='softmax'))
    print(model.summary())
    # compile
    model.compile(loss=loss, optimizer='rmsprop', metrics=['accuracy'])
    # train
    if loss == 'categorical_crossentropy':
        early_stopping = EarlyStopping(
            min_delta=0.005, patience=3, restore_best_weights=True)
    elif loss == 'mean_squared_error':
        early_stopping = EarlyStopping(
            min_delta=0.001, patience=3, restore_best_weights=True)
    else:
        raise ValueError(
            "loss should be 'categorical_crossentropy' or 'mean_squared_error'.")
    model.fit(inputs_train, outputs_train, epochs=100, batch_size=128,
              validation_data=(inputs_validation, outputs_validation), callbacks=[early_stopping])
    # evaluate
    outputs_test_pred = np.asarray(model.predict(inputs_test))
    acc_eval = accuracy(outputs_test, outputs_test_pred)
    fscore_eval = fscore(outputs_test, outputs_test_pred)
    coef_eval = coef(outputs_test, outputs_test_pred)
    print("Evaluation: acc - %.4f - fscore: %.4f - coef: %.4f - pvalue: %.4f" %
          (acc_eval, fscore_eval, coef_eval[0], coef_eval[1]))
    # return model
    return model


lstm_static = functools.partial(lstm, train_embedding=0)
lstm_non_static = functools.partial(lstm, train_embedding=1)
lstm_rand = functools.partial(lstm, train_embedding=2)

if __name__ == "__main__":
    inputs_train, outputs_train, inputs_test, outputs_test = \
        load_data("classification")
    model = lstm_static(inputs_train, outputs_train, inputs_test, outputs_test,
                        loss='categorical_crossentropy')
    save_model(model, "lstm_static_classification")
    model = lstm_non_static(inputs_train, outputs_train, inputs_test, outputs_test,
                            loss='categorical_crossentropy')
    save_model(model, "lstm_non_static_classification")
    model = lstm_rand(inputs_train, outputs_train, inputs_test, outputs_test,
                      loss='categorical_crossentropy')
    save_model(model, "lstm_rand_classification")

    inputs_train, outputs_train, inputs_test, outputs_test = \
        load_data("regression")
    model = lstm_static(inputs_train, outputs_train, inputs_test, outputs_test,
                        loss='mean_squared_error')
    save_model(model, "lstm_static_regression")
    model = lstm_non_static(inputs_train, outputs_train, inputs_test, outputs_test,
                            loss='mean_squared_error')
    save_model(model, "lstm_non_static_regression")
    model = lstm_rand(inputs_train, outputs_train, inputs_test, outputs_test,
                      loss='mean_squared_error')
    save_model(model, "lstm_rand_regression")
