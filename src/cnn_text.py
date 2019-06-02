import os
import functools
import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Concatenate, Dense, Dropout
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


def cnn(inputs_train, outputs_train, inputs_test, outputs_test, loss, train_embedding):
    """
    Convolutional neural network for sentence classification.
    See "Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP 2014), 1746–1751." for details.
    @param loss: 'categorical_crossentropy' or 'mean_squared_error'
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
    model_input = Input(shape=(inputs_train.shape[1],))
    embedding = embedding_layer(model_input)
    conv_blocks = []
    for kernel_size in range(2, 8):
        for _ in range(2):  # 2 filters for each kernel_size
            conv = Conv1D(4, kernel_size, activation="relu")(embedding)
            conv = MaxPooling1D(3)(conv)
            conv = Dropout(0.25)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
    x = Concatenate()(conv_blocks)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    model_output = Dense(outputs_train.shape[1], activation="softmax")(x)
    model = Model(model_input, model_output)
    print(model.summary())
    # compile
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    # train
    if loss == 'categorical_crossentropy':
        early_stopping = EarlyStopping(
            min_delta=0.003, patience=5, restore_best_weights=True)
    elif loss == 'mean_squared_error':
        early_stopping = EarlyStopping(
            min_delta=0.0003, patience=5, restore_best_weights=True)
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
