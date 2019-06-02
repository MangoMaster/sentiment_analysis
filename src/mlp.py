import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from utilnn import accuracy, fscore, coef


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


def save_model(model, model_file_name):
    model_file_path = os.path.join(
        os.path.pardir, "models", model_file_name + ".h5")
    model.save(model_file_path)


def mlp(inputs_train, outputs_train, inputs_test, outputs_test, loss):
    """
    Fully connected neural network.
    @param loss: 'categorical_crossentropy' or 'mean_squared_error'
    """
    # Split to train-set and validation-set
    split_at = len(inputs_train) - len(inputs_train) * 2 // 10
    (inputs_train, inputs_validation) = \
        (inputs_train[:split_at], inputs_train[split_at:])
    (outputs_train, outputs_validation) = \
        (outputs_train[:split_at], outputs_train[split_at:])
    # Build MLP model
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=inputs_train.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(outputs_train.shape[1], activation='softmax'))
    print(model.summary())
    # compile
    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    # train
    if loss == 'categorical_crossentropy':
        early_stopping = EarlyStopping(
            min_delta=0.001, patience=5, restore_best_weights=True)
    elif loss == 'mean_squared_error':
        early_stopping = EarlyStopping(
            min_delta=0.0003, patience=5, restore_best_weights=True)
    else:
        raise ValueError(
            "loss should be 'categorical_crossentropy' or 'mean_squared_error'.")
    model.fit(inputs_train, outputs_train, epochs=100, batch_size=256,
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


if __name__ == "__main__":
    inputs_train, outputs_train, inputs_test, outputs_test = \
        load_data("bags-of-words", "classification")
    model = mlp(inputs_train, outputs_train, inputs_test, outputs_test,
                loss='categorical_crossentropy')
    save_model(model, "mlp_bags-of-words_classification")

    inputs_train, outputs_train, inputs_test, outputs_test = \
        load_data("bags-of-words", "regression")
    model = mlp(inputs_train, outputs_train, inputs_test, outputs_test,
                loss='mean_squared_error')
    save_model(model, "mlp_bags-of-words_regression")

    inputs_train, outputs_train, inputs_test, outputs_test = \
        load_data("tf-idf", "classification")
    model = mlp(inputs_train, outputs_train, inputs_test, outputs_test,
                loss='categorical_crossentropy')
    save_model(model, "mlp_tf-idf_classification")

    inputs_train, outputs_train, inputs_test, outputs_test = \
        load_data("tf-idf", "regression")
    model = mlp(inputs_train, outputs_train, inputs_test, outputs_test,
                loss='mean_squared_error')
    save_model(model, "mlp_tf-idf_regression")
