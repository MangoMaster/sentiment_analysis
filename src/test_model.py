import os
import numpy as np
from keras.models import load_model
from utilnn import accuracy, fscore, coef


def load_data(texts_prefix):
    """
    @param texts_prefix: 'bags-of-words' or 'tf-idf' or 'word-embedding'
    @return: (inputs_test, outputs_test)
    """
    # Generate file path
    inputs_test_file_path = os.path.join(
        os.path.pardir, "data", texts_prefix + "_test")
    outputs_test_file_path = os.path.join(
        os.path.pardir, "data", "regression" + "_test")  # test label always use regression
    # Get data
    with open(inputs_test_file_path, 'rb') as inputs_test_file:
        inputs_test = np.load(inputs_test_file)
    with open(outputs_test_file_path, 'rb') as outputs_test_file:
        outputs_test = np.load(outputs_test_file)
    return (inputs_test, outputs_test)


def verify_model(model_file_name, inputs_test, outputs_test):
    # load model
    model_file_path = os.path.join(
        os.path.pardir, "models", model_file_name + ".h5")
    model = load_model(model_file_path)
    print(model_file_name)
    # model structure
    print(model.summary())
    # model evaluation
    outputs_test_pred = np.asarray(model.predict(inputs_test))
    acc_eval = accuracy(outputs_test, outputs_test_pred)
    fscore_eval = fscore(outputs_test, outputs_test_pred)
    coef_eval = coef(outputs_test, outputs_test_pred)
    print("Evaluation: acc - %.4f - fscore: %.4f - coef: %.4f - pvalue: %.4f" %
          (acc_eval, fscore_eval, coef_eval[0], coef_eval[1]))


if __name__ == "__main__":
    inputs_test, outputs_test = load_data("bags-of-words")
    verify_model("mlp_bags-of-words_classification", inputs_test, outputs_test)
    verify_model("mlp_bags-of-words_regression", inputs_test, outputs_test)

    inputs_test, outputs_test = load_data("tf-idf")
    verify_model("mlp_tf-idf_classification", inputs_test, outputs_test)
    verify_model("mlp_tf-idf_regression", inputs_test, outputs_test)

    inputs_test, outputs_test = load_data("word-embedding")
    for model_name in ("cnn", "cnn-text", "lstm", "rnn"):
        for train_embedding in ("static", "non_static", "rand"):
            for labels_prefix in ("classification", "regression"):
                verify_model(model_name + '_' + train_embedding+'_' + labels_prefix,
                             inputs_test, outputs_test)
