import os
import re
import pickle
import numpy as np
from keras.preprocessing.text import Tokenizer


def collect_data(data_file_name, output_file_name_suffix):
    """
    Collect data: times, texts, labels.
    times: list of time (in str), using pickle to save.
    texts: list of text, using pickle to save.
    labels: 2d array, using np to save.
    """
    # Generate file path
    data_file_path = os.path.join(os.path.pardir, "data", data_file_name)
    times_file_path = os.path.join(
        os.path.pardir, "data", "times_" + output_file_name_suffix)
    texts_file_path = os.path.join(
        os.path.pardir, "data", "texts_" + output_file_name_suffix)
    labels_file_path = os.path.join(
        os.path.pardir, "data", "labels_" + output_file_name_suffix)
    # Collect data
    times = []
    texts = []
    labels = []
    with open(data_file_path, 'r') as data_file:
        for line in data_file:
            time_str, label_str, text = line.split('\t')
            times.append(time_str)
            text = text[:-1]  # pop '\n'
            texts.append(text)
            label = [int(s) for s in re.findall(r'\d+', label_str)]
            label.pop(0)
            labels.append(label)
    # Write data to files
    with open(times_file_path, 'wb') as times_file:
        pickle.dump(times, times_file)
    with open(texts_file_path, 'wb') as texts_file:
        pickle.dump(texts, texts_file)
    with open(labels_file_path, 'wb') as labels_file:
        np.save(labels_file, np.asarray(labels))


def verify_data(file_name_suffix):
    """
    Verify data is properly collected and saved.
    """
    # Generate file path
    times_file_path = os.path.join(
        os.path.pardir, "data", "times_" + file_name_suffix)
    texts_file_path = os.path.join(
        os.path.pardir, "data", "texts_" + file_name_suffix)
    labels_file_path = os.path.join(
        os.path.pardir, "data", "labels_" + file_name_suffix)
    # Read data from files
    with open(times_file_path, 'rb') as times_file:
        times = pickle.load(times_file)
    with open(texts_file_path, 'rb') as texts_file:
        texts = pickle.load(texts_file)
    with open(labels_file_path, 'rb') as labels_file:
        labels = np.load(labels_file)
    print(times)
    print(texts)
    print(labels)


def bags_of_words():
    """
    Convert text data to bags-of-words, using np to save.
    w(i)(j)表示第i个文档中第j个词出现的频率
    """
    # Generate file path
    texts_train_file_path = os.path.join(os.path.pardir, "data", "texts_train")
    output_train_file_path = os.path.join(
        os.path.pardir, "data", "bags-of-words_train")
    texts_test_file_path = os.path.join(os.path.pardir, "data", "texts_test")
    output_test_file_path = os.path.join(
        os.path.pardir, "data", "bags-of-words_test")
    # Convert train text to bags-of-words
    with open(texts_train_file_path, 'rb') as texts_train_file:
        texts_train = pickle.load(texts_train_file)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts_train)
    matrix_train = tokenizer.texts_to_matrix(texts_train, mode="freq")
    # Save train matrix
    print(matrix_train.shape)
    with open(output_train_file_path, 'wb') as output_file:
        np.save(output_file, matrix_train)
    # Convert test text to bags-of-words
    with open(texts_test_file_path, 'rb') as texts_test_file:
        texts_test = pickle.load(texts_test_file)
    matrix_test = tokenizer.texts_to_matrix(texts_test, mode="freq")
    # Save test matrix
    print(matrix_test.shape)
    with open(output_test_file_path, 'wb') as output_file:
        np.save(output_file, matrix_test)


def tf_idf():
    """
    Convert text data to tf-idf, using np to save.
    w(i)(j) = tf(i)(j) * idf(j) 表示第i个文档中第j个词的tf-idf值
    tf(i)(j) = 1 + np.log(count(i)(j)) 
    count(i)(j)表示第i个文档中第j个词出现的次数
    idf(j) = log(1 + document_count / (1 + df(j)))
    df(j)表示出现第j个词的文档数
    """
    # Generate file path
    texts_train_file_path = os.path.join(os.path.pardir, "data", "texts_train")
    output_train_file_path = os.path.join(
        os.path.pardir, "data", "tf-idf_train")
    texts_test_file_path = os.path.join(os.path.pardir, "data", "texts_test")
    output_test_file_path = os.path.join(
        os.path.pardir, "data", "tf-idf_test")
    # Convert train text to bags-of-words
    with open(texts_train_file_path, 'rb') as texts_train_file:
        texts_train = pickle.load(texts_train_file)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts_train)
    matrix_train = tokenizer.texts_to_matrix(texts_train, mode="tfidf")
    # Save train matrix
    print("document count: ", tokenizer.document_count)
    print(matrix_train.shape)
    with open(output_train_file_path, 'wb') as output_file:
        np.save(output_file, matrix_train)
    # Convert test text to bags-of-words
    with open(texts_test_file_path, 'rb') as texts_test_file:
        texts_test = pickle.load(texts_test_file)
    matrix_test = tokenizer.texts_to_matrix(texts_test, mode="tfidf")
    # Save test matrix
    print(matrix_test.shape)
    with open(output_test_file_path, 'wb') as output_file:
        np.save(output_file, matrix_test)


def classification(file_name_suffix):
    """
    Convert label data to classification-label, using np to save.
    转化为单标签预测(分类问题)，标签中最大值为1，其余为0.
    """
    # Generate file path
    labels_file_path = os.path.join(
        os.path.pardir, "data", "labels_" + file_name_suffix)
    output_file_path = os.path.join(
        os.path.pardir, "data", "classification_" + file_name_suffix)
    # Convert labels to classification-labels
    with open(labels_file_path, 'rb') as labels_file:
        labels = np.load(labels_file)
    max_indexes = labels.argmax(axis=1)
    classification = np.zeros(labels.shape)
    for i, j in enumerate(max_indexes):
        classification[i][j] = 1
    # Save matrix
    print(classification)
    with open(output_file_path, 'wb') as output_file:
        np.save(output_file, classification)


def regression(file_name_suffix):
    """
    Convert label data to regression-label, using np to save.
    归一化情感分布(回归问题)，标签中所有值求和为1.
    """
    # Generate file path
    labels_file_path = os.path.join(
        os.path.pardir, "data", "labels_" + file_name_suffix)
    output_file_path = os.path.join(
        os.path.pardir, "data", "regression_" + file_name_suffix)
    # Convert labels to regression-labels
    with open(labels_file_path, 'rb') as labels_file:
        labels = np.load(labels_file)
    sums = labels.sum(axis=1)
    regression = labels.astype('float64')
    for i, j in np.ndindex(regression.shape):
        regression[i][j] /= sums[i]
    # Save matrix
    print(regression)
    with open(output_file_path, 'wb') as output_file:
        np.save(output_file, regression)


if __name__ == "__main__":
    # Demo data
    file_name_suffix = "demo"
    collect_data("sinanews.demo", file_name_suffix)
    verify_data(file_name_suffix)
    classification(file_name_suffix)
    regression(file_name_suffix)
    # Train data
    file_name_suffix = "train"
    collect_data("sinanews.train", file_name_suffix)
    verify_data(file_name_suffix)
    classification(file_name_suffix)
    regression(file_name_suffix)
    # Test data
    file_name_suffix = "test"
    collect_data("sinanews.test", file_name_suffix)
    verify_data(file_name_suffix)
    classification(file_name_suffix)
    regression(file_name_suffix)
    # Train
    bags_of_words()
    tf_idf()
