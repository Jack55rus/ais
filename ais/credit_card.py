import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from immune_system import NegativeSelection
import numpy as np


def credit_card():
    np.random.seed(3)
    # load whole dataset
    df = pd.read_csv('../data/creditcard.csv')

    # split into normal and abnormal
    norm_df = df.loc[df.Class == 0]
    norm_df = norm_df.drop(columns=['Time', 'Class'])

    abnorm_df = df.loc[df.Class == 1]
    abnorm_df = abnorm_df.drop(columns=['Time', 'Class'])
    # make min-max scaling
    scaler = MinMaxScaler()

    norm_df = scaler.fit_transform(norm_df)
    random_indices = np.random.permutation(len(norm_df))
    norm_df = norm_df[random_indices]
    test_size = int(norm_df.shape[0] * 0.01)
    norm_df_train = norm_df[test_size:, :]
    norm_df_test = norm_df[:test_size, :]

    abnorm_df = scaler.fit_transform(abnorm_df)
    # feed normal to NS
    nsa = NegativeSelection(num_detectors=350)
    nsa.fit(norm_df_train)
    # feed abnormal to predict
    ans = nsa.predict(abnorm_df)
    print('number of 1 preds: {} out of {} samples'.format(np.sum(nsa.predict(norm_df_test)), norm_df_test.shape[0]))
    print('number of 1 preds: {} out of {} samples'.format(np.sum(ans), abnorm_df.shape[0]))


def speech():
    feat = np.load('../data/features.npy')
    labels = np.load('../data/labels.npy')

    norm_inds = np.where(labels == 0)[0]
    anom_inds = np.where(labels == 1)[0]

    scaler = MinMaxScaler()

    feat = scaler.fit_transform(feat)
    test_size = int(len(norm_inds) * 0.05)
    feat_train_norm = feat[norm_inds][test_size:, :]
    feat_test_norm = feat[norm_inds][:test_size, :]
    feat_test_anom = feat[anom_inds]

    nsa = NegativeSelection(num_detectors=2500, eps=0.0)
    nsa.fit(feat_train_norm)
    # feed abnormal to predict
    ans = nsa.predict(feat_test_anom)
    print('number of 1 preds: {} out of {} samples'.format(np.sum(nsa.predict(feat_test_norm)), feat_test_norm.shape[0]))
    print('number of 1 preds: {} out of {} samples'.format(np.sum(ans), feat_test_anom.shape[0]))


    # print(labels[np.argwhere(labels > 0)])
    # print(labels[np.nonzero(labels)])
    # print(np.argwhere(labels > 0).tolist())
    # print(np.where(labels == 0)[0])

def mnist():
    digit = 8
    scaler = MinMaxScaler()
    mnist_train = pd.read_csv('../data/mnist_train.csv')
    train_set = mnist_train.loc[mnist_train.label != digit]
    train_set = np.array(train_set.iloc[:, 1:]) / 255 + 0.25  # normalize
    random_indices = np.random.permutation(len(train_set))
    train_set = train_set[random_indices]
    train_set = scaler.fit_transform(train_set)

    mnist_test = pd.read_csv('../data/mnist_test.csv')
    mnist_test_not_digit = mnist_test.loc[mnist_test.label != digit]
    mnist_test_not_digit = np.array(mnist_test_not_digit.iloc[:, 1:]) / 255
    mnist_test_not_digit = scaler.fit_transform(mnist_test_not_digit)

    mnist_test_digit = mnist_test.loc[mnist_test.label == digit]
    mnist_test_digit = np.array(mnist_test_digit.iloc[:, 1:]) / 255
    mnist_test_digit = scaler.fit_transform(mnist_test_digit)

    nsa = NegativeSelection(num_detectors=200)
    nsa.fit(train_set)
    ans_not_digit = nsa.predict(mnist_test_not_digit)
    ans_digit = nsa.predict(mnist_test_digit)

    print('number of 1 preds on normal data: {} out of {} samples'.format(np.sum(ans_not_digit), ans_not_digit.shape[0]))
    print('number of 1 preds on abnormal data: {} out of {} samples'.format(np.sum(ans_digit), ans_digit.shape[0]))

if __name__ == '__main__':
    credit_card()
    # speech()
    # mnist()
