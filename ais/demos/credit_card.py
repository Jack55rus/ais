import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ais.config import CommonConfig
from ais.immune_tools.immune_system import NegativeSelection

# todo: multiprocessing
# todo: additional stoppage criterion


def credit_card():
    np.random.seed(3)
    # load whole dataset
    df = pd.read_csv(CommonConfig.data_dir / "creditcard.csv")
    # split into normal and abnormal
    norm_df = df.loc[df.Class == 0]
    norm_df = norm_df.drop(columns=["Time", "Class"])

    abnorm_df = df.loc[df.Class == 1]
    abnorm_df = abnorm_df.drop(columns=["Time", "Class"])
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
    # nsa.save_model(CommonConfig.data_dir / 'model.pkl')
    # nsa.load_model(CommonConfig.data_dir / "model.pkl")
    # feed abnormal to predict
    ans = nsa.predict(abnorm_df)
    print("number of 1 preds: {} out of {} samples".format(np.sum(nsa.predict(norm_df_test)), norm_df_test.shape[0]))
    print("number of 1 preds: {} out of {} samples".format(np.sum(ans), abnorm_df.shape[0]))


if __name__ == "__main__":
    credit_card()
