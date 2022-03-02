import time

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
import xgboost as xgb
import shap
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    roc_auc_score,
    r2_score,
    mean_absolute_error,
)
import pickle
from datetime import datetime

from sklearn import ensemble


def read_csv_cryptodatadownload(url):
    """
    Function to load data from website www.CryptoDataDownload.com to pandas
    Also replaces spaces in column names by underscores

    """

    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context
    from io import StringIO
    import urllib.request

    # loading raw data from site
    try:
        with urllib.request.urlopen(url) as f:
            raw_data = f.read().decode("utf-8")
    except urllib.error.URLError as e:
        print(e.reason)
        return 1

    # Getting data to normal csv format
    raw_data = raw_data.replace("https://www.CryptoDataDownload.com\n", "")
    df = pd.read_csv(StringIO(raw_data))

    df.columns = [x.replace(" ", "_") for x in df.columns]

    return df


def prepare_data(df, start_date, end_date):
    """
    Helper function to filter data by date column

    """
    try:
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df[
            (df["date"] <= pd.Timestamp(end_date))
            & (df["date"] >= pd.Timestamp(start_date))
        ]
        df = df.drop(columns="unix")
    except:
        print("Wrong data format")
        return 1
    return df


def transform_features(data, train_split):
    """
    Helper function to create new features by direct transforms

    """
    try:
        data["open_diff"] = data["open"].diff()
        data["high_diff"] = data["high"].diff()
        data["low_diff"] = data["low"].diff()
        data["Volume_SOL_diff"] = data["Volume_SOL"].diff()
        data["Volume_USDT_diff"] = data["Volume_USDT"].diff()
        data["close_diff"] = data["close"].diff()
        ##
        data["open_btc_diff"] = data["open_btc"].diff()
        data["high_btc_diff"] = data["high_btc"].diff()
        data["low_btc_diff"] = data["low_btc"].diff()
        data["Volume_BTC_diff"] = data["Volume_BTC"].diff()
        data["Volume_USDT_btc_diff"] = data["Volume_USDT_btc"].diff()
        data["close_btc_diff"] = data["close_btc"].diff()

        data["daily_change"] = data["close"] - data["open"]
        data["daily_range"] = data["high"] - data["low"]
        data["daily_change_btc"] = data["close_btc"] - data["open_btc"]
        data["daily_range_btc"] = data["high_btc"] - data["low_btc"]

        data["train"] = (data["date"] < pd.Timestamp(train_split)).astype("int")

    except:
        print("Wrong data format")
        return 1

    return data


def rolling_features(data, cols, n_steps, date):
    """
    data - pandas dataframe , index will be reset!
    cols - list of column names to use for moving average features
    n_steps - length of moving window

    returns dataframe with moving window aggregations using max/min/avg/std
    the window for date==t starts with t-1 till t-n_steps -> can be directly used as feature (no future info)

    """
    data = data.sort_values(by=date, ascending=False).reset_index(drop=True)
    avg = []
    std = []
    max = []
    min = []

    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data) - 1:
            break
        std.append(data.loc[i + 1 : end_ix, cols].std().values)
        avg.append(data.loc[i + 1 : end_ix, cols].mean().values)
        max.append(data.loc[i + 1 : end_ix, cols].max().values)
        min.append(data.loc[i + 1 : end_ix, cols].min().values)

    for i in range(n_steps):
        avg.append(len(cols) * [np.nan])
        std.append(len(cols) * [np.nan])
        max.append(len(cols) * [np.nan])
        min.append(len(cols) * [np.nan])

    max2 = pd.DataFrame(
        columns=[x + "_max_" + str(n_steps) + "D" for x in cols], data=max
    )
    avg2 = pd.DataFrame(
        columns=[x + "_avg_" + str(n_steps) + "D" for x in cols], data=avg
    )
    min2 = pd.DataFrame(
        columns=[x + "_min_" + str(n_steps) + "D" for x in cols], data=min
    )
    std2 = pd.DataFrame(
        columns=[x + "_stdev_" + str(n_steps) + "D" for x in cols], data=std
    )

    result = pd.concat([max2, avg2, min2, std2], axis=1)

    return result


def prepare_batch(data, col_X, col_y, n_steps, date):

    """
    Prepares data in right format for the training of ML model

    data - pandas dataframe
    col_X - columns to use to get lagged values
    col_y - columns to keep unchanged
    n_steps - max lagged value to take

    returns original dataframe with columns created
     by lagging the columns in col_X and col_y unchanged

    """
    # prepares lagged features for list col_X,col_y are features that will stay in the row unchanged
    if type(col_X) != list:
        col_X = [col_X]
    if type(col_y) != list:
        col_y = [col_y]

    features = []
    data = data.sort_values(by=date, ascending=True).reset_index(drop=True)
    for n in reversed(range(1, n_steps + 1)):
        for x in col_X:
            features.append(x + "_lag" + str(n))
    pom = []
    for i in range(len(data)):
        end_ix = i + n_steps
        if end_ix > len(data) - 1:
            break
        pom.append(
            np.append(
                data.loc[i : end_ix - 1, col_X].values.reshape(1, len(col_X) * n_steps),
                data.loc[end_ix, col_y].values,
            )
        )

    result = pd.DataFrame(columns=features + col_y, data=pom)

    return result


def rf_quantile(m, X, q):
    """
    m -  random forest scikit model
    X - data to predict
    q - quantile

    Returns given quantile of random forest ensamble

    """
    rf_preds = []
    for estimator in m.estimators_:
        rf_preds.append(estimator.predict(X))
    rf_preds = np.array(rf_preds).transpose()  # One row per record.
    return np.percentile(rf_preds, q * 100, axis=1)


def tilted_loss(q, y, f):
    """

    q - quantile
    y - actuals
    f - predictions

    Standard quantile loss function
    tilted_loss(q, y, f)[0] - average loss
    tilted_loss(q, y, f)[1] - average relative loss
    """
    e = y - f
    return (
        np.mean(np.maximum(q * e, (q - 1) * e)),
        np.mean(np.maximum(q * e, (q - 1) * e)) / np.mean(y),
    )


def select_by_corr(cor_matrix, threshold):
    """
    corr_matrix - correlation matrix of features of model, features ordered by importance
    threshold -  correlation threshold

    Out of features with pairwise correlation higher than threshold selects only feature with highest feature importance
    Returns feature to drop


    """
    features_top = list(cor_matrix.index)
    cor_clusters = []
    for i in cor_matrix.index:
        cor_cand = []
        for k, j in enumerate(list(cor_matrix.loc[i])):
            if abs(j) > threshold:
                cor_cand.append(cor_matrix.loc[i].index[k])
        if len(cor_cand) > 1:
            cor_clusters.append(cor_cand)
    cor_clusters = [list(x) for x in set(tuple(x) for x in cor_clusters)]

    to_drop = []
    for c in cor_clusters:
        for f in features_top:
            if f in c:
                c2 = [x for x in c if x != f]
                to_drop = to_drop + c2
                break
    to_drop = list(set(to_drop))
    return to_drop, cor_clusters


def plot_shap(model, data, name):
    """
    Saves SHAP feature importance plot

    """
    fig = plt.figure(tight_layout=True)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    shap.summary_plot(shap_values, data, max_display=20, plot_type="bar")
    fig.savefig(name)


def plot_prediction(data, time_index, cols, name):
    """
    Helper function to save pandas time series plot
    """
    ax = data.set_index(time_index)[cols].plot(figsize=(20, 10))
    ax.figure.savefig(name)
