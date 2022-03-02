from functions import *
from params import *


def main():

    ## Load data for SOL,BTC to USDT
    sol_usdt = read_csv_cryptodatadownload(
        "https://www.cryptodatadownload.com/cdd/Binance_SOLUSDT_d.csv"
    )
    btc_usdt = read_csv_cryptodatadownload(
        "https://www.cryptodatadownload.com/cdd/Binance_BTCUSDT_d.csv"
    )

    ## Filter the selected time period
    sol_usdt = prepare_data(sol_usdt, start_date, end_date)
    btc_usdt = prepare_data(btc_usdt, start_date, end_date)

    ## merge tha data
    data = sol_usdt.merge(btc_usdt, on=["date"], suffixes=("", "_btc"))

    ## Add extra features
    data = transform_features(data, train_split)

    ## Add moving average features
    data = pd.concat(
        [data.sort_values(by="date", ascending=False).reset_index(drop=True)]
        + [rolling_features(data, rolling_window_ftr, x, "date") for x in ma],
        axis=1,
    )

    ## Create train data by lagging the features
    df = (
        prepare_batch(
            data,
            rolling_window_ftr,
            [x for x in data if x not in dont_include] + ["Volume_SOL", "train"],
            ar,
            "date",
        )
        .dropna()
        .sort_values(by="date", ascending=False)
        .reset_index(drop=True)
    )

    non_ftr = ["date", "Volume_SOL", "train", "symbol"]
    features = [x for x in df if x not in non_ftr]
    target = "Volume_SOL"

    ## Train initial random forest
    rf = ensemble.RandomForestRegressor(**params)

    X_train, X_test = (
        df.loc[df["train"] == 1, features],
        df.loc[df["train"] == 0, features],
    )
    y_train, y_test = df.loc[df["train"] == 1, target], df.loc[df["train"] == 0, target]

    rf.fit(X_train, y_train)

    ## Get top 20 features based on SHAP feature importance
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(df[features])
    cols_pred_shap = (
        pd.DataFrame(shap_values, columns=features)
        .abs()
        .mean()
        .sort_values(ascending=False)
    )
    n = min(20, cols_pred_shap.shape[0])
    features_top = list(
        cols_pred_shap.iloc[
            0:n,
        ].index
    )

    ## Out of highly correlated features select only once with highest feature importance
    cor = df[features_top].corr()
    drop_cor = select_by_corr(cor, threshold=0.9)
    ## drop those features
    features_top = [x for x in features_top if x not in drop_cor[0]]

    X_train, X_test = (
        df.loc[df["train"] == 1, features_top],
        df.loc[df["train"] == 0, features_top],
    )
    y_train, y_test = df.loc[df["train"] == 1, target], df.loc[df["train"] == 0, target]

    ## Retrain RF with final features (and tweak the hyper paramaters)
    params["max_features"] = 8
    rf2 = ensemble.RandomForestRegressor(**params)
    rf2.fit(X_train, y_train)

    ## Do the predictions
    df["volume_predict"] = rf2.predict(df[features_top])
    df["volume_predict_q" + str(q)] = rf_quantile(rf2, df[features_top], q)
    df["volume_predict_q" + str(p)] = rf_quantile(rf2, df[features_top], p)

    ## Save predictions
    df[
        ["date", "Volume_SOL", "train"]
        + ["volume_predict", "volume_predict_q" + str(q), "volume_predict_q" + str(p)]
        + features_top
    ].to_csv("results_scored.csv")

    ## Save model
    pickle.dump(rf2, open("random_forest.pkl", "wb"))

    ## Save performnance metric
    with open("model_accuracy.txt", "w") as f:
        f.write("Perfromance for quantile %s \n" % p)
        f.write(
            "Train loss for quantile %s is : %s"
            % (
                str(p),
                tilted_loss(
                    p,
                    df[df["train"] == 1].dropna()["Volume_SOL"],
                    df[df["train"] == 1]["volume_predict_q" + str(p)],
                )[1],
            )
        )
        f.write("\n")
        f.write(
            "Test loss for quantile %s is : %s"
            % (
                str(p),
                tilted_loss(
                    p,
                    df[df["train"] == 0].dropna()["Volume_SOL"],
                    df[df["train"] == 0]["volume_predict_q" + str(p)],
                )[1],
            )
        )

        f.write("\n\n")
        f.write("Perfromance for quantile %s \n" % q)
        f.write(
            "Train loss for quantile %s is : %s"
            % (
                str(q),
                tilted_loss(
                    q,
                    df[df["train"] == 1].dropna()["Volume_SOL"],
                    df[df["train"] == 1]["volume_predict_q" + str(q)],
                )[1],
            )
        )
        f.write("\n")
        f.write(
            "Test loss for quantile %s is : %s"
            % (
                str(q),
                tilted_loss(
                    q,
                    df[df["train"] == 0].dropna()["Volume_SOL"],
                    df[df["train"] == 0]["volume_predict_q" + str(q)],
                )[1],
            )
        )

    # Save feature importance
    plot_shap(rf2, df[features_top], "shap_feature_importance.pdf")

    # Save plotter predictions
    plot_prediction(
        df[df["train"] == 0],
        "date",
        [
            "Volume_SOL",
            "volume_predict",
            "volume_predict_q" + str(p),
            "volume_predict_q" + str(q),
        ],
        "Plotted_predictions.pdf",
    )

    return


if __name__ == "__main__":
    main()
