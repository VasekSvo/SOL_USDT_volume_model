##########################
## Various model paramaters
##########################


# selected quantiles
q = 0.8
p = round(1 - q, 1)

## end date of dataset
end_date = "2022-02-12"
## start date of dataset
start_date = "2021-04-01"
## date where test set starts
train_split = "2022-01-01"

# number of lag values that will be taken for feature prep
ar = 2
# moving average windows
ma = [28, 14, 7]
# threshold for correlation based feature reduction
threshold_corr=0.9

## columns to use for moving average/lagged features
rolling_window_ftr = [
    "open",
    "high",
    "low",
    "close",
    "Volume_SOL",
    "Volume_USDT",
    "tradecount",
    "open_btc",
    "high_btc",
    "low_btc",
    "close_btc",
    "Volume_BTC",
    "Volume_USDT_btc",
    "tradecount_btc",
    "open_diff",
    "high_diff",
    "low_diff",
    "Volume_SOL_diff",
    "Volume_USDT_diff",
    "close_diff",
    "open_btc_diff",
    "high_btc_diff",
    "low_btc_diff",
    "Volume_BTC_diff",
    "Volume_USDT_btc_diff",
    "close_btc_diff",
    "daily_change",
    "daily_range",
    "daily_change_btc",
    "daily_range_btc",
]

## features to drop (they cannot be use for training they contain future info)
dont_include = [
    "open",
    "high",
    "low",
    "close",
    "train",
    "Volume_SOL",
    "Volume_USDT",
    "tradecount",
    "symbol_btc",
    "open_btc",
    "high_btc",
    "low_btc",
    "close_btc",
    "Volume_BTC",
    "Volume_USDT_btc",
    "tradecount_btc",
    "open_diff",
    "high_diff",
    "low_diff",
    "Volume_SOL_diff",
    "Volume_USDT_diff",
    "close_diff",
    "open_btc_diff",
    "high_btc_diff",
    "low_btc_diff",
    "Volume_BTC_diff",
    "Volume_USDT_btc_diff",
    "close_btc_diff",
    "daily_change",
    "daily_range",
    "daily_change_btc",
    "daily_range_btc",
]

## Random forest params
N_ESTIMATORS = 10000
params = {
    "criterion": "mae",
    "max_features": 150,
    "min_samples_leaf": 8,
    "min_samples_split": 8,
    "min_impurity_decrease": 1,
    "random_state": 2,
    "verbose": True,
    "ccp_alpha": 0.3,
    "max_samples": None,
    "bootstrap": True,
    "max_depth": 8,
    "n_jobs": -1,
}
