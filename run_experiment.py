import pandas as pd
import numpy as np
import time
from model import Model
from sklearn.model_selection import StratifiedKFold
from utils import cat_cols_info

features_float = ['days_from_last_order', 'order_price', 'days_to_flight',
                  'num_tickets', 'baby_ratio', 'adult_ratio', 'teen_ratio',
                  'field12_log', 'field13_log', 'field14_log', 'field17_log',
                  'field22_log', 'field25_log', 'filed26_log']

features_cat = ['order_month', 'flight_month',
                'week_day_flight', 'week_day',
                'year', 'quarter', 'hour',
                'field19', 'field23', 'field27']

features_binary = ['is_promo', 'field6_is_zero',
                   'field10', 'field7', 'field8',
                   'indicator_goal21', 'indicator_goal22',
                   'indicator_goal23', 'indicator_goal24',
                   'indicator_goal25']
features_final = features_binary + features_float + features_cat + ['field6']

target = 'goal1'
targets = ['goal21', 'goal22', 'goal23', 'goal24', 'goal25', 'goal1']


def preprocess(df):
    df['days_from_last_order'] = np.log(df['field0'] + 1)
    df['order_price'] = np.log(df['field1'] / 0.07757136 + 1.10765391 * 10)
    df['order_month'] = df['field2']
    df['flight_month'] = df['field3']
    df['order_num'] = np.log(df['field4'] + 1)  # other
    df['is_promo'] = df['field5']

    df['days_to_flight'] = np.log(df['field16'] + 1)
    df['week_day'] = df['field18']
    df['week_day_flight'] = df['field20']
    df['hour'] = df['field11']
    df['year'] = df['field21']
    df['quarter'] = df['field29']

    df['num_tickets'] = df['field15']
    df['num_tickets_baby'] = df['field9']
    df['num_tickets_adult'] = df['field24']
    df['num_tickets_teen'] = df['field28']

    df['baby_ratio'] = df['num_tickets_baby'] / df['num_tickets']
    df['adult_ratio'] = df['num_tickets_adult'] / df['num_tickets']
    df['teen_ratio'] = df['num_tickets_teen'] / df['num_tickets']

    df['field12_log'] = np.log(df['field12'] + 1)
    df['field13_log'] = np.log(df['field13'] + 1)
    df['field14_log'] = np.log(df['field14'] + 1)
    df['field17_log'] = np.log(df['field17'] + 1)
    df['field22_log'] = np.log(df['field22'] + 1)
    df['field25_log'] = np.log(df['field25'] + 1)
    df['filed26_log'] = np.log(df['field26'] + 1)
    df['field6_is_zero'] = df['field6'].map(lambda x: int(x == 0))

    user_agg_means = df.groupby(['userid'])[features_float].mean().reset_index()
    user_agg_means.columns = [x for x in \
                              map(lambda x: x if (x == 'userid') else \
                                  'user_mean_' + x, user_agg_means.columns)]

    user_orders_count = df.groupby(['userid'])['orderid'].count().reset_index()
    user_orders_count.columns = ['userid', 'user_order_count']
    df = df.merge(user_agg_means, on='userid').merge(user_orders_count, on='userid')
    return df


def execute_experiment(encoders_list, validation_type, experiment_description, models_params):
    train_pth = './data/onetwotrip_challenge_train.csv'
    test_pth = './data/onetwotrip_challenge_test.csv'
    results = {}

    # training params
    N_SPLITS = 5
    model_validation = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    encoder_validation = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=2019)

    # load train
    train = pd.read_csv(train_pth)
    train = preprocess(train)

    # load test
    test = pd.read_csv(test_pth)
    test = preprocess(test)

    cat_features = features_cat

    X_train, X_test, y_train = train[features_final], test[features_final], train[target]
    X_train, X_test = X_train.reset_index(drop=False), X_test.reset_index(drop=False)
    y_train = np.array(y_train)

    results[train_pth] = {}
    results[train_pth]["info"] = {
        "experiment_description": experiment_description,
        "train_shape": X_train.shape, "test_shape": X_test.shape,
        "mean_target_train": np.mean(y_train),
        "num_cat_cols": len(cat_features), "cat_cols_info": cat_cols_info(X_train, X_test, cat_features),
    }

    predict = np.zeros((X_test.shape[0], len(encoders_list)))
    for i, encoders_tuple in enumerate(encoders_list):
        print(f"\n\nCurrent itteration : {encoders_tuple}, {train_pth}\n\n")

        time_start = time.time()

        # train models
        lgb_model = Model(cat_validation=validation_type, encoders_names=encoders_tuple, cat_cols=cat_features,
                          model_params=models_params)
        train_score, val_score, avg_num_trees = lgb_model.fit(X_train, y_train)
        y_hat, test_features = lgb_model.predict(X_test)

        # pd.DataFrame({"predictions": y_hat}).to_csv(file_pth, index=False)
        time_end = time.time()

        # write and save results
        results[train_pth][str(encoders_tuple)] = {"train_score": train_score,
                                                     "val_score": val_score,
                                                     "time": time_end - time_start,
                                                     "features_before_encoding": X_train.shape[1],
                                                     "features_after_encoding": test_features,
                                                     "avg_tress_number": avg_num_trees}
        predict[:, i] = y_hat
    for k, v in results[train_pth].items():
        print(k, v, "\n\n")

    return predict.mean(axis=1)

if __name__ == "__main__":
    models_params = {'n_estimators': 203,
                     'boosting_type': 'gbdt',
                     'class_weight': None,
                     'colsample_bytree': 0.38,
                     'importance_type': 'split',
                     'learning_rate': 0.02,
                     'max_depth': 7,
                     'metrics': 'AUC',
                     'min_child_samples': 12,
                     'min_child_weight': 0.003,
                     'min_split_gain': 0.96,
                     'n_jobs': -1,
                     'num_leaves': 32,
                     'objective': 'binary',
                     'reg_alpha': 0.8,
                     'reg_lambda': 0.8,
                     'silent': True,
                     'subsample': 0.6,
                     'subsample_for_bin': 200000,
                     'subsample_freq': 14,
                     "random_state": 42}

    encoders_list = [
        ## ("HelmertEncoder",),  # non double
        ## ("SumEncoder",),  # non double
        ## ("LeaveOneOutEncoder",),
        ("FrequencyEncoder",),
        ("MEstimateEncoder",),
        ("TargetEncoder",),
        ("WOEEncoder",),
        ## ("BackwardDifferenceEncoder",),  # non double
        ("JamesSteinEncoder",),
        ("OrdinalEncoder",),
        ("CatBoostEncoder",),
    ]

    validation_type = "Single"
    experiment_description = f"Check single encoder, {validation_type} validation"
    execute_experiment(encoders_list, validation_type, experiment_description, models_params)
