import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from data.metrics import get_smoothed_mean_log_accuracy_ratio

## Target processing
def box_cox(x, lmbda):
    return (x ** lmbda - 1) / lmbda

def anti_box_cox(x, lmbda):
    x = lmbda * x + 1
    return np.sign(x) * np.abs(x) ** (1 / lmbda)


### Features
def make_harmonic_features(value, period=24):
    value *= 2 * np.pi / period
    return np.cos(value), np.sin(value)

def aggregate_history(history_merged_users):
    """
    Какую инфу можно вытащить из history (publisher-user_id)?
    1) Агрегаты
    2) ALS
    """

    # Временные признаки
    df = history_merged_users.copy()
    df['time'] = df['hour'].map(lambda x: x % 24)
    df['day_of_week'] = df['hour'].map(lambda x: (x // 24) % 7 + 1)
    df['is_weekend'] = df['day_of_week'].map(lambda x: int(x in [6, 7]))
    df['time_harmonic_0'] = make_harmonic_features(df['hour'])[0]
    df['time_harmonic_1'] = make_harmonic_features(df['hour'])[1]

    # Аггрегаты по user_id
    users = df \
        .groupby(['user_id']) \
        [['cpm', 'is_weekend', 'time', 'time_harmonic_0', 'time_harmonic_1', 'publisher']] \
        .mean()
    users.columns = ['user_mean_' + x for x in users.columns]
    users['user_id'] = users.index
    users = users.reset_index(drop=True)

    user_unique_publishers = df \
        .groupby(['user_id'])['publisher'] \
        .apply(set) \
        .to_frame()

    user_unique_publishers['user_unique_publishers_cnt'] = \
        user_unique_publishers['publisher'].map(len)
    user_unique_publishers = user_unique_publishers.drop(columns='publisher')
    user_unique_publishers['user_id'] = user_unique_publishers.index
    user_unique_publishers = user_unique_publishers.reset_index(drop=True)

    # Аггрегаты по publisher
    publishers = df \
        .groupby(['publisher']) \
        [['time', 'time_harmonic_0', 'time_harmonic_1', 'is_weekend', \
          'cpm', 'user_id', 'sex', 'age', 'city_id']] \
        .mean()
    publishers.columns = ['publisher_mean_' + x for x in publishers.columns]
    publishers['publisher'] = publishers.index
    publishers = publishers.reset_index(drop=True)

    publisher_unique_users = df \
        .groupby(['publisher'])['user_id'] \
        .apply(set) \
        .to_frame()

    publisher_unique_users['publisher_unique_users_cnt'] = \
        publisher_unique_users['user_id'].map(len)
    publisher_unique_users = publisher_unique_users.drop(columns='user_id')
    publisher_unique_users['publisher'] = publisher_unique_users.index
    publisher_unique_users = publisher_unique_users.reset_index(drop=True)

    return (users.merge(user_unique_publishers, on='user_id'), \
            publishers.merge(publisher_unique_users, on='publisher'))

def merge_history(data, users, publishers):
    df = data.copy()
    df['id'] = df.index

    # Временные признаки
    df['duration'] = df['hour_end'] - df['hour_start']
    df['publishers_size'] = df['publishers'].map(lambda x: len(x.split(',')))
    df['time_start'] = df['hour_start'].map(lambda x: x % 24)
    df['time_end'] = df['hour_end'].map(lambda x: x % 24)
    df['day_of_week_start'] = df['hour_start'].map(lambda x: (x // 24) % 7 + 1)
    df['day_of_week_end'] = df['hour_end'].map(lambda x: (x // 24) % 7 + 1)
    df['day_start_is_weekend'] = df['day_of_week_start'].map(lambda x: int(x in [6, 7]))
    df['day_end_is_weekend'] = df['day_of_week_end'].map(lambda x: int(x in [6, 7]))
    df['time_start_harmonic_0'] = make_harmonic_features(df['hour_start'])[0]
    df['time_start_harmonic_1'] = make_harmonic_features(df['hour_start'])[1]
    df['time_end_harmonic_0'] = make_harmonic_features(df['hour_end'])[0]
    df['time_end_harmonic_1'] = make_harmonic_features(df['hour_end'])[1]

    # Агрегация по пользователям
    df['user_ids'] = df['user_ids'].map(lambda row: [int(x) for x in row.split(',')])
    data_user_mean = df.explode('user_ids') \
        .merge(users, left_on='user_ids', right_on='user_id', how='left') \
        .groupby(['id']) \
        [['sex', 'age', 'city_id', 'user_mean_cpm',
          'user_mean_is_weekend', 'user_mean_time',
          'user_mean_time_harmonic_0', 'user_mean_time_harmonic_1',
          'user_mean_publisher', 'user_unique_publishers_cnt']] \
        .mean()

    data_user_mean['id'] = data_user_mean.index
    data_user_mean = data_user_mean.reset_index(drop=True)

    # Агрегация по площадкам
    df['publishers'] = df['publishers'].map(lambda row: [int(x) for x in row.split(',')])
    data_publisher_mean = df.explode('publishers') \
        .merge(publishers, left_on='publishers', right_on='publisher', how='left') \
        .groupby(['id']) \
        [['publisher_mean_time', 'publisher_mean_time_harmonic_0',
          'publisher_mean_time_harmonic_1', 'publisher_mean_is_weekend',
          'publisher_mean_cpm', 'publisher_mean_user_id', 'publisher_mean_sex',
          'publisher_mean_age', 'publisher_mean_city_id', 'publisher_unique_users_cnt']] \
        .mean()

    data_publisher_mean['id'] = data_publisher_mean.index
    data_publisher_mean = data_publisher_mean.reset_index(drop=True)

    return df.merge(data_user_mean, on='id').merge(data_publisher_mean, on='id')

def s_mape(y, y_hat, s=False):
    if s:
        return np.mean(200 * np.abs(y - y_hat) / (y + y_hat))
    else:
        return np.mean(100 * np.abs(y - y_hat) / np.abs(y))


def run_cv(X, ys, pars, models_path=None, cat_feature=None, num_folds=5, score=r2_score, box_cox_lmbda=-1.0, drop_noise=False):

    train_scores_hist = []
    test_scores_hist = []

    model = LGBMRegressor(**pars)

    folds = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    for i, (train_index, test_index) in enumerate(folds.split(X)):
        X_train, X_test = X[train_index], X[test_index]

        responses = pd.DataFrame()
        answers = pd.DataFrame()
        for j, y in enumerate(ys):
            y_train, y_test = y[train_index], y[test_index]

            if box_cox_lmbda > 0:
                y_train = box_cox(y_train, box_cox_lmbda)
                y_test = box_cox(y_test, box_cox_lmbda)

            model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)],
                      verbose=False,
                      early_stopping_rounds=100)

            if drop_noise:
                train_residual = np.abs(y_train.flatten() - y_train_hat)

                X_train = X_train[train_residual < 1.5]
                y_train = y_train[train_residual < 1.5]
                model.fit(X_train, y_train,
                          # cat_features=list(range(len(cat_features))),
                          eval_set=(X_test, y_test),
                          verbose=False)
                y_train_hat = model.predict(X_train)

            if models_path:
                model.booster_.save_model(models_path[j] + str(i) + '.txt')

            y_train_hat = model.predict(X_train)
            y_test_hat = model.predict(X_test)

            if box_cox_lmbda > 0:
                y_test_hat = anti_box_cox(y_test_hat, box_cox_lmbda)
                y_test = anti_box_cox(y_test, box_cox_lmbda)

            answers[str(j)] = y_test
            responses[str(j)] = y_test_hat

            """
                print('Test MAE: {} MAPE: {} SMAPE: {}' \
                  .format(mean_absolute_error(y_test, y_test_hat), \
                          s_mape(y_test, y_test_hat),
                          s_mape(y_test, y_test_hat, s=True))) 
                          """
        answers.columns = ['at_least_one', 'at_least_two', 'at_least_three']
        responses.columns = ['at_least_one', 'at_least_two', 'at_least_three']
        test_scores_hist.append(get_smoothed_mean_log_accuracy_ratio(responses, answers))

    return test_scores_hist, model

if __name__ == '__main__':

    models_path = ['./deploy/models/one/', './deploy/models/two/', './deploy/models/three/']

    validate_answers = pd.read_csv('./validate_answers.tsv', delimiter='\t')
    validate = pd.read_csv('./validate.tsv', delimiter='\t')
    users = pd.read_csv('./users.tsv', delimiter='\t')
    history = pd.read_csv('./history.tsv', delimiter='\t')

    (users_insights, publishers) = aggregate_history(history.merge(users))

    df = merge_history(validate, users.merge(users_insights, how='left'), publishers)

    features = ['duration', 'publishers_size', 'time_start',
                'time_end', 'day_of_week_start', 'day_of_week_end',
                'day_start_is_weekend', 'day_end_is_weekend', 'time_start_harmonic_0',
                'time_start_harmonic_1', 'time_end_harmonic_0', 'time_end_harmonic_1',
                'cpm', 'hour_start', 'hour_end', 'audience_size',
                'sex', 'age', 'city_id', 'user_mean_cpm',
                'user_mean_is_weekend', 'user_mean_time', 'user_mean_time_harmonic_0',
                'user_mean_time_harmonic_1', 'user_mean_publisher',
                'user_unique_publishers_cnt', 'publisher_mean_time',
                'publisher_mean_time_harmonic_0', 'publisher_mean_time_harmonic_1',
                'publisher_mean_is_weekend', 'publisher_mean_cpm',
                'publisher_mean_user_id', 'publisher_mean_sex', 'publisher_mean_age',
                'publisher_mean_city_id', 'publisher_unique_users_cnt']

    df = df[features]

    X, y1, y2, y3 = df.values, validate_answers.at_least_one.values, validate_answers.at_least_two.values, validate_answers.at_least_three.values

    pars = {'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31}

    history = []
    ls = [0.23]
    for lmbda in ls:
        test_scores_hist, model = run_cv(X, [y1, y2, y3], pars, box_cox_lmbda=lmbda, models_path=models_path)
        print('lmbda: {}\tscore: {} +/- {}'.format(lmbda, np.mean(test_scores_hist), np.std(test_scores_hist)))
        history.append(np.mean(test_scores_hist))

    """
    plt.figure()
    plt.plot(ls, history)
    plt.savefig('./box_cox2.png')
    """
    print(sorted([x for x in zip(features, model.feature_importances_)], key=lambda x: x[1], reverse=True))


