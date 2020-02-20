import numpy as np
import pandas as pd


def anti_box_cox(x, lmbda):
    x = lmbda * x + 1
    return np.sign(x) * np.abs(x) ** (1 / lmbda)


def make_harmonic_features(value, period=24):
    value *= 2 * np.pi / period
    return np.cos(value), np.sin(value)


def get_history_users_features(history_merged_users):
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
        [['time', 'time_harmonic_0', 'time_harmonic_1', 'is_weekend',
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

    return (users.merge(user_unique_publishers, on='user_id'),
            publishers.merge(publisher_unique_users, on='publisher'))


def get_test_features(data, users, publishers):
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

    return df \
        .merge(data_user_mean, on='id') \
        .merge(data_publisher_mean, on='id')


def make_prediction(X_test, estimators, box_cox_lmbda=-1):
    prediction = pd.DataFrame()

    for i, estimator in enumerate(estimators):
        y_test_hat = np.zeros(X_test.shape[0])

        for model in estimator:
            predict = model.predict(X_test)

            if box_cox_lmbda > 0:
                predict = anti_box_cox(predict, box_cox_lmbda)

            y_test_hat += predict

        prediction[str(i)] = np.clip(y_test_hat / len(estimator), 0, 1)

    prediction.columns = ['at_least_one', 'at_least_two', 'at_least_three']
    return prediction
