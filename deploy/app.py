from utils import get_history_users_features, get_test_features, make_prediction
import sys
import pandas as pd
import lightgbm as lgb
import os

def load_estimators(paths):
    estimators = []
    for path in paths:
        estimators.append([lgb.Booster(model_file=path + model) for model in os.listdir(path)])

    return estimators

def main():
    models_path = ['./models/one/', './models/two/', './models/three/']
    users_path = './data/users.tsv'
    history_path = './data/history.tsv'
    test = pd.read_csv(sys.argv[1], sep="\t")
    users = pd.read_csv(users_path, delimiter='\t')
    history = pd.read_csv(history_path, delimiter='\t')


    estimators = load_estimators(models_path)
    lmbda = 0.23

    (users_features, publishers_features) = get_history_users_features(history.merge(users))
    df = get_test_features(test, users.merge(users_features, how='left'), publishers_features)
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

    prediction = make_prediction(df[features].values, estimators, box_cox_lmbda=lmbda)
    prediction.to_csv(sys.stdout, sep="\t", index=False, header=True)


if __name__ == '__main__':
    main()
