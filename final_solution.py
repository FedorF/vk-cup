from run_experiment import execute_experiment
import pandas as pd


def make_submit(n_models, sub_path, submit_path):
    pred_df = pd.DataFrame()
    for i in range(n_models):
        models_params = {'n_estimators': 203,
                         'boosting_type': 'gbdt',
                         'class_weight': None,
                         'colsample_bytree': 0.3811304824216669,
                         'importance_type': 'split',
                         'learning_rate': 0.016659893432998896,
                         'max_depth': 20,
                         'metrics': 'AUC',
                         'min_child_samples': 12,
                         'min_child_weight': 0.0036350360157623163,
                         'min_split_gain': 0.9619961904304549,
                         'n_jobs': -1,
                         'num_leaves': 32,
                         'objective': 'binary',
                         'reg_alpha': 0.8022751137716411,
                         'reg_lambda': 0.8186913351852023,
                         'silent': True,
                         'subsample': 0.5967826661188248,
                         'subsample_for_bin': 200000,
                         'subsample_freq': 14,
                         "random_state": i}

        encoders_list = [
            ("FrequencyEncoder",),
            ("MEstimateEncoder",),
            ("TargetEncoder",),
            ("WOEEncoder",),
            ("JamesSteinEncoder",),
            ("OrdinalEncoder",),
            ("CatBoostEncoder",)
        ]

        validation_type = "Single"
        experiment_description = f"Check single encoder, {validation_type} validation"

        y_hat = execute_experiment(encoders_list, validation_type, experiment_description, models_params)
        pred_df[str(i)] = y_hat

        if (i % 2 == 0):
            print('step: {}'.format(i))

    submit = pd.read_csv(sub_path)
    submit['proba'] = pred_df.mean(axis=1)
    pred_df.to_csv(submit_path, index=False)


if __name__ == '__main__':
    n_models = 2
    sub_sample_path = './data/onetwotrip_challenge_sub1.csv'
    submit_path = './submits/blending10.csv'
    make_submit(n_models, sub_sample_path, submit_path)
