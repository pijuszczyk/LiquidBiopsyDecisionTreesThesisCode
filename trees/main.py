import datetime
import os
import time
from typing import Optional, Tuple, Iterable, Union

import re

# from cfec.explainers import Fimap
import json
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import random
import tqdm
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix, RocCurveDisplay, make_scorer
from sklearn.model_selection import StratifiedKFold, ParameterGrid
from xgboost import XGBClassifier, plot_importance, plot_tree

import data.KEGG_Tree as KEGG

LOGS_DIR = KEGG.k.CURRENT_DATASET_PATHS.LOGS_DIR_NAME
ALL_RESULT_DIR = os.path.join(LOGS_DIR, 'all_result')  # path where most results will be written
INITIAL_RESULT_DIR = os.path.join(LOGS_DIR, 'initial')  # path where selected results will be moved for further analysis
COMBINED_RESULT_DIR = os.path.join(LOGS_DIR, 'combined')  # path where further analyzed results will be written
MIN_N_BEST = 5
RANDOM_FOLD_SEED = 14 # 1, 5, 81
RANDOM_SAMPLE_SEED = 42#RANDOM_FOLD_SEED
N_JOBS = 7
N_FOLDS = 5


def main(reduced=True, col_group: Optional[str] = None):
    KEGG.k.print_dataset_paths_info('main.main')

    # mag_dane_wykresy()
    # mag_umap()

    # perform_exploration_training(reduced, col_group, 15)
    # perform_exploration_training_with_optuna(reduced, col_group, 15)
    # mag_pelny(reduced)
    # mag_grupy(col_group)
    # plot_single_groups_results_after_optuna()
    # plot_double_groups_results_after_optuna()
    # plot_importance_plots_from_initial()

    # plot_importances_for_best(reduced, col_group)
    # mag_importances_table()

    # perform_importances_k_fold_test_training_for_best(reduced, col_group, np.arange(1, 501, 1))
        # [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])#0.001, 0.005,

    # plot_roc_curve(reduced, col_group)


def mag_dane_wykresy():
    from sklearn.decomposition import PCA

    X_train, y_train, X_test, y_test = load_data(False, None)

    ress = []
    thrs = np.concatenate([np.arange(0.01, 1.0, 0.1), np.array([0.99999])])
    for thr in tqdm.tqdm(thrs):
        pca = PCA(thr)
        x_tr = pca.fit_transform(X_train)
        x_te = pca.transform(X_test)

        results = run_kfold_test_training(x_tr, x_te, y_train, y_test, [get_best_params()])
        results = results[['mean_train_bal_acc', 'mean_val_bal_acc', 'mean_test_bal_acc', 'mean_train_roc_auc',
                           'mean_val_roc_auc', 'mean_test_roc_auc', 'mean_fit_time', 'mean_test_predict_time']]
        n_comp = x_tr.shape[1]
        results['n_components'] = [n_comp]
        # results['mb'] = [688.75 * n_comp / (531 * 267)]
        results['threshold'] = [thr]
        ress.append(results)
    ress = pd.concat(ress, ignore_index=True)

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    print(ress)

    no_pca_res = run_kfold_test_training(X_train, X_test, y_train, y_test, [get_best_params()])
    no_pca_res = no_pca_res[['mean_train_bal_acc', 'mean_val_bal_acc', 'mean_test_bal_acc', 'mean_train_roc_auc',
                             'mean_val_roc_auc', 'mean_test_roc_auc', 'mean_fit_time', 'mean_test_predict_time']]
    assert X_train.shape[1] == 531 * 267
    no_pca_res['n_components'] = [X_train.shape[1]]
    # no_pca_res['mb'] = [688.75]

    print(no_pca_res)

    plt.figure()
    plt.plot(thrs, ress['n_components'])
    plt.xlabel("Wartość progu minimalnej wyjaśnianej wariancji")
    plt.ylabel("Liczba wygenerowanych cech")
    max_n_comp = int(ress['n_components'].tail(1))
    upper_margin = 1.1
    max_yaxis = int(np.ceil(max_n_comp * upper_margin / 100) * 100)
    plt.yticks(range(0, max_yaxis + 1, 100))
    plt.show()

    fig, ax1 = plt.subplots()
    plot1 = ax1.plot(thrs, ress['mean_fit_time'], color='blue', label='Trening')
    ax1.set_xlabel("Wartość progu minimalnej wyjaśnianej wariancji")
    ax1.set_ylabel("Średni czas treningu [s]")
    ax2 = ax1.twinx()
    plot2 = ax2.plot(thrs, ress['mean_test_predict_time'], color='red', label='Wnioskowanie')
    ax2.set_ylabel("Średni czas wnioskowania [s]")
    plots = plot1 + plot2
    labels = [p.get_label() for p in plots]
    plt.legend(plots, labels, loc=0)
    plt.show()

    plt.figure()
    plt.plot(thrs, ress['mean_train_bal_acc'], label='Zbiór treningowy')
    plt.plot(thrs, ress['mean_val_bal_acc'], label='Zbiór walidacyjny')
    plt.plot(thrs, ress['mean_test_bal_acc'], label='Zbiór testowy')
    plt.xlabel("Wartość progu minimalnej wyjaśnianej wariancji")
    plt.ylabel("Średnia zbilansowana dokładność na danym zbiorze")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(thrs, ress['mean_train_roc_auc'], label='Zbiór treningowy')
    plt.plot(thrs, ress['mean_val_roc_auc'], label='Zbiór walidacyjny')
    plt.plot(thrs, ress['mean_test_roc_auc'], label='Zbiór testowy')
    plt.xlabel("Wartość progu minimalnej wyjaśnianej wariancji")
    plt.ylabel("Średnia wartość ROC AUC na danym zbiorze")
    plt.legend()
    plt.show()


def mag_umap():
    import umap
    import umap.plot
    X_train, y_train, X_test, y_test = load_data(False, None)
    mapper = umap.UMAP(n_components=3).fit(np.concatenate([X_train, X_test]))
    umap.plot.points(mapper, labels=np.concatenate([y_train, y_test]))
    plt.show()


def print_data_stats(X_train, y_train, X_test, y_test):
    num_cancer = list(y_train).count(1)
    num_healthy = list(y_train).count(0)

    num_cancer_test = list(y_test).count(1)
    num_healthy_test = list(y_test).count(0)

    print(f"Total {len(y_train) + len(y_test)} samples in dataset")
    print(f"{num_healthy} healthy samples and {num_cancer} cancer samples in train dataset ({len(y_train)} total)")
    print(
        f"{num_healthy_test} healthy samples and {num_cancer_test} cancer samples in test dataset ({len(y_test)} total)")
    print(f"{X_train.shape[1]} columns")
    print('--------------------------------')


def create_logs_dirs_if_dont_exist():
    for d in [ALL_RESULT_DIR, INITIAL_RESULT_DIR, COMBINED_RESULT_DIR]:
        if not os.path.exists(d):
            os.makedirs(d)


def load_data(reduced: bool, col_group: Optional[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if reduced:
        X_train, y_train, X_test, y_test = KEGG.load_reduced_data(col_group)
    else:
        X_train, y_train, X_test, y_test = KEGG.load_original_data(col_group)  # TODO load 2D (from Tep)

    print_data_stats(X_train, y_train, X_test, y_test)

    return X_train, y_train, X_test, y_test


def get_const_params(train_y, early_stopping: bool):
    num_cancer = list(train_y).count(1)
    num_healthy = list(train_y).count(0)
    const_params = {'min_split_loss': 0, 'min_child_weight': 0, 'max_delta_step': 0,
                    'sampling_method': 'uniform', 'alpha': 1, 'lambda': 0, 'tree_method': 'auto',
                    'grow_policy': 'depthwise', 'objective': 'binary:logistic', 'predictor': 'auto',
                    'scale_pos_weight': num_healthy / num_cancer, 'eval_metric': 'logloss'}
    if early_stopping:
        const_params['early_stopping_rounds'] = 15
    return const_params


def select_max_random_params_configs(whole_grid, max_configs):
    params = list(ParameterGrid(whole_grid))
    configs_to_test = min(max_configs, len(params))
    random.seed(RANDOM_SAMPLE_SEED)
    print(f"Checking {configs_to_test} out of {len(params)} possible configurations "
          f"({round(configs_to_test / len(params) * 100, 2)}%):")
    params = random.sample(params, configs_to_test)
    for params_set in params:
        print(params_set)
    print()
    return params


def get_exploration_param_grid():
    return {
        'n_estimators': [900, 1000, 1100],
        'max_depth': [2, 3],
        'learning_rate': [0.1, 0.075, 0.05, 0.025, 0.01],
        'rate_drop': [None],
        'colsample_bytree': [0.9, 0.8, 0.7, 0.6, 0.5],
        'subsample': [0.9, 0.8, 0.7, 0.6, 0.5]
    }


def get_exploration_param_optuna_distributions(y_train):
    # based on examples in https://github.com/optuna/optuna-examples/blob/main
    distr = {
        "n_estimators": optuna.distributions.IntDistribution(700, 1200),
        # L2 regularization weight.
        "lambda": optuna.distributions.FloatDistribution(1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": optuna.distributions.FloatDistribution(1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": optuna.distributions.FloatDistribution(0.5, 1.0),
        # sampling according to each tree.
        "colsample_bytree": optuna.distributions.FloatDistribution(0.5, 1.0),
        # maximum depth of the tree, signifies complexity of the tree.
        "max_depth": optuna.distributions.IntDistribution(2, 7),
        # minimum child weight, larger the term more conservative the tree.
        "min_child_weight": optuna.distributions.IntDistribution(0, 5),
        "learning_rate": optuna.distributions.FloatDistribution(1e-4, 1.0, log=True),
        # defines how selective algorithm is.
        "gamma": optuna.distributions.FloatDistribution(1e-8, 1.0, log=True),
        "grow_policy": optuna.distributions.CategoricalDistribution(["depthwise", "lossguide"])
    }

    return distr


def get_start_string() -> str:
    start_date = datetime.datetime.now()
    return start_date.strftime("%d-%m-%Y_%H-%M-%S")


def perform_exploration_training(reduced: bool, col_group: Optional[str], max_configs_to_test: int = 100):
    start_str = get_start_string()

    X_train, y_train, X_test, y_test = load_data(reduced, col_group)

    param_grid = get_exploration_param_grid()
    params = select_max_random_params_configs(param_grid, max_configs_to_test)
    random.seed()
    df_result = run_kfold_test_training(X_train, X_test, y_train, y_test, params)
    result_filename = f"result_dataset_" \
                      f"{'reduced' if reduced else 'original'}_" \
                      f"group_{'all' if col_group is None else col_group}_" \
                      f"seed_{RANDOM_FOLD_SEED}_" \
                      f"{start_str}.csv"
    df_result.to_csv(os.path.join(ALL_RESULT_DIR, result_filename))
    print("Results saved to {}".format(result_filename))


def apply_pca(thr: float, X_train, X_test):
    from sklearn.decomposition import PCA
    p = PCA(thr)
    X_train = p.fit_transform(X_train)
    X_test = p.transform(X_test)
    print(f'Liczba cech po PCA: {X_train.shape[1]}')
    return X_train, X_test


def mag_pelny(reduced):
    start_str = get_start_string()

    X_train, y_train, X_test, y_test = load_data(reduced, None)

    results = []

    # params = [{'rate_drop': None, 'n_estimators': 1062, 'lambda': 0.01583130410006606, 'alpha': 2.9255911242767914e-07, 'subsample': 0.6521449210861506, 'colsample_bytree': 0.7037723536191082, 'max_depth': 3, 'min_child_weight': 2, 'learning_rate': 0.0832140184072569, 'gamma': 1.9842356667763806e-05, 'grow_policy': 'lossguide'}]
    # params = [{'rate_drop': None, 'n_estimators': 974, 'lambda': 0.0005019696201551437, 'alpha': 4.438919563834647e-06, 'subsample': 0.8033246441405956, 'colsample_bytree': 0.7594199892876792, 'max_depth': 4, 'min_child_weight': 1, 'learning_rate': 0.009055399498324119, 'gamma': 2.9006038846863782e-06, 'grow_policy': 'depthwise'}]
    # params = [{'rate_drop': None, 'n_estimators': 888, 'lambda': 2.0395159557795782e-08, 'alpha': 1.0123708651291661e-07, 'subsample': 0.8294883506155506, 'colsample_bytree': 0.9880548154171203, 'max_depth': 7, 'min_child_weight': 0, 'learning_rate': 0.012001753674392766, 'gamma': 0.46007733515734356, 'grow_policy': 'depthwise'}]
    params = [{'rate_drop': None, 'n_estimators': 1062, 'lambda': 0.01583130410006606, 'alpha': 2.9255911242767914e-07, 'subsample': 0.6521449210861506, 'colsample_bytree': 0.7037723536191082, 'max_depth': 3, 'min_child_weight': 2, 'learning_rate': 0.0832140184072569, 'gamma': 1.9842356667763806e-05, 'grow_policy': 'lossguide'}]
    for _ in range(3):
        random.seed()
        # X_train1, X_test1 = apply_pca(0.99999, X_train, X_test)
        df_result = run_kfold_test_training(X_train, X_test, y_train, y_test, params)

        clf = XGBClassifier(n_jobs=N_JOBS, use_label_encoder=False)
        clf.set_params(**get_const_params(y_train, False))
        clf.set_params(**params[0])
        clf.fit(X_train, y_train)
        pred_probs_test = clf.predict_proba(X_test)
        preds_test = np.argmax(pred_probs_test, 1)
        balacc = balanced_accuracy_score(y_test, preds_test)
        rocauc = roc_auc_score(y_test, pred_probs_test[:, 1])
        df_result['final_test_balacc'] = [balacc]
        df_result['final_test_rocauc'] = [rocauc]

        results.append(df_result)

    result = pd.concat(results, ignore_index=True)
    result_filename = f'after_optuna_{start_str}.csv'
    result.to_csv(os.path.join(ALL_RESULT_DIR, result_filename))
    mean_std_res = pd.DataFrame({'mean': result.mean(), 'std': result.std()})
    mean_std_filename = f'after_optuna_mean_std_{start_str}.csv'
    mean_std_res.to_csv(os.path.join(ALL_RESULT_DIR, mean_std_filename))


def mag_grupy(col_groups: str):
    start_str = get_start_string()

    X_train, y_train, X_test, y_test = load_data(True, col_groups)

    results = []

    best_params_filename_base = f'optuna_result_dataset_reduced_group_{col_groups}_seed_'
    matching_files = [name for name in os.listdir(ALL_RESULT_DIR) if name.startswith(best_params_filename_base)]
    assert len(matching_files) == 1
    best_params_df = pd.read_csv(os.path.join(ALL_RESULT_DIR, matching_files[0]))
    params = [json.loads(best_params_df['params'][0].replace('\'', '"'))]
    params[0]['rate_drop'] = None
    for _ in range(3):
        random.seed()
        df_result = run_kfold_test_training(X_train, X_test, y_train, y_test, params)

        clf = XGBClassifier(n_jobs=N_JOBS, use_label_encoder=False)
        clf.set_params(**get_const_params(y_train, False))
        clf.set_params(**params[0])
        clf.fit(X_train, y_train)
        pred_probs_test = clf.predict_proba(X_test)
        preds_test = np.argmax(pred_probs_test, 1)
        balacc = balanced_accuracy_score(y_test, preds_test)
        rocauc = roc_auc_score(y_test, pred_probs_test[:, 1])
        df_result['final_test_balacc'] = [balacc]
        df_result['final_test_rocauc'] = [rocauc]

        results.append(df_result)

    result = pd.concat(results, ignore_index=True)
    result_filename = f'after_optuna_groups_{col_groups}_{start_str}.csv'
    result.to_csv(os.path.join(ALL_RESULT_DIR, result_filename))
    mean_std_res = pd.DataFrame({'mean': result.mean(), 'std': result.std()})
    mean_std_filename = f'after_optuna_mean_std_groups_{col_groups}_{start_str}.csv'
    mean_std_res.to_csv(os.path.join(ALL_RESULT_DIR, mean_std_filename))


def plot_single_groups_results_after_optuna():
    base_filename = 'after_optuna_mean_std_groups_'
    files = [file for file in os.listdir(INITIAL_RESULT_DIR) if file.startswith(base_filename)]
    groups = [file.split('_')[5] for file in files]
    dfs = [pd.read_csv(os.path.join(INITIAL_RESULT_DIR, file)) for file in files]
    train_bal_accs = [df['mean'][df['Unnamed: 0'].to_list().index('mean_train_bal_acc')] for df in dfs]
    val_bal_accs = [df['mean'][df['Unnamed: 0'].to_list().index('mean_val_bal_acc')] for df in dfs]
    test_bal_accs = [df['mean'][df['Unnamed: 0'].to_list().index('mean_test_bal_acc')] for df in dfs]
    train_roc_aucs = [df['mean'][df['Unnamed: 0'].to_list().index('mean_train_roc_auc')] for df in dfs]
    val_roc_aucs = [df['mean'][df['Unnamed: 0'].to_list().index('mean_val_roc_auc')] for df in dfs]
    test_roc_aucs = [df['mean'][df['Unnamed: 0'].to_list().index('mean_test_roc_auc')] for df in dfs]

    fig = plt.figure()
    fig.set_size_inches(7, 5)
    plt.scatter(groups, train_bal_accs, label='Zbiór treningowy')
    plt.scatter(groups, val_bal_accs, label='Zbiór walidacyjny')
    plt.scatter(groups, test_bal_accs, label='Zbiór testowy')
    plt.legend()
    plt.xlabel('Grupa ścieżek')
    plt.ylabel('Średnia zbilansowana dokładność')
    plt.show()

    fig = plt.figure()
    fig.set_size_inches(7, 5)
    plt.scatter(groups, train_roc_aucs, label='Zbiór treningowy')
    plt.scatter(groups, val_roc_aucs, label='Zbiór walidacyjny')
    plt.scatter(groups, test_roc_aucs, label='Zbiór testowy')
    plt.legend()
    plt.xlabel('Grupa ścieżek')
    plt.ylabel('Średnie AUROC')
    plt.show()


def plot_double_groups_results_after_optuna():
    base_filename = 'after_optuna_mean_std_groups_'
    files = [file for file in os.listdir(INITIAL_RESULT_DIR) if file.startswith(base_filename)]
    groups = np.array([file.split('_')[5] for file in files])
    dfs = [pd.read_csv(os.path.join(INITIAL_RESULT_DIR, file)) for file in files]
    train_bal_accs = np.array([df['mean'][df['Unnamed: 0'].to_list().index('mean_train_bal_acc')] for df in dfs])
    val_bal_accs = np.array([df['mean'][df['Unnamed: 0'].to_list().index('mean_val_bal_acc')] for df in dfs])
    test_bal_accs = np.array([df['mean'][df['Unnamed: 0'].to_list().index('mean_test_bal_acc')] for df in dfs])
    train_roc_aucs = np.array([df['mean'][df['Unnamed: 0'].to_list().index('mean_train_roc_auc')] for df in dfs])
    val_roc_aucs = np.array([df['mean'][df['Unnamed: 0'].to_list().index('mean_val_roc_auc')] for df in dfs])
    test_roc_aucs = np.array([df['mean'][df['Unnamed: 0'].to_list().index('mean_test_roc_auc')] for df in dfs])

    best_5_ind = np.argsort(-test_bal_accs)[:6]
    worst_5_ind = np.argsort(-test_bal_accs)[-6:]
    chosen_ind = np.concatenate([best_5_ind, worst_5_ind])

    groups = groups[chosen_ind]

    fig = plt.figure()
    fig.set_size_inches(7, 5)
    plt.scatter(groups, train_bal_accs[chosen_ind], label='Zbiór treningowy')
    plt.scatter(groups, val_bal_accs[chosen_ind], label='Zbiór walidacyjny')
    plt.scatter(groups, test_bal_accs[chosen_ind], label='Zbiór testowy')
    plt.legend()
    plt.xlabel('Grupa ścieżek')
    plt.ylabel('Średnia zbilansowana dokładność')
    plt.show()

    fig = plt.figure()
    fig.set_size_inches(7, 5)
    plt.scatter(groups, train_roc_aucs[chosen_ind], label='Zbiór treningowy')
    plt.scatter(groups, val_roc_aucs[chosen_ind], label='Zbiór walidacyjny')
    plt.scatter(groups, test_roc_aucs[chosen_ind], label='Zbiór testowy')
    plt.legend()
    plt.xlabel('Grupa ścieżek')
    plt.ylabel('Średnie AUROC')
    plt.show()


def perform_exploration_training_with_optuna(reduced: bool, col_group: Optional[str], max_configs_to_test: int = 100):
    start_str = get_start_string()

    X_train, y_train, X_test, y_test = load_data(reduced, col_group)

    # X_train, X_test = apply_pca(0.99999, X_train, X_test)

    clf = XGBClassifier(use_label_encoder=False)
    clf.set_params(**get_const_params(y_train, False))
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_FOLD_SEED)
    distributions = get_exploration_param_optuna_distributions(y_train)
    scorer = make_scorer(balanced_accuracy_score)

    optuna_search = optuna.integration.OptunaSearchCV(
        clf, distributions, skf, scoring=scorer, n_jobs=N_JOBS, random_state=RANDOM_FOLD_SEED,
        n_trials=max_configs_to_test, verbose=2
    )

    optuna_search.fit(X_train, y_train)
    score = optuna_search.score(X_test, y_test)

    result = pd.DataFrame({'params': [optuna_search.best_params_], 'test_bal_acc': score})
    result_filename = f"optuna_result_dataset_" \
                      f"{'reduced' if reduced else 'original'}_" \
                      f"group_{'all' if col_group is None else col_group}_" \
                      f"seed_{RANDOM_FOLD_SEED}_" \
                      f"{start_str}.csv"
    result.to_csv(os.path.join(ALL_RESULT_DIR, result_filename))
    print("Results saved to {}".format(result_filename))


def get_best_params():
    return {'rate_drop': None, 'n_estimators': 1062, 'lambda': 0.01583130410006606, 'alpha': 2.9255911242767914e-07, 'subsample': 0.6521449210861506, 'colsample_bytree': 0.7037723536191082, 'max_depth': 3, 'min_child_weight': 2, 'learning_rate': 0.0832140184072569, 'gamma': 1.9842356667763806e-05, 'grow_policy': 'lossguide'}
    # return {
    #     'n_estimators': 1100,
    #     'max_depth': 3,
    #     'learning_rate': 0.05,
    #     'rate_drop': None,
    #     'colsample_bytree': 0.7,
    #     'subsample': 0.8,
    # }


def get_best_model(train_x, train_y):
    params = get_best_params()
    const_params = get_const_params(train_y, False)
    classifier = XGBClassifier(n_jobs=N_JOBS, use_label_encoder=False)
    classifier.set_params(**const_params)
    classifier.set_params(**params)
    classifier.fit(train_x, train_y, verbose=False)
    return classifier


def mag_importances_table():
    start_str = get_start_string()

    X_train, y_train, _, _ = load_data(False, None)
    classifier = get_best_model(X_train, y_train)

    max_feats = 30
    importances = classifier.feature_importances_
    best_ind = np.argsort(-importances)[:max_feats]

    y = best_ind // 531
    x = best_ind % 531

    group = []
    for i in range(max_feats):
        ok = False
        for g in KEGG.k.GROUPS_INFO:
            if y[i] < KEGG.k.GROUPS_INFO[g][0][1]:
                group.append(g)
                ok = True
                break
        if not ok:
            raise RuntimeError(f"Group for y={y} not existing")

    best_ind_names = [f'f{ind}' for ind in best_ind]
    wei_imp = classifier.get_booster().get_score(importance_type='weight', fmap='')
    gain_imp = classifier.get_booster().get_score(importance_type='gain', fmap='')
    cov_imp = classifier.get_booster().get_score(importance_type='cover', fmap='')
    wei_imp = [wei_imp[name] for name in best_ind_names]
    gain_imp = [gain_imp[name] for name in best_ind_names]
    cov_imp = [cov_imp[name] for name in best_ind_names]

    importances_pos_df = pd.DataFrame({'index': best_ind, 'y': y, 'x': x, 'group': group, 'imp_weight': wei_imp, 'imp_gain': gain_imp, 'imp_cover': cov_imp})
    importances_pos_path = os.path.join(ALL_RESULT_DIR, f"best_features_imps_{start_str}.csv")
    importances_pos_df.to_csv(importances_pos_path)


def plot_importances_for_best(reduced, col_group):
    start_str = get_start_string()

    X_train, y_train, _, _ = load_data(reduced, col_group)
    classifier = get_best_model(X_train, y_train)

    importances = classifier.feature_importances_
    best_ind = np.argsort(-importances)
    y = best_ind // 531
    x = best_ind % 531
    imp = importances[best_ind]
    importances_pos_df = pd.DataFrame({'y': y, 'x': x, 'importance': imp})
    importances_pos_path = os.path.join(ALL_RESULT_DIR, f"best_features_{start_str}.csv")
    importances_pos_df.to_csv(importances_pos_path)

    plot_importance(classifier, max_num_features=30, title='', xlabel='Istotność (zysk)', ylabel='Numer cechy', importance_type='gain')
    fig = plt.gcf()
    fig.set_size_inches(7, 7)
    plt.show()
    plot_tree(classifier)
    fig = plt.gcf()
    fig.set_size_inches(7, 7)
    plt.show()


def run_kfold_test_training(train_x, test_x, train_y, test_y, params_list):
    const_params = get_const_params(train_y, True)
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=None)

    train_bal_acc = []
    test_bal_acc = []
    val_bal_acc = []

    train_roc_auc = []
    test_roc_auc = []
    val_roc_auc = []

    train_conf_mtxs = []
    test_conf_mtxs = []
    val_conf_mtxs = []

    fit_time = []
    test_predict_time = []

    mean_train_bal_acc = []
    mean_test_bal_acc = []
    mean_val_bal_acc = []

    mean_train_roc_auc = []
    mean_test_roc_auc = []
    mean_val_roc_auc = []

    mean_fit_time = []
    mean_test_predict_time = []

    random.shuffle(params_list)

    progress = tqdm.tqdm(total=len(params_list)*skf.n_splits)
    for params in params_list:
        if params['rate_drop'] is None:
            del params['rate_drop']
        else:
            params['booster'] = 'dart'

        train_bal_acc.append([])
        test_bal_acc.append([])
        val_bal_acc.append([])

        train_roc_auc.append([])
        test_roc_auc.append([])
        val_roc_auc.append([])

        train_conf_mtxs.append([])
        test_conf_mtxs.append([])
        val_conf_mtxs.append([])

        fit_time.append([])
        test_predict_time.append([])

        for k, (train_index, val_index) in enumerate(skf.split(train_x, train_y)):
            X_train_fold = train_x[train_index]
            y_train_fold = train_y[train_index]
            X_val_fold = train_x[val_index]
            y_val_fold = train_y[val_index]

            classifier = XGBClassifier(n_jobs=N_JOBS, use_label_encoder=False)
            classifier.set_params(**const_params)
            classifier.set_params(**params)
            train_time = time.time()
            classifier.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
            train_time = time.time() - train_time

            test_time = time.time()
            pred_probs_test = classifier.predict_proba(test_x)
            test_time = time.time() - test_time

            pred_probs_train = classifier.predict_proba(X_train_fold)
            pred_probs_val = classifier.predict_proba(X_val_fold)

            preds_test = np.argmax(pred_probs_test, 1)
            preds_train = np.argmax(pred_probs_train, 1)
            preds_val = np.argmax(pred_probs_val, 1)

            train_bal_acc[-1].append(balanced_accuracy_score(y_train_fold, preds_train))
            test_bal_acc[-1].append(balanced_accuracy_score(test_y, preds_test))
            val_bal_acc[-1].append(balanced_accuracy_score(y_val_fold, preds_val))

            train_roc_auc[-1].append(roc_auc_score(y_train_fold, pred_probs_train[:, 1]))
            test_roc_auc[-1].append(roc_auc_score(test_y, pred_probs_test[:, 1]))
            val_roc_auc[-1].append(roc_auc_score(y_val_fold, pred_probs_val[:, 1]))

            train_conf_mtxs[-1].append(str(confusion_matrix(y_train_fold, preds_train)).replace('\n', ''))
            test_conf_mtxs[-1].append(str(confusion_matrix(test_y, preds_test)).replace('\n', ''))
            val_conf_mtxs[-1].append(str(confusion_matrix(y_val_fold, preds_val)).replace('\n', ''))

            fit_time[-1].append(train_time)
            test_predict_time[-1].append(test_time)

            progress.update()

        mean_train_bal_acc.append(np.mean(train_bal_acc[-1]))
        mean_test_bal_acc.append(np.mean(test_bal_acc[-1]))
        mean_val_bal_acc.append(np.mean(val_bal_acc[-1]))

        mean_train_roc_auc.append(np.mean(train_roc_auc[-1]))
        mean_test_roc_auc.append(np.mean(test_roc_auc[-1]))
        mean_val_roc_auc.append(np.mean(val_roc_auc[-1]))

        mean_fit_time.append(np.mean(fit_time[-1]))
        mean_test_predict_time.append(np.mean(test_predict_time[-1]))

        if 'rate_drop' not in params:
            params['rate_drop'] = None  # revert deletion, for cleaner results
        else:
            del params['booster']

        print('')
        print(f'Params: {params}')
        print(f'Mean train/test/val bal acc: {mean_train_bal_acc[-1]}/{mean_test_bal_acc[-1]}/{mean_val_bal_acc[-1]}')
        print(f'Mean train/test/val roc auc: {mean_train_roc_auc[-1]}/{mean_test_roc_auc[-1]}/{mean_val_roc_auc[-1]}')

    train_conf_mtxs = np.transpose(train_conf_mtxs)
    test_conf_mtxs = np.transpose(test_conf_mtxs)
    val_conf_mtxs = np.transpose(val_conf_mtxs)

    train_bal_acc = np.transpose(train_bal_acc)
    test_bal_acc = np.transpose(test_bal_acc)
    val_bal_acc = np.transpose(val_bal_acc)

    train_roc_auc = np.transpose(train_roc_auc)
    test_roc_auc = np.transpose(test_roc_auc)
    val_roc_auc = np.transpose(val_roc_auc)

    best_ind = np.argsort(mean_val_bal_acc)[-1]
    print('')
    print('Best:')
    print(f'Params: {params_list[best_ind]}')
    print(f'Mean train/test/val bal acc: {mean_train_bal_acc[best_ind]}/{mean_test_bal_acc[best_ind]}/{mean_val_bal_acc[best_ind]}')
    print(f'Mean train/test/val roc auc: {mean_train_roc_auc[best_ind]}/{mean_test_roc_auc[best_ind]}/{mean_val_roc_auc[best_ind]}')

    result = {'params': params_list,
              'train_conf_mtx_fold_0': train_conf_mtxs[0],
              'train_conf_mtx_fold_1': train_conf_mtxs[1],
              'train_conf_mtx_fold_2': train_conf_mtxs[2],
              'train_conf_mtx_fold_3': train_conf_mtxs[3],
              'train_conf_mtx_fold_4': train_conf_mtxs[4],
              'test_conf_mtx_fold_0': test_conf_mtxs[0],
              'test_conf_mtx_fold_1': test_conf_mtxs[1],
              'test_conf_mtx_fold_2': test_conf_mtxs[2],
              'test_conf_mtx_fold_3': test_conf_mtxs[3],
              'test_conf_mtx_fold_4': test_conf_mtxs[4],
              'val_conf_mtx_fold_0': val_conf_mtxs[0],
              'val_conf_mtx_fold_1': val_conf_mtxs[1],
              'val_conf_mtx_fold_2': val_conf_mtxs[2],
              'val_conf_mtx_fold_3': val_conf_mtxs[3],
              'val_conf_mtx_fold_4': val_conf_mtxs[4],
              'train_bal_acc_fold_0': train_bal_acc[0],
              'train_bal_acc_fold_1': train_bal_acc[1],
              'train_bal_acc_fold_2': train_bal_acc[2],
              'train_bal_acc_fold_3': train_bal_acc[3],
              'train_bal_acc_fold_4': train_bal_acc[4],
              'test_bal_acc_fold_0': test_bal_acc[0],
              'test_bal_acc_fold_1': test_bal_acc[1],
              'test_bal_acc_fold_2': test_bal_acc[2],
              'test_bal_acc_fold_3': test_bal_acc[3],
              'test_bal_acc_fold_4': test_bal_acc[4],
              'val_bal_acc_fold_0': val_bal_acc[0],
              'val_bal_acc_fold_1': val_bal_acc[1],
              'val_bal_acc_fold_2': val_bal_acc[2],
              'val_bal_acc_fold_3': val_bal_acc[3],
              'val_bal_acc_fold_4': val_bal_acc[4],
              'train_roc_auc_fold_0': train_roc_auc[0],
              'train_roc_auc_fold_1': train_roc_auc[1],
              'train_roc_auc_fold_2': train_roc_auc[2],
              'train_roc_auc_fold_3': train_roc_auc[3],
              'train_roc_auc_fold_4': train_roc_auc[4],
              'test_roc_auc_fold_0': test_roc_auc[0],
              'test_roc_auc_fold_1': test_roc_auc[1],
              'test_roc_auc_fold_2': test_roc_auc[2],
              'test_roc_auc_fold_3': test_roc_auc[3],
              'test_roc_auc_fold_4': test_roc_auc[4],
              'val_roc_auc_fold_0': val_roc_auc[0],
              'val_roc_auc_fold_1': val_roc_auc[1],
              'val_roc_auc_fold_2': val_roc_auc[2],
              'val_roc_auc_fold_3': val_roc_auc[3],
              'val_roc_auc_fold_4': val_roc_auc[4],
              'mean_train_bal_acc': mean_train_bal_acc,
              'mean_test_bal_acc': mean_test_bal_acc,
              'mean_val_bal_acc': mean_val_bal_acc,
              'mean_train_roc_auc': mean_train_roc_auc,
              'mean_test_roc_auc': mean_test_roc_auc,
              'mean_val_roc_auc': mean_val_roc_auc,
              'mean_fit_time': mean_fit_time,
              'mean_test_predict_time': mean_test_predict_time
              }
    return pd.DataFrame(result)


def perform_importances_k_fold_test_training_for_best(reduced: bool, col_group: Optional[str], importance_thresholds: Iterable[Union[int, float]]):
    start_str = get_start_string()

    X_train, y_train, X_test, y_test = load_data(reduced, col_group)

    print(f'Determining feature importances through best overall model')
    model = get_best_model(X_train, y_train)
    importances = model.feature_importances_

    param_grid = get_exploration_param_grid()
    params = select_max_random_params_configs(param_grid, 100)

    for threshold in importance_thresholds:
        print(f'Checking threshold {threshold}')

        perc_threshold, feat_num_threshold = (threshold, None) if type(threshold) is float else (None, threshold)
        imp_X_train = KEGG.extract_most_important_features(X_train, importances, perc_threshold, feat_num_threshold)
        imp_X_test = KEGG.extract_most_important_features(X_test, importances, perc_threshold, feat_num_threshold)

        df_result = run_kfold_test_training(imp_X_train, imp_X_test, y_train, y_test, params)
        random.seed()
        result_filename = f"importance_training_result_dataset_" \
                          f"{'reduced' if reduced else 'original'}_" \
                          f"group_{'all' if col_group is None else col_group}_" \
                          f"seed_{RANDOM_FOLD_SEED}_" \
                          f"importance_{threshold}_" \
                          f"{start_str}.csv"
        df_result.to_csv(os.path.join(ALL_RESULT_DIR, result_filename))


def get_n_best_from_file(file, N) -> pd.DataFrame:
    result = pd.read_csv(os.path.join(INITIAL_RESULT_DIR, file))
    n_best = result.sort_values(by=['mean_val_bal_acc', 'mean_test_bal_acc'], ascending=[False, False]).head(N)
    summarized = n_best.loc[:, ['params', 'mean_train_bal_acc', 'mean_test_bal_acc', 'mean_val_bal_acc',
                                'mean_train_roc_auc', 'mean_test_roc_auc', 'mean_val_roc_auc']]
    return summarized


def plot_min_max_median_importance_plots_from_initial(number_of_configs):
    x = []
    val_mins, val_maxs, val_meds = [], [], []
    test_mins, test_maxs, test_meds = [], [], []
    for file in os.listdir(INITIAL_RESULT_DIR):
        summarized = get_n_best_from_file(file, number_of_configs)
        x.append(int(file.split('_')[10]))
        val = summarized.loc[:, 'mean_val_bal_acc'].to_numpy()
        test = summarized.loc[:, 'mean_test_bal_acc'].to_numpy()
        val_mins.append(min(val))
        test_mins.append(min(test))
        val_maxs.append(max(val))
        test_maxs.append(max(test))
        val_meds.append(np.median(val))
        test_meds.append(np.median(test))

    order = np.argsort(x)
    x = np.array(x)[order]
    val_mins = np.array(val_mins)[order]
    val_maxs = np.array(val_maxs)[order]
    val_meds = np.array(val_meds)[order]
    test_mins = np.array(test_mins)[order]
    test_maxs = np.array(test_maxs)[order]
    test_meds = np.array(test_meds)[order]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.plot(x, val_mins, label='Minimum validation accuracy', color='purple')
    plt.plot(x, val_maxs, label='Maximum validation accuracy', color='blue')
    plt.fill_between(x, val_mins, val_maxs, alpha=0.2)
    plt.plot(x, val_meds, label='Median validation accuracy', color='green')
    plt.xlabel('Number of most important features used')
    plt.title('Features threshold vs validation accuracy')
    plt.legend()
    plt.ylim([0.8, 1.0])
    plt.savefig(os.path.join(ALL_RESULT_DIR, 'importances_val_accuracy.png'))

    plt.figure(figsize=(8, 8))
    plt.plot(x, test_mins, label='Minimum test accuracy', color='purple')
    plt.plot(x, test_maxs, label='Maximum test accuracy', color='blue')
    plt.fill_between(x, test_mins, test_maxs, alpha=0.2)
    plt.plot(x, test_meds, label='Median test accuracy', color='green')
    plt.xlabel('Number of most important features used')
    plt.title('Features threshold vs test accuracy')
    plt.legend()
    plt.ylim([0.8, 1.0])
    plt.savefig(os.path.join(ALL_RESULT_DIR, 'importances_test_accuracy.png'))


def plot_importance_plots_from_initial():
    x = []
    train_bal_acc = []
    val_bal_acc = []
    test_bal_acc = []
    train_roc_auc = []
    val_roc_auc = []
    test_roc_auc = []

    for file in os.listdir(INITIAL_RESULT_DIR):
        df = pd.read_csv(os.path.join(INITIAL_RESULT_DIR, file))

        train_bal_acc.append(df['mean'][df['Unnamed: 0'].to_list().index('mean_train_bal_acc')])
        val_bal_acc.append(df['mean'][df['Unnamed: 0'].to_list().index('mean_val_bal_acc')])
        test_bal_acc.append(df['mean'][df['Unnamed: 0'].to_list().index('mean_test_bal_acc')])
        train_roc_auc.append(df['mean'][df['Unnamed: 0'].to_list().index('mean_train_roc_auc')])
        val_roc_auc.append(df['mean'][df['Unnamed: 0'].to_list().index('mean_val_roc_auc')])
        test_roc_auc.append(df['mean'][df['Unnamed: 0'].to_list().index('mean_test_roc_auc')])

        x.append(int(file.split('_')[5]))

    order = np.argsort(x)
    x = np.array(x)[order]
    train_bal_acc = np.array(train_bal_acc)[order]
    val_bal_acc = np.array(val_bal_acc)[order]
    test_bal_acc = np.array(test_bal_acc)[order]
    train_roc_auc = np.array(train_roc_auc)[order]
    val_roc_auc = np.array(val_roc_auc)[order]
    test_roc_auc = np.array(test_roc_auc)[order]

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.plot(x, train_bal_acc, label='Zbiór treningowy')
    plt.plot(x, val_bal_acc, label='Zbiór walidacyjny')
    plt.plot(x, test_bal_acc, label='Zbiór testowy')
    plt.xlabel('Liczba użytych najbardziej istotnych cech')
    plt.ylabel('Zbilansowana dokładność')
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 8))
    plt.plot(x, train_roc_auc, label='Zbiór treningowy')
    plt.plot(x, val_roc_auc, label='Zbiór walidacyjny')
    plt.plot(x, test_roc_auc, label='Zbiór testowy')
    plt.xlabel('Liczba użytych najbardziej istotnych cech')
    plt.ylabel('AUROC')
    plt.legend()
    plt.show()



def extract_combined_features_num_vs_acc():
    p = os.path.join(COMBINED_RESULT_DIR)
    combined = pd.read_csv(os.path.join(p, sorted(os.listdir(p))[-1]))
    extracted = combined.loc[:, ['mean_train_bal_acc', 'mean_test_bal_acc', 'mean_val_bal_acc', 'mean_train_roc_auc',
                                 'mean_test_roc_auc', 'mean_val_roc_auc', 'file']]
    extracted['features_num'] = extracted['file'].apply(lambda filename: int(filename.split('_')[10]))
    del extracted['file']
    extracted = extracted.sort_values(by='features_num')

    extr_date = datetime.datetime.now()
    extr_str = extr_date.strftime("%d-%m-%Y_%H-%M-%S")

    extracted.to_csv(os.path.join(COMBINED_RESULT_DIR, f'top_1_importances_{extr_str}.csv'))


def combine_results():
    all_summarized = []
    for file in os.listdir(INITIAL_RESULT_DIR):
        summarized = get_n_best_from_file(file, 1)
        summarized['file'] = file
        all_summarized.append(summarized)

    combined = pd.concat(all_summarized, ignore_index=True)

    combine_date = datetime.datetime.now()
    combine_str = combine_date.strftime("%d-%m-%Y_%H-%M-%S")

    combined.to_csv(os.path.join(COMBINED_RESULT_DIR, f'top_1_{combine_str}.csv'))


def combine_results_from_different_runs_on_same_config():
    """
    Requires results from a few runs that have different FOLD_SEED (related to division into folds) but the same
    SAMPLE_SEED (related to sampling hyperparameters configs) and configs list.
    """
    all_summarized = None
    for i, file in enumerate(os.listdir(INITIAL_RESULT_DIR)):
        result = pd.read_csv(os.path.join(INITIAL_RESULT_DIR, file))
        summarized = result.loc[:, ['params', 'mean_train_bal_acc', 'mean_test_bal_acc', 'mean_val_bal_acc',
                                    'mean_train_roc_auc', 'mean_test_roc_auc', 'mean_val_roc_auc']]
        summarized = summarized.sort_values(by='params')
        summarized.columns = ['params', f'mean_train_bal_acc{i}', f'mean_test_bal_acc{i}', f'mean_val_bal_acc{i}',
                              f'mean_train_roc_auc{i}', f'mean_test_roc_auc{i}', f'mean_val_roc_auc{i}']
        if all_summarized is None:
            all_summarized = summarized
        else:
            all_summarized = all_summarized.merge(summarized, on='params')

    train_bal_accs = all_summarized.loc[:, [col for col in all_summarized.columns if col.startswith('mean_train_bal_acc')]]
    test_bal_accs = all_summarized.loc[:, [col for col in all_summarized.columns if col.startswith('mean_test_bal_acc')]]
    val_bal_accs = all_summarized.loc[:, [col for col in all_summarized.columns if col.startswith('mean_val_bal_acc')]]

    train_roc_aucs = all_summarized.loc[:, [col for col in all_summarized.columns if col.startswith('mean_train_roc_auc')]]
    test_roc_aucs = all_summarized.loc[:, [col for col in all_summarized.columns if col.startswith('mean_test_roc_auc')]]
    val_roc_aucs = all_summarized.loc[:, [col for col in all_summarized.columns if col.startswith('mean_val_roc_auc')]]

    all_summarized['mean_train_bal_acc_mean'] = train_bal_accs.mean(1)
    all_summarized['mean_test_bal_acc_mean'] = test_bal_accs.mean(1)
    all_summarized['mean_val_bal_acc_mean'] = val_bal_accs.mean(1)

    all_summarized['mean_train_roc_auc_mean'] = train_roc_aucs.mean(1)
    all_summarized['mean_test_roc_auc_mean'] = test_roc_aucs.mean(1)
    all_summarized['mean_val_roc_auc_mean'] = val_roc_aucs.mean(1)

    all_summarized['mean_train_bal_acc_std'] = train_bal_accs.std(1)
    all_summarized['mean_test_bal_acc_std'] = test_bal_accs.std(1)
    all_summarized['mean_val_bal_acc_std'] = val_bal_accs.std(1)

    all_summarized['mean_train_roc_auc_std'] = train_roc_aucs.std(1)
    all_summarized['mean_test_roc_auc_std'] = test_roc_aucs.std(1)
    all_summarized['mean_val_roc_auc_std'] = val_roc_aucs.std(1)

    combine_date = datetime.datetime.now()
    combine_str = combine_date.strftime("%d-%m-%Y_%H-%M-%S")

    all_summarized.to_csv(os.path.join(COMBINED_RESULT_DIR, f'combined_{combine_str}.csv'))


def extract_combined_results_based_on_group(N):
    """
    Requires initial_result dir to contain only the files to be merged.
    Please copy the relevant files from all_result dir
    """
    regex = re.compile(r'.+group_([\d.,al]).+')
    processed_times = {}
    all_summarized = []
    for file in os.listdir(INITIAL_RESULT_DIR):
        matches = regex.search(file)
        group = matches.group(1)
        summarized = get_n_best_from_file(file, N)

        summarized['group'] = [group] * N

        if group not in processed_times:
            processed_times[group] = 1
        else:
            processed_times[group] += 1

        summarized['attempt'] = processed_times[group]

        all_summarized.append(summarized)
    combined = pd.concat(all_summarized, ignore_index=True)

    means = combined.groupby(by=['params', 'group']).mean()
    stds = combined.groupby(by=['params', 'group']).std()

    combine_date = datetime.datetime.now()
    combine_str = combine_date.strftime("%d-%m-%Y_%H-%M-%S")

    combined.to_csv(os.path.join(COMBINED_RESULT_DIR, f'top_{N}_{combine_str}.csv'))
    means.to_csv(os.path.join(COMBINED_RESULT_DIR, f'means_{combine_str}.csv'))
    stds.to_csv(os.path.join(COMBINED_RESULT_DIR, f'stds_{combine_str}.csv'))


def plot_accuracy_for_single_groups(file: str):
    plot_date = datetime.datetime.now()
    plot_str = plot_date.strftime("%d-%m-%Y_%H-%M-%S")

    d = pd.read_csv(file)
    test = d['mean_test_score'].to_numpy()
    val = d['mean_val_score'].to_numpy()
    group = d['group'].to_numpy()

    plt.figure()
    plt.scatter(group, val, label='Validation set', marker='+', color='blue')
    plt.scatter(group, test, label='Testing set', marker='*', color='red')
    plt.xlabel('Group number')
    plt.ylabel('Balanced accuracy [%]')
    plt.title('Balanced accuracy for single groups')
    plt.legend()
    plt.savefig(os.path.join(ALL_RESULT_DIR, f'accuracy_for_groups_{plot_str}.png'))


def plot_roc_curve(reduced, col_group):
    start_str = get_start_string()

    X_train, y_train, X_test, y_test = load_data(reduced, col_group)
    classifier = get_best_model(X_train, y_train)
    RocCurveDisplay.from_estimator(classifier, X_test, y_test).plot()
    # plt.show()
    plt.savefig(os.path.join(ALL_RESULT_DIR, f'roc_curve_'
                                             f'dataset_{"reduced" if reduced else "original"}_'
                                             f'group_{col_group}_'
                                             f'{start_str}.png'))



# def fit_augmentation_model(reduced, col_group):
#     start_str = get_start_string()
#
#     X_train, y_train, X_test, y_test = load_data(reduced, col_group)




# def generate_cf(reduced, col_group):
#     X_train, y_train, X_test, y_test = load_data(reduced, col_group)
#     classifier = get_best_model(X_train, y_train)
#     y_train_pred = classifier.predict(X_train)
#
#     fimap = Fimap()
#     fimap.fit(pd.DataFrame(X_train), y_train_pred)
#     x = X_test[0]
#     cf.generate(x)


if __name__ == "__main__":
    create_logs_dirs_if_dont_exist()

    # done = []
    # for group_a in KEGG.k.GROUPS_INFO:
    #     for group_b in KEGG.k.GROUPS_INFO:
    #         if group_a != group_b and (group_b, group_a) not in done:
    #             main(reduced=True, col_group=f'{group_a},{group_b}')
    #             done.append((group_a, group_b))

    # for col_group in KEGG.k.GROUPS_INFO:
    #     main(reduced=True, col_group=col_group)

    # for _ in range(5):
    #     main(reduced=False, col_group=0.1)

    main(reduced=True)

    # combine_results()
    # combine_results_from_different_runs_on_same_config()
    # plot_min_max_median_importance_plots_from_initial(100)
    # extract_combined_features_num_vs_acc()
    # extract_combined_results_based_on_group(1)

    # generate_cf(True, None)
