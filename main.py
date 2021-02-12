import pandas as pd
import matplotlib.pyplot as plt
plt.rc("font", size=14)

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from preprocesing import categorical_one_hot_encoding, group_education
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

from analysis import analyze_binary_classifier, plot_purchase_freq, plot_params_dist
from sklearn.ensemble import RandomForestClassifier


def prepare_data(data):
    data = group_education(data)
    encoded = categorical_one_hot_encoding(data, ['age', 'euribor3m', 'pdays', 'previous', 'duration'],
        ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome'])

    X = encoded
    y = data['y']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    os = SMOTE()
    os_data_X, os_data_y = os.fit_sample(X_train, y_train)
    return os_data_X, X_test, os_data_y, y_test


def find_features(X_train, y_train, columns, features_size = 20):
    model = LogisticRegression()

    rfe = RFE(model, n_features_to_select=features_size)
    rfe = rfe.fit(X_train, y_train.values.ravel())
    return columns.values[rfe.support_]


def fit_model(model, X_train, y_train, columns):
    X = X_train[columns]
    y = y_train

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model.fit(X_train, y_train)
    return X_test, y_test, model


if __name__ == '__main__':
    feature_limit = 20
    data = pd.read_csv('./data/banking.csv')
    data = data.dropna()

    # build data plots
    plot_purchase_freq(data)
    plot_params_dist(data)

    X_train, X_test, y_train, y_test = prepare_data(data)
    columns = X_train.columns

    cols = find_features(X_train, y_train, columns, feature_limit)

    # train and analyze different classifiers
    model = LogisticRegression()
    X_test, y_test, model = fit_model(model, X_train, y_train, cols)
    analyze_binary_classifier(model, X_test, y_test, model_classifier='LR {} f'.format(feature_limit))

    model = LogisticRegression()
    X_test, y_test, model = fit_model(model, X_train, y_train, columns)
    analyze_binary_classifier(model, X_test, y_test, model_classifier='LR {} f'.format(len(columns)))

    model = SVC(kernel='linear', probability=True)
    X_test, y_test, model = fit_model(model, X_train, y_train, cols)
    analyze_binary_classifier(model, X_test, y_test, model_classifier='SVC linear {} f'.format(feature_limit))

    model = SVC(probability=True)
    X_test, y_test, model = fit_model(model, X_train, y_train, cols)
    analyze_binary_classifier(model, X_test, y_test, model_classifier='SVC rbf {} f'.format(feature_limit))

    model = RandomForestClassifier(n_estimators=100, min_samples_split=2)
    X_test, y_test, model = fit_model(model, X_train, y_train, cols)
    analyze_binary_classifier(model, X_test, y_test, model_classifier='Fandom forest {} f'.format(feature_limit))

    model = RandomForestClassifier(n_estimators=100, min_samples_split=2)
    X_test, y_test, model = fit_model(model, X_train, y_train, columns)
    analyze_binary_classifier(model, X_test, y_test, model_classifier='Fandom forest {} f'.format(len(columns)))
