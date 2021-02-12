from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


def analyze_binary_classifier(model, X_test, y_test, model_classifier='Default'):
    y_pred = model.predict(X_test)
    y_pred = y_pred > 0.5
    print('Accuracy of {} classifier on test set: {:.2f}'.format(model_classifier, model.score(X_test, y_test)))

    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)
    print(classification_report(y_test, y_pred))

    logit_roc_auc = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='{} (area = {:.2f}) % '.format(model_classifier, logit_roc_auc))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig(model_classifier + ' Log_ROC')
    plt.show()


def plot_purchase_freq(data):
    sns.countplot(x='y', data=data, palette='hls')
    plt.savefig('count_plot')

    pd.crosstab(data['job'], data['y']).plot(kind='bar')
    plt.title('Purchase Frequency for Job Title')
    plt.xlabel('Job')
    plt.ylabel('Frequency of Purchase')
    plt.savefig('purchase_fre_job')

    pd.crosstab(data['marital'], data['y']).plot(kind='bar')
    plt.title('Purchase Frequency for Marital Title')
    plt.xlabel('Marital')
    plt.ylabel('Frequency of Purchase')
    plt.savefig('purchase_fre_marital')

    pd.crosstab(data['education'], data['y']).plot(kind='bar')
    plt.title('Purchase Frequency for Education Title')
    plt.xlabel('Education')
    plt.ylabel('Frequency of Purchase')
    plt.savefig('purchase_fre_education')

    pd.crosstab(data['day_of_week'], data['y']).plot(kind='bar')
    plt.title('Purchase Frequency for Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Frequency of Purchase')
    plt.savefig('pur_dayofweek_bar')

    pd.crosstab(data['month'], data['y']).plot(kind='bar')
    plt.title('Purchase Frequency for Month')
    plt.xlabel('Month')
    plt.ylabel('Frequency of Purchase')
    plt.savefig('pur_month_bar')

    pd.crosstab(data['poutcome'], data['y']).plot(kind='bar')
    plt.title('Purchase Frequency for previous campaign outcome')
    plt.xlabel('Previous outcome')
    plt.ylabel('Frequency of Purchase')
    plt.savefig('pur_poutcome_bar')

    table = pd.crosstab(data['job'], data['y'])
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Stacked Bar Chart of Marital Status vs Purchase')
    plt.xlabel('Job Status')
    plt.ylabel('Proportion of Customers')
    plt.savefig('job_vs_pur_stack')

    table = pd.crosstab(data['marital'], data['y'])
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Stacked Bar Chart of Marital Status vs Purchase')
    plt.xlabel('Marital Status')
    plt.ylabel('Proportion of Customers')
    plt.savefig('mariral_vs_pur_stack')

    table = pd.crosstab(data['education'], data['y'])
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Stacked Bar Chart of Education vs Purchase')
    plt.xlabel('Education Status')
    plt.ylabel('Proportion of Customers')
    plt.savefig('education_vs_pur_stack')

    table = pd.crosstab(data['day_of_week'], data['y'])
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Stacked Bar Chart of day of week vs Purchase')
    plt.xlabel('Day of week')
    plt.ylabel('Proportion of Customers')
    plt.savefig('day_of_week_vs_pur_stack')

    table = pd.crosstab(data['month'], data['y'])
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Stacked Bar Chart of month vs Purchase')
    plt.xlabel('Month')
    plt.ylabel('Proportion of Customers')
    plt.savefig('month_vs_pur_stack')

    table = pd.crosstab(data['poutcome'], data['y'])
    table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
    plt.title('Stacked Bar Chart of previous campaign outcome vs Purchase')
    plt.xlabel('Previous campaign outcome')
    plt.ylabel('Proportion of Customers')
    plt.savefig('poutcome_vs_pur_stack')

    plt.show()


def plot_params_dist(data):
    data.age.hist()
    plt.title('Histogram of Age')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.savefig('hist_age')

    plt.show()
