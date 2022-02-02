import statistics
import time
import shap
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

import plotly.graph_objects as go

from fairness_utils import FairnessUtils

np.random.seed(int(time.time()))


def create_dataframe(x, y_gold, base_columns, y_pred=None):
    if y_pred is not None:
        data_array = np.concatenate((x, np.array([y_gold]).T, np.array([y_pred]).T), axis=1)
    else:
        data_array = np.concatenate((x, np.array([y_gold]).T), axis=1)
    columns = list(base_columns).copy()
    columns.append('target')
    if y_pred is not None:
        columns.append('target_pred')
    df = pd.DataFrame(data=data_array, columns=columns)
    return df


def add_priv_class_col(df, column, condition):
    # Adding a priv_clas
    df['priv_class'] = df[column].map(eval('lambda x:' + condition))
    return df


def recalculate_from_demographic_parity(df):
    def change_f(delta):
        change_f = [0.0, 1.0] if delta < 0 else [1.0, 0.0]
        nb_priv_false = len(df[df['priv_class'] == False])
        # change F
        diff_nb_f = delta * nb_priv_false
        int_diff_nb_f = int(diff_nb_f)
        df_f_condition = df[(df['target_pred'] == change_f[0]) & (df['priv_class'] == False)]
        nb_f = min(abs(int_diff_nb_f), len(df_f_condition))
        st.text('  F ' + str(change_f[0]) + ' -> ' + str(change_f[1]) + ' to change ' + str(nb_f) + '/' + str(
            len(df_f_condition)))
        random_df_f_condition = df_f_condition.sample(n=abs(nb_f))
        for idx in random_df_f_condition.index:
            df.iloc[idx, df.columns.get_loc('target_pred')] = change_f[1]

    def change_t(delta):
        change_t = [1.0, 0.0] if delta < 0 else [0.0, 1.0]
        nb_priv_true = len(df[df['priv_class'] == True])
        # change T
        diff_nb_t = - delta * nb_priv_true
        df_t_condition = df[(df['target_pred'] == change_t[0]) & (df['priv_class'] == True)]
        int_diff_t = int(diff_nb_t)
        nb_t = min(abs(int_diff_t), len(df_t_condition))
        st.text('  T ' + str(change_t[0]) + ' -> ' + str(change_t[1]) + ' to change ' + str(nb_t) + '/' + str(
            len(df_t_condition)))
        random_df_t_1_condition = df_t_condition.sample(n=abs(nb_t))
        for idx in random_df_t_1_condition.index:
            df.iloc[idx, df.columns.get_loc('target_pred')] = change_t[1]

    st.write('--- Recalculate from demographic parity ---')
    delta_dp = old_demographic_parity - wanted_demographic_parity
    delta_dp_f = delta_dp * correction_priority
    delta_dp_t = delta_dp * (1 - correction_priority)
    st.text('  delta_dp_f ' + str(delta_dp_f))
    st.text('  delta_dp_t ' + str(delta_dp_t))

    change_f(delta_dp_f)
    change_t(delta_dp_t)

    new_dp = FairnessUtils.demographic_parity(df)
    delta_dp = new_dp - wanted_demographic_parity
    st.text('  After first changes, diff ' + str(delta_dp))

    # T is entirely used -> change F
    change_f(delta_dp)
    new_dp = FairnessUtils.demographic_parity(df)
    delta_dp = new_dp - wanted_demographic_parity
    st.text('  After change F, diff ' + str(delta_dp))
    change_t(delta_dp)
    new_dp = FairnessUtils.demographic_parity(df)
    delta_dp = new_dp - wanted_demographic_parity
    st.text('  After change T, diff ' + str(delta_dp))

    return df


def recalculate_from_demographic_parity_multiple(df, n=100):
    all_df = []
    for i in range(n):
        temp_df = recalculate_from_demographic_parity(df)
        all_df.append(temp_df.copy())
    return all_df


def calculate_metrics_and_errors(dfs, func_metrics):
    metrics = []
    for d in dfs:
        m = func_metrics(d)
        if m is not None:
            metrics.append(m)
    mean = statistics.mean(metrics)
    std_error = statistics.pstdev(metrics)
    conf_interval = 1.96 * std_error

    return mean, std_error, conf_interval


def calculate_accuracy(df):
    return sklearn.metrics.accuracy_score(y_true=df['target'].tolist(),
                                          y_pred=df['target_pred'].tolist())


def create_figure(data, metrics):
    fig = go.Figure()

    # Add graph data
    olds = [d[0] for d in data]
    wanteds = [d[1] for d in data]
    news = [d[2] for d in data]
    news_error = [d[3] for d in data]

    # Make traces for graph
    trace1 = go.Bar(x=metrics, y=olds, xaxis='x2', yaxis='y2',
                    name='Before')
    trace2 = go.Bar(x=metrics, y=wanteds, xaxis='x2', yaxis='y2',
                    name='Wanted')
    trace3 = go.Bar(x=metrics, y=news, error_y=dict(type='data', array=news_error), xaxis='x2', yaxis='y2',
                    name='After')

    # Add trace data to figure
    fig.add_traces([trace1, trace2, trace3])
    return fig


x, y = shap.datasets.adult()
x_display, y_display = shap.datasets.adult(display=True)
x_train, x_valid, y_train, y_valid = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state=7)

prediction = st.selectbox('Prediction',
                          ['Gold', 'KNN', 'AdaBoost', 'SVM'],
                          index=0)

if prediction != 'Gold':
    model = None
    if prediction == 'KNN':
        model = sklearn.neighbors.KNeighborsClassifier()
    if prediction == 'SVM':
        model = sklearn.svm.SVC()
    if prediction == 'AdaBoost':
        model = sklearn.ensemble.AdaBoostClassifier(n_estimators=100)
    model.fit(x_train, y_train)
    y_prediction = model.predict(x_valid)
    y_prediction_all = model.predict(x)
else:
    y_prediction = y_valid.copy()
    y_prediction_all = y.copy()

data_df_display = create_dataframe(x_display, y_display, x_display.columns)
st.dataframe(data_df_display)

fairness_mesured_set = st.selectbox('Fairness mesured set', ['All', 'Valid'], index=1)

data_df = None
if fairness_mesured_set == 'All':
    data_df = create_dataframe(x=x, y_gold=y, base_columns=x_display.columns, y_pred=y_prediction_all)
if fairness_mesured_set == 'Valid':
    data_df = create_dataframe(x=x_valid, y_gold=y_valid, base_columns=x_display.columns, y_pred=y_prediction)

privilege_col = st.selectbox('discrimination column',
                             list(x_display.columns),
                             index=7)
condition_str = st.text_input('condition for privilege class', value='x==1')

if condition_str is not None:
    data_df = add_priv_class_col(data_df, privilege_col, condition_str)

st.dataframe(data_df)

gold_demographic_parity = FairnessUtils.demographic_parity(data_df, target='target')
old_demographic_parity = FairnessUtils.demographic_parity(data_df)
old_disparate_impact = FairnessUtils.disparate_impact_rate(data_df)
old_equal_opportunity_succ = FairnessUtils.equal_opportunity_succes(data_df)
old_avg_equalized_odds = FairnessUtils.average_equalized_odds(data_df)
old_predictive_rate_parity = FairnessUtils.predictive_rate_parity(data_df)

accuracy_before = sklearn.metrics.accuracy_score(y_true=data_df['target'].tolist(),
                                                 y_pred=data_df['target_pred'].tolist())

st.subheader('Desired metrics')

wanted_demographic_parity = st.slider('Demographic Parity', min_value=-1.0, max_value=1.0, step=0.01,
                                      value=old_demographic_parity)

correction_priority = st.slider('Priorize Non privilege', min_value=0.0, max_value=1.0, step=0.1,
                                value=1.0)

# wanted_equal_opportunity = st.slider('Equal Oportunity', min_value=-1.0, max_value=1.0, step=0.01,
#                                      value=old_equal_opportunity)

st.subheader('Recalculate')
with st.expander('Details'):
    all_data_df = recalculate_from_demographic_parity_multiple(data_df)
new_demographic_parity = calculate_metrics_and_errors(all_data_df, FairnessUtils.demographic_parity)
new_disparate_impact = calculate_metrics_and_errors(all_data_df, FairnessUtils.disparate_impact_rate)
new_equal_opportunity_succ = calculate_metrics_and_errors(all_data_df, FairnessUtils.equal_opportunity_succes)
new_avg_equalized_odds = calculate_metrics_and_errors(all_data_df, FairnessUtils.average_equalized_odds)
new_predictive_rate_parity = calculate_metrics_and_errors(all_data_df, FairnessUtils.predictive_rate_parity)
accuracy_after = calculate_metrics_and_errors(all_data_df, calculate_accuracy)

data = [[accuracy_before, 1.0, accuracy_after[0], accuracy_after[2]],
        [old_demographic_parity, wanted_demographic_parity, new_demographic_parity[0], new_demographic_parity[2]],
        [old_equal_opportunity_succ, 0.0, new_equal_opportunity_succ[0], new_equal_opportunity_succ[2]],
        [old_avg_equalized_odds, 0.0, new_avg_equalized_odds[0], new_avg_equalized_odds[2]],
        [old_predictive_rate_parity, 0.0, new_predictive_rate_parity[0], new_predictive_rate_parity[2]]]

fig_res = create_figure(data=data, metrics=['Accuracy',
                                            'Demographic parity',
                                            'Equal Opportunity (succes)',
                                            'Avg Equalized Odds',
                                            'Predictive rate parity'])

data = [[old_disparate_impact, 1.0, new_disparate_impact[0], new_disparate_impact[2]]]
fig_ratio = create_figure(data=data, metrics=['Disparate Impact'])

st.header('Results')
st.write('Demographic parity on gold :', gold_demographic_parity)
st.write('Demographic parity :', new_demographic_parity)
st.write('Disparate impact:', new_disparate_impact)
st.write('Equal Opportunity (succes):', new_equal_opportunity_succ)
st.write('Avg Equalized Odds:', new_avg_equalized_odds)
st.write('Predictive rate parity', new_predictive_rate_parity)
st.plotly_chart(fig_res, use_container_width=True)
st.plotly_chart(fig_ratio, use_container_width=True)
