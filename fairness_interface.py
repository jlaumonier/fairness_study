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


# https://machinesgonewrong.com/fairness/

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
        int_diff_nb_f = round(diff_nb_f)
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
        int_diff_t = round(diff_nb_t)
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
    st.dataframe(df)

    return df


def recalculate_from_demographic_parity_multiple(df, n=1):
    all_df = []
    for i in range(n):
        temp_df = recalculate_from_demographic_parity(df.copy())
        all_df.append(temp_df)
    return all_df


def calculate_metrics_and_errors(dfs, func_metrics):
    metrics = []
    for d in dfs:
        m = func_metrics(d)
        if m is not None:
            metrics.append(m)
    mean = None
    std_error = None
    conf_interval = None
    if len(metrics) > 0:
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


def encode_categories_to_num(df):
    category_columns = list(df.select_dtypes(include=['object']).columns)
    oe = sklearn.preprocessing.OrdinalEncoder()
    df[category_columns] = oe.fit_transform(df[category_columns])
    df = df.apply(pd.to_numeric, args=('coerce',))
    df = df.fillna(0.0)
    print(df)
    return df


def load_data_set(name, target_col=None):
    x_d, y_d = None, None
    if name == 'Adult':
        x_d, y_d = shap.datasets.adult(display=True)
    if name == 'CatDog':
        df = pd.read_csv('artificial.csv', sep=';')
        x_d = df
        if target_col:
            x_d = df.loc[:, ~df.columns.isin([target_col])]
            y_d = df[target_col]
    if name == 'Compas':
        df = pd.read_csv('compas-scores-two-years.csv', sep=',')
        x_d = df
        x_d['Score (Medium, High)'] = (x_d['score_text'] == 'Medium') | (x_d['score_text'] == 'High')
        x_d['Score (Medium, High)'] = x_d['Score (Medium, High)'].astype(int)
        if target_col:
            x_d = df.loc[:, ~df.columns.isin([target_col])]
            y_d = df[target_col]
    return x_d, y_d


def conf_matrices(df):
    all_matrix = sklearn.metrics.confusion_matrix(y_true=df['target'].to_numpy(),
                                                  y_pred=df['target_pred'].to_numpy())
    df_t_condition = df[(df['priv_class'] == True)]
    priv_t_matrix = sklearn.metrics.confusion_matrix(y_true=df_t_condition['target'].to_numpy(),
                                                     y_pred=df_t_condition['target_pred'].to_numpy())
    df_f_condition = df[(df['priv_class'] == False)]
    priv_f_matrix = sklearn.metrics.confusion_matrix(y_true=df_f_condition['target'].to_numpy(),
                                                     y_pred=df_f_condition['target_pred'].to_numpy())
    return all_matrix, priv_t_matrix, priv_f_matrix


def condition_operators(df, column):
    result = ["=="]
    if (str(df[column].dtype) == 'int32') or (str(df[column].dtype) == 'int64') \
            or (str(df[column].dtype) == 'float32') or (str(df[column].dtype) == 'float64'):
        result.append('<')
        result.append('<=')
        result.append('>')
        result.append('>=')
    return result


def condition_values(df, column):
    result = []
    if str(df[column].dtype) == 'object' or str(df[column].dtype) == 'category':
        result.extend(['"' + sub + '"' for sub in df[column].unique()])
    return result


def select_columns(l):
    result = l
    result.insert(0, '<select>')
    return result


def calculate_all_ml_metrics(conf_matrix):
    r = {'tp': conf_matrix[1][1],
         'fp': conf_matrix[0][1],
         'tn': conf_matrix[0][0],
         'fn': conf_matrix[1][0]}
    r['ppv (precision)'] = r['tp'] / (r['tp'] + r['fp'])
    r['tpr (rappel)'] = r['tp'] / (r['tp'] + r['fn'])
    r['fpr'] = r['fp'] / (r['fp'] + r['tn'])
    return r


def print_metrics(metrics):
    for k, v in metrics.items():
        st.write(str(k) + ' = ' + str(v))


st.markdown(f'''
    <style>
        section[data-testid="stSidebar"] .css-ng1t4o {{width: 60rem;}}
        section[data-testid="stSidebar"] .css-1d391kg {{width: 60rem;}}
    </style>
''', unsafe_allow_html=True)

can_calcul = False

with st.sidebar:
    dataset = st.selectbox('Dataset',
                           ['Adult', 'CatDog', 'Compas'],
                           index=0)
    x_display, temp_y = load_data_set(dataset, None)

    target_col = st.selectbox('Target column', select_columns(list(x_display.columns)))

    if target_col != '<select>' or temp_y is not None:

        x_display, y_display = load_data_set(dataset, target_col)

        data_df_display = create_dataframe(x_display, y_display, x_display.columns, )
        labels = data_df_display['target'].unique()
        # strip spaces
        data_df_display = data_df_display.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        with st.expander('Raw data'):
            st.dataframe(data_df_display)

        # privilege parameters
        col1, col2, col3 = st.columns(3)
        privilege_col = col1.selectbox('privilege column',
                                       list(x_display.columns),
                                       index=0)
        condition_operator = col2.selectbox('operator',
                                            condition_operators(data_df_display, privilege_col),
                                            index=0)

        cond_possible_values = condition_values(data_df_display, privilege_col)
        if len(cond_possible_values) > 0:
            condition_value = col3.selectbox('value for privilege class',
                                             cond_possible_values,
                                             index=0)
        else:
            condition_value = col3.text_input('value for privilege class', value=data_df_display[privilege_col].iloc[0])

        condition_str = 'x' + str(condition_operator) + str(condition_value)

        if condition_str is not None:
            data_df_display = add_priv_class_col(data_df_display, privilege_col, condition_str)

        st.dataframe(data_df_display)

        encoded_data_df = encode_categories_to_num(data_df_display).copy()

        colp1, colp2 = st.columns(2)
        prediction = colp1.selectbox('Prediction',
                                     ['Gold', 'KNN', 'AdaBoost', 'SVM', 'Other column'],
                                     index=0)

        target_pred_col = colp2.selectbox('Target prediction column',
                                          select_columns(list(x_display.columns)))

        if prediction != 'Other column' or target_pred_col != '<select>':

            x = encoded_data_df.loc[:, ~encoded_data_df.columns.isin(['target', 'priv_class'])].to_numpy()
            priv_class = encoded_data_df['priv_class'].to_numpy()
            y = encoded_data_df['target'].to_numpy()
            if target_pred_col != '<select>':
                y_pred_col = encoded_data_df[target_pred_col].to_numpy()
                x_train, x_valid, \
                y_train, y_valid, \
                priv_class_train, priv_class_valid, \
                pred_train, y_pred_valid = sklearn.model_selection.train_test_split(x, y, priv_class, y_pred_col,
                                                                                    test_size=0.2, random_state=7)
            else:
                x_train, x_valid, \
                y_train, y_valid, \
                priv_class_train, priv_class_valid = sklearn.model_selection.train_test_split(x, y, priv_class,
                                                                                              test_size=0.2,
                                                                                              random_state=7)

            if prediction not in ['Gold', 'Other column']:
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
                if prediction == 'Gold':
                    y_prediction = y_valid.copy()
                    y_prediction_all = y.copy()
                else:
                    y_prediction = y_pred_valid
                    y_prediction_all = encoded_data_df[target_pred_col].to_numpy()

            fairness_mesured_set = st.selectbox('Fairness mesured set', ['All', 'Valid'], index=0)

            data_df = None
            if fairness_mesured_set == 'All':
                data_df = create_dataframe(x=x, y_gold=y, base_columns=x_display.columns, y_pred=y_prediction_all)
                data_df['priv_class'] = priv_class
            if fairness_mesured_set == 'Valid':
                data_df = create_dataframe(x=x_valid, y_gold=y_valid, base_columns=x_display.columns,
                                           y_pred=y_prediction)
                data_df['priv_class'] = priv_class_valid

            value_positive_choice = st.selectbox('Positive decision',
                                                 ['1 is positive decision', '0 is positive decision'])

            if value_positive_choice == '0 is positive decision':
                data_df['target'] = data_df['target'].replace({0.0: 1.0, 1.0: 0.0})
                data_df['target_pred'] = data_df['target_pred'].replace({0.0: 1.0, 1.0: 0.0})

            st.dataframe(data_df)

            gold_demographic_parity = FairnessUtils.demographic_parity(data_df, target='target')
            old_demographic_parity = FairnessUtils.demographic_parity(data_df)
            old_disparate_impact = FairnessUtils.disparate_impact_rate(data_df)
            old_equal_opportunity_succ = FairnessUtils.equal_opportunity_succes(data_df)
            old_equal_opportunity_succ_ratio = FairnessUtils.equal_opportunity_succes_ratio(data_df)
            old_avg_equalized_odds = FairnessUtils.average_equalized_odds(data_df)
            old_predictive_rate_parity = FairnessUtils.predictive_rate_parity(data_df)
            old_predictive_rate_parity_ratio = FairnessUtils.predictive_rate_parity_ratio(data_df)
            old_predictive_equality = FairnessUtils.predictive_equality(data_df)
            old_predictive_equality_ratio = FairnessUtils.predictive_equality_ratio(data_df)

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

            can_calcul = True

if can_calcul:
    st.subheader('Confusion Matrices')
    matrices = conf_matrices(all_data_df[0])
    with st.expander('All confusion matrix'):
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=matrices[0], display_labels=labels)
        disp.plot()
        st.pyplot(disp.figure_)
        m = calculate_all_ml_metrics(matrices[0])
        print_metrics(m)
    with st.expander('Priv True matrix'):
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=matrices[1], display_labels=labels)
        disp.plot()
        st.pyplot(disp.figure_)
        m = calculate_all_ml_metrics(matrices[1])
        print_metrics(m)
    with st.expander('Priv False matrix'):
        disp = sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix=matrices[2], display_labels=labels)
        disp.plot()
        st.pyplot(disp.figure_)
        m = calculate_all_ml_metrics(matrices[2])
        print_metrics(m)

    new_demographic_parity = calculate_metrics_and_errors(all_data_df, FairnessUtils.demographic_parity)
    new_disparate_impact = calculate_metrics_and_errors(all_data_df, FairnessUtils.disparate_impact_rate)
    new_equal_opportunity_succ = calculate_metrics_and_errors(all_data_df, FairnessUtils.equal_opportunity_succes)
    new_equal_opportunity_succ_ratio = calculate_metrics_and_errors(all_data_df,
                                                                    FairnessUtils.equal_opportunity_succes_ratio)
    new_avg_equalized_odds = calculate_metrics_and_errors(all_data_df, FairnessUtils.average_equalized_odds)
    new_predictive_rate_parity = calculate_metrics_and_errors(all_data_df, FairnessUtils.predictive_rate_parity)
    new_predictive_rate_parity_ratio = calculate_metrics_and_errors(all_data_df,
                                                                    FairnessUtils.predictive_rate_parity_ratio)
    new_predictive_equality = calculate_metrics_and_errors(all_data_df,
                                                           FairnessUtils.predictive_equality)
    new_predictive_equality_ratio = calculate_metrics_and_errors(all_data_df,
                                                                 FairnessUtils.predictive_equality_ratio)
    accuracy_after = calculate_metrics_and_errors(all_data_df, calculate_accuracy)

    data = [[accuracy_before, 1.0, accuracy_after[0], accuracy_after[2]],
            [old_demographic_parity, wanted_demographic_parity, new_demographic_parity[0], new_demographic_parity[2]],
            [old_equal_opportunity_succ, 0.0, new_equal_opportunity_succ[0], new_equal_opportunity_succ[2]],
            [old_avg_equalized_odds, 0.0, new_avg_equalized_odds[0], new_avg_equalized_odds[2]],
            [old_predictive_rate_parity, 0.0, new_predictive_rate_parity[0], new_predictive_rate_parity[2]],
            [old_predictive_equality, 0.0, new_predictive_equality[0], new_predictive_equality[2]]]

    fig_res = create_figure(data=data, metrics=['Accuracy',
                                                'Demographic parity',
                                                'Equal Opportunity (succes)',
                                                'Avg Equalized Odds',
                                                'Predictive rate parity',
                                                'Predictive equality'])

    data = [[old_disparate_impact, 1.0, new_disparate_impact[0], new_disparate_impact[2]],
            [old_equal_opportunity_succ_ratio, 1.0, new_equal_opportunity_succ_ratio[0],
             new_equal_opportunity_succ_ratio[2]],
            [old_predictive_rate_parity_ratio, 1.0, new_predictive_rate_parity_ratio[0],
             new_predictive_rate_parity_ratio[2]],
            [old_predictive_equality_ratio, 1.0, new_predictive_equality_ratio[0],
             new_predictive_equality_ratio[2]]
            ]
    fig_ratio = create_figure(data=data, metrics=['Disparate Impact',
                                                  'Equal Opportunity ratio (succes)',
                                                  'Predictive rate parity ratio',
                                                  'Predictive equality ratio'])

    st.header('Results')
    st.write('Demographic parity on gold :', gold_demographic_parity)
    st.write('Demographic parity :', new_demographic_parity)
    st.write('Disparate impact:', new_disparate_impact)
    st.write('Equal Opportunity (succes):', new_equal_opportunity_succ)
    st.write('Avg Equalized Odds:', new_avg_equalized_odds)
    st.write('Predictive rate parity', new_predictive_rate_parity)
    st.write('Predictive rate parity ratio', new_predictive_rate_parity_ratio)
    st.write('Predictive equality', new_predictive_equality)
    st.write('Predictive equality ratio', new_predictive_equality_ratio)
    st.plotly_chart(fig_res, use_container_width=True)
    st.plotly_chart(fig_ratio, use_container_width=True)
