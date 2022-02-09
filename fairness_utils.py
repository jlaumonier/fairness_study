import operator
from functools import reduce
from operator import eq


class FairnessUtils:

    @classmethod
    def calculate_cond_probability(cls, df, ofs, givens):
        """ calculate conditional probability of 'ofs' given 'givens' """
        givens_conditions = []
        ofs_conditions = []
        for of in ofs:
            ofs_conditions.append(eq(df[of[0]], of[1]))
        for given in givens:
            givens_conditions.append(eq(df[given[0]], given[1]))
            ofs_conditions.append(eq(df[given[0]], given[1]))

        ofs_logical_conditions = reduce(operator.and_, ofs_conditions)
        givens_logical_conditions = reduce(operator.and_, givens_conditions)

        set_ofs_inter_givens = df.loc[ofs_logical_conditions]
        card_ofs_inter_givens = len(set_ofs_inter_givens.index)
        set_givens = df.loc[givens_logical_conditions]
        card_givens = len(set_givens.index)
        result = card_ofs_inter_givens / card_givens
        return result

    @classmethod
    def demographic_parity(cls, df, target='target_pred'):
        # Demographic parity
        # P(Y=1 | Priv_class = False) - P(Y=1 | Priv_class = True)

        false_proba = FairnessUtils.calculate_cond_probability(df, [(target, 1)], [('priv_class', False)])
        true_proba = FairnessUtils.calculate_cond_probability(df, [(target, 1)], [('priv_class', True)])
        demographic_parity = false_proba - true_proba
        return demographic_parity

    @classmethod
    def disparate_impact_rate(cls, df):
        # Disparate impact ratio
        # P(Ŷ=1 | Priv_class = False) / P(Ŷ=1 | Priv_class = True)
        false_proba = FairnessUtils.calculate_cond_probability(df, [('target_pred', 1)], [('priv_class', False)])
        true_proba = FairnessUtils.calculate_cond_probability(df, [('target_pred', 1)], [('priv_class', True)])
        if true_proba != 0.0:
            disparate_impact_rate = false_proba / true_proba
        else:
            disparate_impact_rate = None
        return disparate_impact_rate

    @classmethod
    def equal_opportunity_error(cls, df):
        # equal opportunity
        # P(Y_hat=0 | Y=1, Priv_class = False) - P(Y_hat=0 | Y=1, Priv_class = True)
        false_proba = FairnessUtils.calculate_cond_probability(df, [('target_pred', 0)],
                                                               [('target', 1),
                                                                ('priv_class', False)])
        true_proba = FairnessUtils.calculate_cond_probability(df, [('target_pred', 0)],
                                                              [('target', 1),
                                                               ('priv_class', True)])
        equal_opportunity = false_proba - true_proba
        return equal_opportunity

    @classmethod
    def equal_opportunity_succes(cls, df):
        # equal opportunity
        # P(Y_hat=1 | Y=1, Priv_class = False) - P(Y_hat=1 | Y=1, Priv_class = True)
        false_proba = FairnessUtils.calculate_cond_probability(df, [('target_pred', 1)],
                                                               [('target', 1),
                                                                ('priv_class', False)])
        true_proba = FairnessUtils.calculate_cond_probability(df, [('target_pred', 1)],
                                                              [('target', 1),
                                                               ('priv_class', True)])
        equal_opportunity = false_proba - true_proba
        return equal_opportunity

    @classmethod
    def equal_opportunity_succes_ratio(cls, df):
        # equal opportunity
        # P(Y_hat=1 | Y=1, Priv_class = False) / P(Y_hat=1 | Y=1, Priv_class = True)
        equal_opportunity = None
        false_proba = FairnessUtils.calculate_cond_probability(df, [('target_pred', 1)],
                                                               [('target', 1),
                                                                ('priv_class', False)])
        true_proba = FairnessUtils.calculate_cond_probability(df, [('target_pred', 1)],
                                                              [('target', 1),
                                                               ('priv_class', True)])
        if true_proba != 0.0:
            equal_opportunity = false_proba / true_proba
        return equal_opportunity

    @classmethod
    def average_equalized_odds(cls, df):
        # average equalized odds
        # Sum_i\inI [P(Y_hat=1 | Y=i, Priv_class = False) - P(Y_hat=1 | Y=i, Priv_class = True)] / |I|
        diff = 0
        for i in [0, 1]:
            false_proba = FairnessUtils.calculate_cond_probability(df,
                                                                   [('target_pred', 1)],
                                                                   [('target', i),
                                                                    ('priv_class', False)])
            true_proba = FairnessUtils.calculate_cond_probability(df,
                                                                  [('target_pred', 1)],
                                                                  [('target', i),
                                                                   ('priv_class', True)])
            diff = diff + (false_proba - true_proba)
        average_equalized_odds = diff / 2.0
        return average_equalized_odds

    @classmethod
    def predictive_rate_parity(cls, df):
        # predictive rate parity
        # P(Y=1|Y_hat=1, Priv_class=False) - P(Y=1|Y_hat=1, Priv_class=True)
        predictive_rate_parity = None
        if ((len(df[(df['target_pred'] == 1) & (df['priv_class'] == False)]) != 0) and
                (len(df[(df['target_pred'] == 1) & (df['priv_class'] == True)]) != 0)):
            false_proba = FairnessUtils.calculate_cond_probability(df, [('target', 1)],
                                                                   [('target_pred', 1),
                                                                    ('priv_class', False)])
            true_proba = FairnessUtils.calculate_cond_probability(df, [('target', 1)],
                                                                  [('target_pred', 1),
                                                                   ('priv_class', True)])
            predictive_rate_parity = false_proba - true_proba
        return predictive_rate_parity

    @classmethod
    def predictive_rate_parity_ratio(cls, df):
        # predictive rate parity ratio
        # P(Y=1|Y_hat=1, Priv_class=False) / P(Y=1|Y_hat=1, Priv_class=True)
        predictive_rate_parity = None
        if ((len(df[(df['target_pred'] == 1) & (df['priv_class'] == False)]) != 0) and
                (len(df[(df['target_pred'] == 1) & (df['priv_class'] == True)]) != 0)):
            false_proba = FairnessUtils.calculate_cond_probability(df, [('target', 1)],
                                                                   [('target_pred', 1),
                                                                    ('priv_class', False)])
            true_proba = FairnessUtils.calculate_cond_probability(df, [('target', 1)],
                                                                  [('target_pred', 1),
                                                                   ('priv_class', True)])
            if true_proba != 0.0:
                predictive_rate_parity = false_proba / true_proba
        return predictive_rate_parity
