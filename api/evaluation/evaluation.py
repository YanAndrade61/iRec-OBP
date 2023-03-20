import numpy as np
from data.newSynthetic import NewSyntheticBanditDataset

from obp.ope import RegressionModel
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

from obp.ope import (
    
    DirectMethod as DM,
    DoublyRobust as DR,
    InverseProbabilityWeighting as IPW,
    DoublyRobustWithShrinkageTuning as DRos,
    SelfNormalizedDoublyRobust as SNDR,
    SelfNormalizedInverseProbabilityWeighting as SNIPW,
    SwitchDoublyRobustTuning as SwitchDR
)
from obp.ope import OffPolicyEvaluation
from obp.ope import RegressionModel


base_model_dict = dict(
    linear_regression=LinearRegression,
    logistic_regression=LogisticRegression,
    lightgbm=GradientBoostingClassifier,
    random_forest=RandomForestClassifier,
)

ope_estimators = [
    DM(),
    IPW(),
    SNIPW(),
    DR(),
    SNDR(),
    SwitchDR(lambdas=[10, 50, 100, 500, 1000, 5000, 10000, np.inf]),
    DRos( lambdas=[10, 50, 100, 500, 1000, 5000, 10000, np.inf]),
]

def counterfactual_evaluation(dataset, bandit_train, bandit_test, config):

    regression_model = RegressionModel(
        n_actions = dataset['n_actions'],
        action_context = dataset['action_context'],
        base_model = base_model_dict[config['regression_model']]
    )
    regression_model.fit(
        context = bandit_train['context'],
        action = bandit_train['action'],
        reward = bandit_train['reward'],
    )
    estimated_rewards = regression_model.predict(
        context = bandit_test['context']
    )
    
