import numpy as np
import os
import pickle
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
    SwitchDoublyRobustTuning as SwitchDR,
)
from obp.ope import OffPolicyEvaluation
from obp.ope import RegressionModel


def get_class(class_name: str):
    try:
        return globals()[class_name]
    except KeyError:
        raise Exception(f"Class not found: {class_name}.")


def get_experimental_files(dataset_name: str):
    cwd = os.popen("pwd").read().rstrip()
    all_models = os.listdir(f"{cwd}/irec/irec-cmdline/app/data/exported_data")
    models = [file for file in all_models if dataset_name in file]

    return models


def load_datasets_pickles(dataset_name: str):
    cwd = os.popen("pwd").read().rstrip()
    load_dir = os.path.join(cwd, "api", "cache")

    with open(f"{load_dir}/{dataset_name}_dataset.pkl", "rb") as fp:
        dataset = pickle.load(fp)
    with open(f"{load_dir}/{dataset_name}_train.pkl", "rb") as fp:
        train = pickle.load(fp)
    with open(f"{load_dir}/{dataset_name}_test.pkl", "rb") as fp:
        test = pickle.load(fp)

    return dataset, train, test


def get_estimated_reward(train: dict, test: dict, regression_model: RegressionModel):
    regression_model.fit(
        context=train["context"],
        action=train["action"],
        reward=train["reward"],
    )
    estimated_rewards = regression_model.predict(
        context=test["context"],
    )

    return estimated_rewards


def irec_action_dist(model: str, test: dict):
    cwd = os.popen("pwd").read().rstrip()
    path = os.path.join(cwd, "irec", "irec-cmdline", "app", "data", "exported_data")

    with open(f"{path}/{model}", mode="rb") as fp:
        data = pickle.load(file=fp)

    u_rec = {}
    action_dist_irec = np.zeros((test["n_rounds"], test["n_actions"], 1))

    for uid, item in data:
        u_rec.setdefault(uid, []).append(item)

    for i, uid in enumerate(test["users"]):
        action_dist_irec[i][u_rec[uid].pop(0)][0] = 1

    return action_dist_irec


def counterfactual_evaluation(args: dict):
    model_files = get_experimental_files(args["name"])

    dataset, train, test = load_datasets_pickles(args["name"])

    regression_model = RegressionModel(
        n_actions=dataset.n_actions, base_model=get_class(args["regression_model"])()
    )

    ope_estimators = [get_class(e)() for e in args["estimators"]]

    estimated_reward = get_estimated_reward(train, test, regression_model)

    for model in model_files:
        action_dist = irec_action_dist(model, test)

        policy_value = dataset.calc_ground_truth_policy_value(
            expected_reward=test["expected_reward"], action_dist=action_dist
        )

        ope = OffPolicyEvaluation(
            bandit_feedback=test,
            ope_estimators=ope_estimators,
        )

        estimated_policy_value, estimated_interval = ope.summarize_off_policy_estimates(
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_reward,
            random_state=12345,
            n_bootstrap_samples=1000,
        )

        squared_errors = ope.evaluate_performance_of_estimators(
            ground_truth_policy_value=policy_value,
            action_dist=action_dist,
            estimated_rewards_by_reg_model=estimated_reward,
            metric="se"
        )

        print(model)
        print(estimated_policy_value)
        print(estimated_interval)
        print(squared_errors)

        break
