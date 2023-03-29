import pandas as pd
import numpy as np
import pickle
from irec.environment.loader.full_data import FullData
from .newSynthetic import NewSyntheticBanditDataset
from obp.dataset.synthetic import *
from .utils import check_args
import json
import os


def get_class(class_name: str):
    try:
        return globals()[class_name]
    except KeyError:
        raise Exception(f"Class not found: {class_name}.")


def check_function(obp_args: dict, function: str):
    """Check if a function is require, and load it

    Args:
        obp_args (dict): Contain the parameters of obp dataset.
        function (str): Function to be check, can be behavior_function or reward_function
    """

    if obp_args.get((function), None) is None:
        return

    obp_args[function] = get_class(obp_args[function])


def load_synthetic(obp_args: dict, extra_args: dict):
    """Create a syntheticBanditDataset with arguments passed in synthetic.yaml

    Args:
        obp_args (dict): Contain the parameters of obp dataset.
        extra_args (dict): Contain range of reward, and context file.
    Returns:
        NewBanditDataset: Dataset with all specifications requires.
    """

    synthetic_dataset = NewSyntheticBanditDataset(**obp_args)
    synthetic_dataset.user_context_file = extra_args.user_context_file

    if obp_args.reward_type is "continuous":
        synthetic_dataset.reward_min = (
            extra_args.min_reward or synthetic_dataset.reward_min
        )
        synthetic_dataset.reward_max = (
            extra_args.max_reward or synthetic_dataset.reward_max
        )
    return synthetic_dataset


def save_data_csv(df: pd.DataFrame, name: str):
    """Save dataset in the path that iRec requires.

    Args:
        df (pd.DataFrame): Contain the bandit data formatted in dataframe.
        name (str): Name of created dataset.
    """

    cwd = os.popen("pwd").read().rstrip()
    save_dir = os.path.join(cwd, "irec", "irec-cmdline", "app", "data", "datasets")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    df.to_csv(f"{save_dir}/{name}.csv", index=False)


def get_train_test_idx(config: dict, df: pd.DataFrame):
    """Get index of train and test data generate by iRec splitting.

    Args:
        config (dict): Contain the parameters of dataset creation.
        df (pd.DataFrame): Contain the bandit data formatted in dataframe.
    Returns:
        list: Returns the index of train rows.
        list: Returns the index of test rows.
    """

    loader = FullData(config.dataset, config.splitting)
    _, test_dataset, _, _ = loader.process()
    test_user_ids = np.unique(test_dataset.data[:, 0])
    threshold = test_dataset.data.min(axis=0)[3]

    test_df = df[df.user_id.isin(test_user_ids)].groupby("user_id").head(100)
    train_df = df[df.user_id.isin(test_user_ids) & df.timestamp <= threshold]

    train_idx = train_df.index.tolist()
    test_idx = test_df.index.tolist()
    return train_idx, test_idx

def split_synthetic_data(bandit_data: dict, train_idx: list, test_idx: list):
    """Split bandit data based on indexes of the iRec splitting.

    Args:
        bandit_data (dict): Contain the synthetic data with all parameters.
        train_idx (list): Contain the indexes of rows in train.
        train_idx (list): Contain the indexes of rows in test.
    Returns:
        dict: Returns the train dict.
        dict: Returns the test dict.
    """

    train_dict = {
        k: bandit_data[k][train_idx]
        for k, v in bandit_data.items()
        if isinstance(v, list)
    }
    test_dict = {
        k: bandit_data[k][test_idx]
        for k, v in bandit_data.items()
        if isinstance(v, list)
    }

    train_dict.update(
        {k: bandit_data[k] for k, v in bandit_data.items() if isinstance(v, int)}
    )
    test_dict.update(
        {k: bandit_data[k] for k, v in bandit_data.items() if isinstance(v, int)}
    )
    test_dict.position = None

    train_dict.n_rounds = len(train_idx)
    test_dict.n_rounds = len(test_idx)

    return train_dict, test_dict



def create_synthetic_data(config: dict, dataset_name: str):
    """Generate a synthetic dataset based on synthetic.yaml parameters

    Args:
        config (dict): Contain the necessary parameters to create dataset.
        dataset_name (str): Name of created dataset.
    """

    obp_args = config.obp_args
    extra_args = config.extra_args

    check_args(config)
    check_function(obp_args, "reward_function")
    check_function(obp_args, "behavior_function")

    n_rounds = obp_args.pop("n_rounds")
    synthetic_dataset = load_synthetic(obp_args, extra_args)

    bandit_data = synthetic_dataset.obtain_batch_bandit_feedback(n_rounds)

    df = pd.DataFrame(
        {
            "user_id": bandit_data.users,
            "item_id": bandit_data.action,
            "rating": bandit_data.reward,
            "timestamp": range(len(bandit_data.action)),
        }
    )
    save_data_csv(df, dataset_name)

    train_idx, test_idx = get_train_test_idx(config, df)

    train_dict, test_dict = split_synthetic_data(bandit_data, train_idx, test_idx)


    return train_dict, test_dict