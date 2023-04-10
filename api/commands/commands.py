import yaml
import os
from api.dataset.dataset import create_synthetic_data
from api.dataset.utils import load_settings
from api.experimental.experimental import (
    cmd_run_agent_best,
    cmd_eval_agent_best,
    cmd_print_latex_table_results,
    cmd_export_interactions,
)
from api.evaluation.evaluation import counterfactual_evaluation


def run_dataset():
    """Generate a synthetic dataset for every config passed in synthetic.yaml."""

    with open("api/settings/synthetic.yaml", "r") as f:
        synthetic_config = yaml.safe_load(f)

    for name, config in synthetic_config.items():
        config["name"] = name
        create_synthetic_data(config)


def run_experimental():
    """Execute commands from irec-cmdline with args in experimental.yaml
    for execute the entire experimental step.
    """
    with open("api/settings/experimental.yaml", "r") as f:
        experimental_config = yaml.safe_load(f)

    cmd_run_agent_best(experimental_config)
    cmd_export_interactions(experimental_config)


def run_evaluate():
    """Evaluate datasets with offline metrics with iRec and counterfactual estimators with OBP."""
    with open("api/settings/evaluate.yaml", "r") as f:
        evaluate_config = yaml.safe_load(f)

    # cmd_eval_agent_best()
    # cmd_print_latex_table_results()
    for name, config in evaluate_config.items():
        config["obp"]["name"] = name
        counterfactual_evaluation(config["obp"])
