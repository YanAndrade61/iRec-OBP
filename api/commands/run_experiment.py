import yaml
import os
from irec.connector import utils
from api.data.dataset import create_synthetic_data
from api.data.utils import load_settings

def run_experiment(args):

#----------------------LOAD YAMLS----------------------------------#
    with open(args.synthetic_config, 'r') as f:
        synthetic_config = yaml.safe_load(f)
    with open(args.experimental_config, 'r') as f:
        experimental_config = yaml.safe_load(f)
    with open(args.evaluation_config, 'r') as f:
        evaluation_config = yaml.safe_load(f)
    with open(os.path.join(args.irec_config,'dataset_agents.yaml'), 'r') as f:
        dataset_agents_parameters = yaml.safe_load(f)


    settings = load_settings(args.irec_config)

    settings["defaults"]["evaluation_policy"] = experimental_config['evaluation_policy']

    # settings["dataset_loaders"]

    for ds in experimental_config['datasets']:
        synthetic_config[ds]['dataset'] = settings['dataset_loaders'][ds]['FullData']['dataset']
        create_synthetic_data(synthetic_config[ds])

    # utils.run_agent_with_dataset_parameters(
    #     experimental_config["agents"],
    #     experimental_config["dataset_loaders"],
    #     settings,
    #     dataset_agents_parameters,
    #     experimental_config["tasks"],
    #     experimental_config["forced_run"]
    # )
    
    # print(settings)