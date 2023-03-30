import yaml
import os
from irec.connector import utils
from api.dataset.dataset import create_synthetic_data
from api.dataset.utils import load_settings

def run_dataset():

    with open('api/settings/synthetic.yaml', 'r') as f:
        synthetic_config = yaml.safe_load(f)

    for name, config in synthetic_config.items():
        config['name'] = name
        create_synthetic_data(config)

def run_experimental(args):

    cwd = os.popen('pwd').read().rstrip()
    os.system(f"""
        app_path={cwd};
        models={args.agents}; 
        bases={args.dataset_loaders}; 
        eval_pol={args.evaluation_policy}; 
        cd $app_path/irec/irec-cmdline/app/scripts/agents;
        python3 run_agent_best.py --agents $models --dataset_loaders $bases --evaluation_policy $eval_pol
    """)
# cd $app_path/scripts/agents

# python3 run_agent_best.py --agents ${models[@]} --dataset_loaders "${bases[@]}" --evaluation_policy "${eval_pol[@]}"

# cd $app_path/scripts/evaluation

# python3 eval_agent_best.py --agents ${models[@]} --dataset_loaders "${bases[@]}" --evaluation_policy "${eval_pol[@]}" --metrics "${metrics[@]}" --metric_evaluator "${metric_eval[@]}"
# python3 print_latex_table_results.py --agents ${models[@]} --dataset_loaders "${bases[@]}" --evaluation_policy "${eval_pol[@]}" --metrics "${metrics[@]}" --metric_evaluator "${metric_eval[@]}"

