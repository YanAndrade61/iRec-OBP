import os


def cmd_run_agent_best(args: dict):
    """Execute run-agent-best from irec-cmdline with args passed.

    Args:
        args (dict): Arguments to execute cmdline.
    """

    cwd = os.popen("pwd").read().rstrip()
    path = f"{cwd}/irec/irec-cmdline/app/scripts"
    print(" ".join(args["agents"]))
    os.system(
        f"""
        cd {path}/agents;
        python3 run_agent_best.py\
            --agents {' '.join(args['agents'])}\
            --dataset_loaders {' '.join(args['dataset_loaders'])}\
            --evaluation_policy {' '.join(args['evaluation_policy'])}\
            --forced_run
    """
    )


def cmd_eval_agent_best(args: dict):
    """Execute eval-agent-best from irec-cmdline with args passed.

    Args:
        args (dict): Arguments to execute cmdline.
    """

    cwd = os.popen("pwd").read().rstrip()
    path = f"{cwd}/irec/irec-cmdline/app/scripts"

    os.system(
        f"""
        cd {path}/evaluation;
        python3 eval_agent_best.py\
            --agents {' '.join(args['agents'])}\
            --dataset_loaders {' '.join(args['dataset_loaders'])}\
            --evaluation_policy {' '.join(args['evaluation_policy'])}\
            --metrics {' '.join(args['metrics'])}\
            --metric_evaluator {' '.join(args['metric_evaluator'])}\
            --forced_run
    """
    )


def cmd_print_latex_table_results(args: dict):
    """Execute print-latex-table-results from irec-cmdline with args passed.

    Args:
        args (dict): Arguments to execute cmdline.
    """

    cwd = os.popen("pwd").read().rstrip()
    path = f"{cwd}/irec/irec-cmdline/app/scripts"

    os.system(
        f"""
        cd {path}/evaluation;
        python3 print_latex_table_results.py\
            --agents {' '.join(args['agents'])}\
            --dataset_loaders {' '.join(args['dataset_loaders'])}\
            --evaluation_policy {' '.join(args['evaluation_policy'])}\
            --metrics {' '.join(args['metrics'])}\
            --metric_evaluator {' '.join(args['metric_evaluator'])}
    """
    )


def cmd_export_interactions(args: dict):
    """Execute export-interactions from irec-cmdline with args passed.

    Args:
        args (dict): Arguments to execute cmdline.
    """

    cwd = os.popen("pwd").read().rstrip()
    path = f"{cwd}/irec/irec-cmdline/app/scripts"

    os.system(
        f"""
        cd {path}/others;
        python3 export_interactions.py\
            --agents {' '.join(args['agents'])}\
            --dataset_loaders {' '.join(args['dataset_loaders'])}\
            --evaluation_policy {' '.join(args['evaluation_policy'])}
    """
    )
