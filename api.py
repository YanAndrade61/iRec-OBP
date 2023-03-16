from argparse import ArgumentParser
from api.commands import execute_experiment

CLI = ArgumentParser()

SUBPARSERS = CLI.add_subparsers(dest="subcommand")

def subcommand(args=[], parent=SUBPARSERS, **kwargs):

    def decorator(func):
        parser = parent.add_parser(kwargs.get("command_name", func.__name__), description=func.__doc__)

        for arg in args:
            parser.add_argument(*arg[0], **arg[1])

        parser.set_defaults(func=func)

    return decorator

def argument(*name_or_flags, **kwargs):
    return ([ *name_or_flags ], kwargs)

@subcommand(
    args=[
        argument("--synthetic-config", type=str, help="Path to config file to read parameters from.", required=True),
        argument("--experimental-config", type=str, help="Path to config file to read parameters from.", required=True),
        argument("--evaluation-config", type=str, help="Path to config file to read parameters from.", required=True),
    ],
    command_name="execute-experiment"
)
def execute_experiment(args):
    execute_experiment(args)

if __name__ == "__main__":
    args = CLI.parse_args()

    args.func(args) if args.subcommand is not None else CLI.print_help()