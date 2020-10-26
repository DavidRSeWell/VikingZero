"""
Module to assist in creating an instantiated class based on
passed parameters from an end user
"""


def load_agent(name):

    module_name = "vikingzero.agents"
    if "tictactoe" in name.lower():
        module_name += ".tictactoe_agents"

    elif "connect4" in name.lower():
        module_name += ".connect4_agent"

    elif "zero" in name.lower():
        module_name += ".alphago"

    print(f"Loading agent {name}")
    mod = __import__(module_name, fromlist=[name])
    klass = getattr(mod, name)
    return klass


def load_env(name):
    module_name = "vikingzero.environments"
    if "tictactoe" in name.lower():
        module_name += ".tictactoe_env"

    elif "connect4" in name.lower():
        module_name += ".connect4_env"

    mod = __import__(module_name, fromlist=[name])
    klass = getattr(mod, name)
    return klass
