import importlib

def load_agent(agent_path: str):
    """
    Loads an agent from the given path.

    Args:
        agent_path (str): The path to the agent to be loaded.

    Returns:
        Agent: The loaded agent.
    """
    module_name, class_name = agent_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls
