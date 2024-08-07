from abc import ABC, abstractmethod
import concurrent.futures

from genor_agents.utils.agents.agent_loader import load_agent

class Agent(ABC):
    """
    Abstract base class for agents.
    """

    def __init__(self) -> None:
        """
        Initializes the Agent object.
        """
        pass # TODO: Implement this function
    
    @abstractmethod
    def __call__(self, **kwargs):
        """
        Abstract method that should be implemented in the child class.
        """
        raise NotImplementedError("This method should be implemented in the child class.")
    
    

class MultiInputsAgentExecuter(Agent):
    """
    Executes an Agent on multiple inputs in parallel.

    Args:
        agent (Agent): The Agent object to be executed on the inputs.

    Returns:
        dict or list: The results of executing the Agent on the inputs.

    Raises:
        ValueError: If the inputs type is not a dictionary or a list.
    """

    def __init__(self, agent_path_to_execute: str) -> None:
        """
        Initializes a new instance of the MultiInputsAgentExecuter class.

        Args:
            agent (Agent): The agent object.

        Returns:
            None
        """
        self.agent = load_agent(agent_path_to_execute)
        super().__init__()

    def __call__(self, inputs, kwargs={}):
        """
        Executes the agent on the given inputs in parallel.

        Args:
            inputs (dict or list): The inputs to be processed by the agent.
            **kwargs: Additional keyword arguments to be passed to the agent.

        Returns:
            dict or list: The results of the agent's execution on the inputs.

        Raises:
            ValueError: If the inputs type is invalid. Inputs must be either a dictionary or a list.
        """
        if isinstance(inputs, dict) or isinstance(inputs, list):
            if isinstance(inputs, list):
                inputs = {i:v for i,v in enumerate(inputs)}
            results = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                agent = self.agent(**kwargs)
                future_to_key = {executor.submit(agent, **value): key for key, value in inputs.items()}
                for future in concurrent.futures.as_completed(future_to_key):
                    key = future_to_key[future]
                    results[key] = future.result()
            if isinstance(inputs, list):
                results = list(results.values())
            return results
        else:
            raise ValueError("Invalid inputs type. Inputs must be either a dictionary or a list.")
        
        

