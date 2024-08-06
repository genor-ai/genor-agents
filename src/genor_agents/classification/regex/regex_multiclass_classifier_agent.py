import logging
from typing import Dict, List

from genor_agents.agent import Agent
from genor_agents.classification.regex.regex_multilabel_classifier_agent import RegexMultiLabelClassifierAgent



class RegexMultiClassClassifierAgent(Agent):
    """
    Agent for performing multi-class classification using regular expressions.
    Args:
        regex_binary_classifier_kwargs_per_class (Dict[str, dict]): A dictionary mapping class names to binary regular expression agent kwargs.
        classes_priorities (List[str]|None, optional): The order of the classes to be considered when multiple classes match the input. Defaults to None. if None, the order of the classes will be random and this is usefull only when the classes are mutually exclusive.
    """
    def __init__(self, regex_binary_classifier_kwargs_per_class: Dict[str, dict], classes_priorities: List[str]|None=None):
        super().__init__()
        self.regex_multilabel_classifier = RegexMultiLabelClassifierAgent(regex_binary_classifier_kwargs_per_class)
        if classes_priorities is None:
            logging.warning("No classes_priorities provided. There is no order of the labels will be used.")
            self.classes_priorities = list(regex_binary_classifier_kwargs_per_class.keys())
        else:
            self.classes_priorities = classes_priorities

    def __call__(self, data: str) -> str:
        """
        Classifies the input data using the regular expression patterns.

        Args:
            data (str): The input data to be classified.

        Returns:
            str: The predicted class name for the input data or None if no class regex pattern matches the input.
        """
        multi_label_res = self.regex_multilabel_classifier(data)
        for class_ in self.classes_priorities:
            if class_ in multi_label_res:
                return class_
        return None


def run_example():
    # Example usage
    regex_patterns = {
        "positive": {"regex_patterns": [".*good.*", ".*great.*", ".*excellent.*"]},
        "negative": {"regex_patterns": [".*bad.*", ".*poor.*", ".*terrible.*"]},
        "neutral": {"regex_patterns": ".*ok.*"}
    }
    classifier = RegexMultiClassClassifierAgent(regex_patterns)
    data = "This is a great product."
    predicted_class = classifier(data)
    print(predicted_class)  # Output: "positive"

if __name__ == "__main__":
    run_example()