from typing import Dict, List

from genor_agents.agent import Agent
from genor_agents.classification.regex.regex_binary_classifier_agent import RegexBinaryClassifierAgent


class RegexMultiLabelClassifierAgent(Agent):
    """
    Agent for performing multi-label classification using regular expressions.

    Args:
        regex_binary_classifier_kwargs_per_label (Dict[str, dict]): A dictionary mapping label names to binary regular expression agent kwargs.
    """

    def __init__(self, regex_binary_classifier_kwargs_per_label: Dict[str, dict]):
        super().__init__()
        self.regex_binary_classifier_per_label = self._get_label_to_regex_binary_classifier_mapping(regex_binary_classifier_kwargs_per_label)

    def __call__(self, data: str) -> List[str]:
        """
        Classify the given data into multiple labels based on the regular expression patterns.

        Args:
            data (str): The input data to be classified.

        Returns:
            List[str]: A list of labels that match the regular expression patterns.

        """
        labels = []
        for label, regex_binary_classifier in self.regex_binary_classifier_per_label.items():
            res = regex_binary_classifier(data)
            if res:
                labels.append(label)
        return labels

    def _get_label_to_regex_binary_classifier_mapping(self, regex_binary_classifier_kwargs_per_label: Dict[str, dict]) -> Dict[str, RegexBinaryClassifierAgent]:
        """
        Create a mapping of labels to corresponding RegexBinaryClassifierAgent instances.

        Args:
            regex_binary_classifier_kwargs_per_label (Dict[str, dict]): A dictionary mapping label names to binary regular expression agent kwargs.

        Returns:
            Dict[str, RegexBinaryClassifierAgent]: A dictionary mapping labels to corresponding RegexBinaryClassifierAgent instances.

        """
        regex_binary_classifier_per_label = {}
        for label, regex_binary_classifier_kwargs in regex_binary_classifier_kwargs_per_label.items():
            regex_binary_classifier_per_label[label] = RegexBinaryClassifierAgent(**regex_binary_classifier_kwargs)
        return regex_binary_classifier_per_label
  
def run_example():
    # Example usage
    regex_binary_classifiers_kwargs_per_label = {
        "positive": {"regex_patterns": [".*good.*", ".*great.*", ".*excellent.*"]},
        "negative": {"regex_patterns": [".*bad.*", ".*poor.*", ".*terrible.*"]},
        "neutral": {"regex_patterns": ".*ok.*"}
    }
    classifier = RegexMultiLabelClassifierAgent(regex_binary_classifiers_kwargs_per_label)
    data = "The first product is a great and the second product is ok."
    predicted_class = classifier(data)
    print(predicted_class)  # Output: ['positive', 'neutral']

if __name__ == "__main__":
    run_example()