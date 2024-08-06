import re

from genor_agents.agent import Agent


class RegexAgent(Agent):
    """
    A class representing a regular expression agent.

    This agent is used to match and extract patterns from input data using regular expressions.

    Attributes:
        regex_pattern (str): The regular expression pattern to match.
        group_number (int): The group number to extract from the match.
    """

    def __init__(self, regex_patterns, group_number=0):
        """
        Initializes a RegexAgent object.

        Args:
            regex_patterns (str or list): The regular expression pattern(s) to be used for matching.
                If a string is provided, it will be converted to a list with a single pattern.
                If a list is provided, each element should be a string representing a regular expression pattern.
            group_number (int, optional): The group number to extract from the matched pattern.
                Defaults to 0, which represents the entire matched pattern.

        Returns:
            None
        """
        super().__init__()
        self.regex_pattern = self._process_regex_patterns(regex_patterns)
        self.group_number = group_number

    def __call__(self, data):
        """
        Matches the regex pattern against the input data and returns the extracted group.

        Args:
            data (str): The input data to match against.

        Returns:
            str: The extracted group if a match is found, empty string otherwise.

        """
        match = re.search(self.regex_pattern, data)
        if isinstance(match, re.Match):
            return match.group(self.group_number)
        else:
            return ""

    def _process_regex_patterns(self, regex_patterns):
        """
        Processes the regex patterns and returns a single pattern string.

        Args:
            regex_patterns (str or list): The regex patterns to process.

        Returns:
            str: The processed regex pattern.

        Raises:
            ValueError: If the regex pattern type is invalid.

        """
        if isinstance(regex_patterns, str):
            return regex_patterns
        elif isinstance(regex_patterns, list):
            return "|".join(regex_patterns)
        else:
            raise ValueError(
                "Invalid regex pattern type. It should be a string or a list of strings."
            )

def run_example():
    # Example
    regex_pattern = r"\d{3}-\d{3}-\d{4}"
    agent = RegexAgent(regex_pattern)
    data = "My phone number is 123-456-7890."
    print(agent(data))  # Output: '123-456-7890'

if __name__ == "__main__":
    run_example()
