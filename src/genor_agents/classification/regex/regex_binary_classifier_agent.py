import re
from typing import List, Literal

from genor_agents.agent import Agent

class RegexBinaryClassifierAgent(Agent):
    """
    A binary classifier agent that uses regular expressions to classify data.
    Args:
        regex_patterns (List[str]|str): The regular expression pattern(s) to match in the data.
        unwanted_regex_patterns (List[str]|str, optional): The regular expression pattern(s) that must be excluded from the matched context. Defaults to [].
        context_range (int|None, optional): The range of context to include around the matched pattern. Defaults to None. if None, the entire data will be considered.
        relevant_match (Literal["any","first","last"], optional): The type of relevant match to consider. Defaults to "any".          
    """
    def __init__(self, regex_patterns: List[str]|str, unwanted_regex_patterns: List[str]|str=[], context_range: int|None=None, relevant_match: Literal["any","first","last"]="any") -> None:
        super().__init__()
        self.regex_pattern = self._process_regex_patterns(regex_patterns)
        self.unwanted_regex_patterns = self._process_regex_patterns(unwanted_regex_patterns)
        self.context_range = context_range
        self.relevant_match = relevant_match

    def __call__(self, data: str) -> bool:
        """
        Applies the regular expression pattern to the given data and searches if none of the unwanted patterns are present in the context range return True, otherwise False.
        Args:
            data (str): The input data to classify.
        Returns:
            bool: True if a match is found and none of the unwanted patterns are found within the relevant match, False otherwise.
        """
        curr_res = False
        for match in re.finditer(self.regex_pattern, data):
            curr_res = True
            start, end = match.span()
            if self.context_range is None:
                context_span = (0, len(data))
            else:
                context_span = (
                    max(0, start - self.context_range),
                    min(len(data), end + self.context_range),
                )
            context_text = data[context_span[0] : context_span[1]]
            
            if self.unwanted_regex_patterns and re.search(self.unwanted_regex_patterns, context_text):
                curr_res = False
                    
            if self.relevant_match == "any":
                if curr_res:
                    return True
            elif self.relevant_match == "first":
                return curr_res
        
        return curr_res # last match result if relevant_match is "last" or False if no match or unwanted pattern found
            

    def _process_regex_patterns(self, regex_patterns) -> str:
        """
        Process the given regex patterns and return a string representation. if it is a list, join the patterns with "|".
        Args:
            regex_patterns (str or list): The regex patterns to be processed.
        Returns:
            str: The processed regex patterns as a string.
        Raises:
            ValueError: If the regex pattern type is invalid. It should be a string or a list of strings.
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
    # Example for RegexBinaryClassifierAgent
    regex_pattern = r"\d{3}-\d{3}-\d{4}"
    agent = RegexBinaryClassifierAgent(regex_pattern)
    data = "My phone number is 123-456-7890."
    print(agent(data))  # Output: True
    data = "My phone number is 123-456-789."
    print(agent(data))  # Output: False  


if __name__ == "__main__":
    run_example()
