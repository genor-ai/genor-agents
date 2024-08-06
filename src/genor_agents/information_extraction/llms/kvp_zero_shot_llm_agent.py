from genor_agents.information_extraction.llms.kvp_llm_agent import (
    KVPLLMAgent,
)

class KVPZeroShotLLMAgent(KVPLLMAgent):
    """
    ZeroShotLLMAgent class represents an agent that uses a zero-shot language model (LLM) for extracting information.

    Args:
        information_scheme (str): The information scheme to be extracted.
        llm (str, optional): The type of language model to use. Defaults to "GPT4".
        temperature (float, optional): The temperature parameter for controlling the randomness of the generated responses. Defaults to 0.2.
    """

    def __init__(
        self,
        information_scheme,
        model_name="GPT-4o",
        temperature=0.2,
        **kwargs,
    ) -> None:
        super().__init__(information_scheme, model_name, temperature, **kwargs)

    def __call__(self, text: str):
        """
        Call the zero-shot LLM to extract information from a given text.

        Args:
            text (str): The text from which to extract information.

        Returns:
            str: The extracted information.
        """

        response = self.llm_caller.generate_text(
            system_content=self.system_prompt,
            user_content=text,
            temperature=self.temperature,
        )
        dict_start_idx = response.find("{")
        dict_end_idx = response.rfind("}")
        response_dict = response[dict_start_idx : dict_end_idx + 1]
        extracted_information = eval(
            response_dict
        )  # Convert the string response to a dictionary
        return extracted_information