from abc import abstractmethod

from genor_agents.agent import Agent
from genor_agents.exception_handling import InvalidDataScheme
from genor_agents.utils.llms.llm_factory import LLMFactory


class KVPLLMAgent(Agent):
    def __init__(
        self, information_scheme, model_name="GPT-4o", temperature=0.2, **kwargs
    ) -> None:
        self.information_scheme = information_scheme
        self.llm_caller = LLMFactory.create_llm(model_name, **kwargs)
        self.temperature = temperature
        self.required_attributes = self._get_required_attributes()
        self.optional_attributes = self._get_optional_attributes()
        self.attributes = self.required_attributes + self.optional_attributes
        self._validate_information_scheme()
        self.system_prompt = self._get_system_prompt()

    @abstractmethod
    def __call__(self, text: str):
        raise NotImplementedError("This method must be implemented in the child class")

    def _get_system_prompt(self):
        prompt = "You are an AI assistant tasked with extracting specific information from texts. You will be provided texts, and your goal is to extract the following keys defined here:\n\nKeys:\n"
        prompt += self._create_keys_list()
        prompt += 'If a an information is not present in the text, leave that field empty (""). Structure your findings in the following JSON dictionary format, ensuring accuracy and completeness in the information extraction: {}'
        prompt = prompt.format(self._create_output_format())
        return prompt

    def _create_keys_list(self):
        keys_list = ""
        fields_dicts = self._get_fields_dicts()
        for fld_dict in fields_dicts:
            curr_key = "- "
            curr_key += f"{fld_dict['name']}: {fld_dict['description']}"
            if "source" in fld_dict:
                curr_key = (
                    f"{fld_dict['source']} "
                    + curr_key
                    + f" according to the {fld_dict['source']}"
                )
            keys_list += curr_key + ".\n"
        return keys_list

    def _create_output_format(self):
        def _create_output_format_recursive(d: dict):
            result = {}
            for key, value in d.items():
                if isinstance(value, dict):
                    if "name" in value and not isinstance(value["name"], dict):
                        leaf_v = f"Extracted {value['name']}"
                        if "source" in value:
                            leaf_v += f" according to the {value['source']}"
                        if "format" in value:
                            leaf_v += f" in {value['format']} format"
                        result[key] = f"[{leaf_v}]"

                    else:
                        result[key] = _create_output_format_recursive(value)
            return result

        output_format = _create_output_format_recursive(self.information_scheme)
        return output_format

    def _validate_information_scheme(self):
        fields_dicts = self._get_fields_dicts()
        for fld_dict in fields_dicts:
            for attribute in self.required_attributes:
                if attribute not in fld_dict:
                    raise InvalidDataScheme(
                        f"Missing '{attribute}' attribute for one of the fields"
                    )

    def _get_fields_dicts(self):
        last_level_dicts = []

        def _recursive_extraction(current_dict: dict):
            for value in current_dict.values():
                if isinstance(value, dict):
                    _recursive_extraction(value)
                else:
                    last_level_dicts.append(current_dict)
                    break

        _recursive_extraction(self.information_scheme)
        return last_level_dicts

    def _get_required_attributes(self):
        return ["name", "description", "type"]

    def _get_optional_attributes(self):
        return ["format", "source", "gt_name"]
