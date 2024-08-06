import os
from abc import ABC, abstractmethod
import yaml
from openai import AzureOpenAI


class LLM(ABC):
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for model_name in cls.supported_models():
            LLM._registry[model_name] = cls

    @staticmethod
    def supported_models():
        return []

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate_text(self, system_content: str, user_content: str, **kwargs) -> str:
        raise NotImplementedError("This method must be implemented in the child class")


class LLMFactory:
    @staticmethod
    def create_llm(model_name: str, **kwargs) -> LLM:
        llm_class = LLM._registry.get(model_name)
        if llm_class is not None:
            return llm_class(model_name, **kwargs)
        else:
            raise ValueError(f"Model {model_name} is not supported by the factory")


class OpenAIModel(LLM):
    @staticmethod
    def supported_models():
        return ["GPT-4", "GPT-4o", "text-davinci-003", "gpt-3.5-turbo"]

    def __init__(self, model_name: str):
        super().__init__(model_name)
        
        with open(os.path.join('conf', 'local', 'credentials', 'azure_openai.yml' ), "r") as f:
            credentials_dict = yaml.safe_load(f)
        self.api_key = credentials_dict['api_key']
        self.endpoint = credentials_dict['endpoint']
        self.api_version = credentials_dict['api_version']
        
        self.llm = AzureOpenAI(
            api_key=self.api_key, azure_endpoint=self.endpoint, api_version=self.api_version
        )

    def generate_text(self, system_content: str, user_content: str, **kwargs) -> str:
        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        response = self.llm.chat.completions.create(
            model=self.model_name, messages=messages, **kwargs
        )
        return response.choices[0].message.content.strip()


def run_example():
    # Example
    model_name = "GPT-4o"
    API_VERSION = os.getenv("OPENAI_API_VERSION")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
    llm_model = LLMFactory.create_llm(
        model_name,
        api_key=AZURE_OPENAI_KEY,
        endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=API_VERSION,
    )
    print(llm_model.generate_text("Hello", "How are you?"))
    

if __name__ == "__main__":
    run_example()