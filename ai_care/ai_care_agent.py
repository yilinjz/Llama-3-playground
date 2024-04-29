import json
import re

from ai_care.llm_llama import LlamaLLM
from ai_care.prompt_generator import PromptGenerator


class AICareAgent():
    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    ) -> None:
        self._llm = LlamaLLM(model_id)

    def load_data(
        self,
        path_to_data: str
    ) -> None:
        
        # json data as dict
        self._data = json.load(open(path_to_data))
        # text data only, separated by ','
        self._context = ','.join([obj['TEXT'] for obj in self._data])

    def start_conversation(
            self
    ) -> None:
        
        self._prompt_generator = PromptGenerator(self._context)

        while True:
            query = input("Ask a question: ")
            response = self.chat(query)
            print(f"[AFTER-FILTERING] {response}")

    def chat(
        self,
        query: str
    ) -> str:
        prompt = self._prompt_generator.create_prompt(query)
        response = self._llm.run_inference(prompt)
        print(f"[PRE-FILTERING] {response}")

        pattern = re.compile(r"<answer>(.*?)</answer>")
        response = pattern.findall(response)

        return response
    