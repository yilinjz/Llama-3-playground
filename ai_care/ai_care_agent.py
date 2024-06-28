import json

from ai_care.llm_llama import LlamaLLM
from ai_care.prompt_generator import PromptGenerator


class AICareAgent():
    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    ) -> None:
        self._llm = LlamaLLM(model_id)
        self._prompt_generator = PromptGenerator()

    def load_data(
        self,
        path_to_data: str
    ) -> None:
        # json data as dict
        self._data = json.load(open(path_to_data, encoding="utf8"))
        # keep text data only, separated by ','
        self._context = f"{', '.join([obj['TEXT'] for obj in self._data])}"
        # print(self._context)

    def chat(
        self,
        query: str
    ) -> str:
        prompt = self._prompt_generator.create_prompt(
            query, 
            self._context, 
            self._llm._pipeline.tokenizer, 
            'prompt/ocr_instruction.json',
            'zh-HK'
        )
        response = self._llm.run_inference(prompt)
        return prompt, response

    def start_conversation(
        self
    ) -> None:
        while True:
            query = input("問我一個問題: ")
            response = self.chat(query)
            print(response)

    def run_experiment(
            self
    ) -> None:
        experiment_result = []
        experiment_prompt = []

        queries = json.load(open("data/query/query_cantonese.json", encoding="utf8"))

        for entry in queries:
            qid = entry['qid']
            query = entry['zh-HK']
            print(f"========== RUNNING TEST #{qid} ==========")

            entry_res = {}
            entry_res['qid'] = qid
            entry_res['query'] = query

            entry_prompt = {}
            entry_prompt['qid'] = qid
            entry_prompt['query'] = query

            for i in range(0, 5):
                print(f"----- Iteration #{i} -----")
                prompt, response = self.chat(query)
                entry_res[f'run_{i}'] = response
                entry_prompt['prompt'] = prompt

            experiment_result.append(entry_res)
            experiment_prompt.append(entry_prompt)
            print(f"========== TEST #{qid} DONE ==========")
        
        with open('data/exp_res/experiment_result-70B.json', 'w', encoding='utf-8') as f:
            json.dump(experiment_result, f, ensure_ascii=False, indent=4)
        with open('data/experiment_prompt.json', 'w', encoding='utf-8') as f:
            json.dump(experiment_prompt, f, ensure_ascii=False, indent=4)
