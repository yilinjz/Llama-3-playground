import json

from transformers import AutoTokenizer

class PromptGenerator():
    def __init__(
        self
    ) -> None:
        self._prompt_words = {
            "query": {
                "en-US": "Query",
                "zh-HK": "問題"
            },
            "context": {
                "en-US": "Context",
                "zh-HK": "上下文"
            },
            "answer": {
                "en-US": "Answer",
                "zh-HK": "答案"
            }
        }

    def create_prompt(
        self,    
        query: str,
        context: str,
        tokenizer: AutoTokenizer,
        instruction_file_path: str,
        language_mode: str
    ) -> str:
        
        # instruction
        # with open('prompt/instruction_cantonese.txt', encoding="utf8", mode='r') as f:
        #     instruction = f"# Instruction\n\n{''.join(f.readlines())}\n\n"
        instruction = json.load(open(instruction_file_path, encoding="utf8"))[language_mode]

        # query
        query_prompt = f"# {self._prompt_words['query']}:\n```{query}```\n\n"
        # context (i.e. the text data)
        context_prompt = f"# {self._prompt_words['context']}:\n```{context}```\n\n"
        # answer
        answer_prompt = f"# {self._prompt_words['answer']}:\n"

        messages = [
            {"role": "system", "content": f"{instruction}"},
            {"role": "user", "content": f"{query_prompt}{context_prompt}{answer_prompt}"}
        ]
        
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        return prompt
