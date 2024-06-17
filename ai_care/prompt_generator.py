import json

from transformers import AutoTokenizer


class PromptGenerator():
    def __init__(
        self
    ) -> None:
        pass

    def create_prompt(
        self,    
        query: str,
        context: str,
        tokenizer: AutoTokenizer
    ) -> str:
        # instruction
        # with open('prompt/instruction_cantonese.txt', encoding="utf8", mode='r') as f:
        #     instruction = f"# Instruction\n\n{''.join(f.readlines())}\n\n"
        instruction = json.load(open('prompt/instruction.json', encoding="utf8"))['zh-HK']

        # query
        query_prompt = f"# Query:\n```{query}```\n\n"
        # context (i.e. the text data)
        context_prompt = f"# Context:\n```{context}```\n\n"
        # answer
        answer_prompt = f"# Answer:\n"

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
