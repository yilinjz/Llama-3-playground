
class PromptGenerator():
    def __init__(
        self,
        context: str
    ) -> None:
        self._context = context

    def create_prompt(
        self,    
        query: str
    ) -> str:
        # base prompt
        with open('prompt/base_prompt.txt', 'r') as f:
            base_prompt = f"{''.join(f.readlines())}\n\n"
        # question
        question = f"Question:{query}\n\n"
        # context (i.e. the text data)
        context = f"Context:{self._context}\n\n"
        # answer
        answer = f"Answer:"
        
        prompt = f"{base_prompt}{question}{context}{answer}"
        return prompt
