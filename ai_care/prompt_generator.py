
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
        # context (i.e. the text data)
        context = f"<context>{self._context}</context>\n\n"
        # instruction
        with open('prompt/instruction.txt', 'r') as f:
            instruction = f"<instruction>{''.join(f.readlines())}</instruction>\n\n"
        # example
        with open('prompt/example_question.txt', 'r') as f:
            question = f"<question>\n{''.join(f.readlines())}</question>\n\n"
        with open('prompt/example_answer.txt', 'r') as f:
            answer = f"<answer>\n{''.join(f.readlines())}</answer>\n"
        example = f"<example>\n{question}{answer}</example>\n\n"
        question = f"<question>{query}</question>\n\n"
        answer = f"<answer></answer>\n\n"

        prompt = f"{base_prompt}{context}{instruction}{example}{question}{answer}"
        return prompt
