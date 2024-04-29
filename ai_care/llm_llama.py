import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer


class LlamaLLM():
    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    ) -> None:
        self._model_id = model_id

        if "Meta-Llama-3" in self._model_id:
            self._pipeline = transformers.pipeline(
                "text-generation", 
                model=model_id, 
                model_kwargs={"torch_dtype": torch.bfloat16}, 
                device_map="auto"
            )
        elif "Llama-2" in self._model_id:
            self._pipeline = transformers.pipeline(
                "text-generation",
                model=LlamaForCausalLM.from_pretrained(model_id),
                tokenizer=LlamaTokenizer.from_pretrained(model_id),
                torch_dtype=torch.float16,
                device_map="auto",
            )
        else:
            raise ValueError("Model not supported.")

    def run_inference(
        self,
        inputs: str,
        temperature: float = 0.7,
        top_k: int = 20,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        max_new_tokens: int = 200
    ) -> str:
        outputs = self._pipeline(
            inputs,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            # pad_token_id=self._tokenizer.eos_token_id,
            # eos_token_id=self._tokenizer.eos_token_id,
        )
        return outputs[0]['generated_text']
        