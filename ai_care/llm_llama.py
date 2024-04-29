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
        temperature: float = 0.6,
        top_k: int = 20,
        top_p: float = 0.9,
        num_return_sequences: int = 1,
        max_new_tokens: int = 256
    ) -> str:
        outputs = self._pipeline(
            inputs,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            max_new_tokens=max_new_tokens,
            eos_token_id=[
                    self._pipeline.tokenizer.eos_token_id,
                    self._pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ],
            pad_token_id=self._pipelinepipeline.tokenizer.eos_token_id,   
        )
        return outputs[0]['generated_text']
        