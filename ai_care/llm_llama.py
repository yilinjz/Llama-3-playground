import torch
import transformers
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig


class LlamaLLM():
    def __init__(
        self,
        model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    ) -> None:
        self._model_id = model_id
        # print(torch.cuda.is_available())
        # print(torch.__version__)
        # print(transformers.__version__)

        if "Meta-Llama-3-8B" in self._model_id:
            self._pipeline = transformers.pipeline(
                "text-generation", 
                model=model_id, 
                model_kwargs={"torch_dtype": torch.bfloat16}, 
                device_map="cuda"
            )
        elif "Meta-Llama-3-70B" in self._model_id:
            quantize_config = BaseQuantizeConfig(
                bits=4,
                group_size=128,
                desc_act=False
            )
            model = AutoGPTQForCausalLM.from_quantized(
                model_id,
                use_safetensors=True,
                device="cuda",
                quantize_config=quantize_config)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            self._pipeline = transformers.pipeline(
                "text-generation", 
                model=model,
                tokenizer=tokenizer,
                model_kwargs={"torch_dtype": torch.bfloat16}, 
                device_map="cuda"
            )
        else:
            raise ValueError("Model not supported.")

    def run_inference(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.6,
        # repetition_penalty: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1
    ) -> str:
        
        terminators = [
            self._pipeline.tokenizer.eos_token_id,
            self._pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = self._pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            pad_token_id=self._pipeline.tokenizer.eos_token_id,
            do_sample=True,
            temperature=temperature,
            # repetition_penalty=repetition_penalty,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
        )

        return outputs[0]['generated_text'][len(prompt):]
        