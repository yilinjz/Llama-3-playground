# coding: utf-8
import logging
import json

import ollama

from transformers import AutoTokenizer
from ai_care.prompt_generator import PromptGenerator
from utils.const import benchmark_list, instruction_list, language_list


def ai_agent_remote(query, context, path_to_instruction, language_mode):
    client = ollama.Client(host = "http://127.0.0.1:11434")

    prompt_generator = PromptGenerator()
    prompt = prompt_generator.create_prompt(
        query=query,
        context=context,
        tokenizer=AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct'),
        path_to_instruction=path_to_instruction,
        language_mode=language_mode,
    )
    response = client.chat(model='llama3:70b', messages=[ { 'role': 'user', 'content': prompt.replace('[dt]',  question), },])
    # print(response)
    return response['message']['content']

def process_context(context_json):
    context = []
    for object in context_json:
        text = object['TEXT']
        oritentation = object['ORIENTATION']
        depth = object['depth']
        position = object['position']
        context.append(f"({text}, {oritentation}, {depth}, {position})")
    return context

def cmd_agent():
    log.info("SYSTEM: cmd agent start >>>>>")

    for k, benchmark_name in enumerate(benchmark_list):
        data = json.load(open(f'benchmark/vqa/{benchmark_name}.json', encoding="utf8"))

        # each scene represents an image in dataset
        for i, scene in enumerate(data):
            scene_id = scene['scene_id']
            context_json = json.load(open(f'benchmark/context/{scene_id}.json', encoding="utf8"))
            context = process_context(context_json)

            # each qa_pair is a question
            for j, qa_pair in enumerate(scene['qa_pairs']):
                data[i]['qa_pairs'][j]['result'] = {}
                # English and Cantonese
                for language in language_list:
                    query = qa_pair[language]
                    # each scene goes through 4 instruction scenarios
                    path_to_instruction = f"benchmark/prompt/{instruction_list[k%4]}.json"
                    # run inferance
                    data[i]['qa_pairs'][j]['result'][language] = ai_agent_remote(
                        query=query,
                        context=context,
                        path_to_instruction=path_to_instruction,
                        language_mode=language,
                    )   

    with open(f'benchmark/experiment_result/f{benchmark_name}-experiment_result.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)                 

    log.info("SYSTEM: cmd agent end <<<<<")


if __name__ == '__main__':
    log = logging.getLogger()
    # cmd agent init
    cmd_agent()
    