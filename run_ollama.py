# coding: utf-8
import logging
import json

import ollama

from utils.const import benchmark_list, instruction_list, language_list


def ai_agent_remote(query, context, system_prompt, language):
    prompt_words = {
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
    # query
    query_prompt = f"# {prompt_words['query'][language]}:\n```{query}```\n\n"
    # context (i.e. the text data)
    context_prompt = f"# {prompt_words['context'][language]}:\n```{context}```\n\n"
    # answer
    answer_prompt = f"# {prompt_words['answer'][language]}:\n"

    messages = [
        {"role": "system", "content": f"{system_prompt}"},
        {"role": "user", "content": f"{query_prompt}{context_prompt}{answer_prompt}"}
    ]

    client = ollama.Client(host = "http://127.0.0.1:11434")
    response = client.chat(model='llama3:70b', messages=messages)
    # print(response)
    return response['message']['content']

def process_context(context_json):
    context = []
    for object in context_json:
        text = object['TEXT']
        oritentation = object['ORIENTATION']
        depth = object['DEPTH']
        position = object['POSITION']
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
                    query = qa_pair['query'][language]
                    # each scene goes through 4 instruction scenarios
                    path_to_instruction = f"benchmark/prompt/{instruction_list[k%4]}.json"
                    system_prompt = json.load(open(path_to_instruction, encoding="utf8"))[language]
                    # run inferance
                    data[i]['qa_pairs'][j]['result'][language] = ai_agent_remote(
                        query=query,
                        context=context,
                        system_prompt=system_prompt,
                        language=language,
                    )   

    with open(f'benchmark/experiment_result/f{benchmark_name}-experiment_result.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)                 

    log.info("SYSTEM: cmd agent end <<<<<")


if __name__ == '__main__':
    log = logging.getLogger()
    # cmd agent init
    cmd_agent()
    