# coding: utf-8
import logging

import ollama

from ai_care.prompt_generator import PromptGenerator


def ai_agent_remote(question):
    client = ollama.Client(host = "http://127.0.0.1:11434")

    response = client.chat(model='llama3:70b', messages=[ { 'role': 'user', 'content': prompt.replace('[dt]',  question), },])
    # print(response)
    return response['message']['content']


def cmd_agent():
    log.info("SYSTEM: cmd agent start >>>>>")

    while True:
        # query_str = asr.speech_to_text()

        query_str = input("Please describe your task, or 'exit' to quit: ")
        if query_str.lower() == 'exit':
            print("Exiting program.")
            break

        if query_str is None:
            continue

        task_class = ai_agent_remote(query_str)

        respond = '好的，我听到"' + task_class + '"指令'

        print(respond)

    log.info("SYSTEM: cmd agent end <<<<<")


if __name__ == '__main__':
    log = logging.getLogger()
    # cmd agent init
    cmd_agent()
    