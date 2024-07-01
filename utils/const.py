benchmark_list = [
    'bathroom_existence',
    #'bathroom_identification',
    #'bathroom_location',
    #'bathroom_multi-location',
]

instruction_list = [
    'existance_instruction',
    #'identification_instruction',
    #'location_instruction',
    #'multi-location_instruction',
]

language_list = [
    'en-US',
    'zh-HK',
]

prompt_words = {
    "query": {
        "en-US": "Question",
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
