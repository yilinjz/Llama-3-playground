import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
parent = os.path.dirname(SCRIPT_DIR)
parent = os.path.dirname(parent)
sys.path.append(parent)


from ai_care.ai_care_agent import AICareAgent

def main(args):
    model_id = args.model_id
    path_to_data = args.path_to_data

    agent = AICareAgent(model_id)
    agent.load_data(path_to_data)
    
    # agent.start_conversation()
    start_time = time.time()
    agent.run_experiment()
    print("--- %s seconds ---" % (time.time() - start_time))


""" sample run:
python run_ai_care.py  --model_id=meta-llama/Meta-Llama-3-8B-Instruct --path_to_data=data/ocr/ocr_azure.json
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_id",
        type=str,
        required=True,
        help="Model ID card.",
    )
    parser.add_argument(
        "--path_to_data",
        type=str,
        required=True,
        help="Path to the data file.",
    )   
    args = parser.parse_args()
    main(args)
