import json
import os
import argparse

import run




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="1", help="task type")
    
    parser.add_argument("--pun_type", type=str, default="homographic", help="pun type")
    parser.add_argument("--model", type=str, default="gpt4_caption", help="model for experiment")
    parser.add_argument("--lang", type=str, default="de", help="target language")

    args = parser.parse_args()
    if args.task=="1":
        result_data=run.run_task1(pun_type=args.pun_type, model=args.model)
    elif args.task=="2":
        result_data=run.run_task2(pun_type=args.pun_type, model=args.model, lang=args.lang)
    elif args.task=="3":
        result_data=run.run_task3(pun_type=args.pun_type, model=args.model, lang=args.lang)
    elif args.task=="4": # Calling our extension task "task 4", so I added a new statement to handle that
        result_data=run.run_task4()
    
    if not os.path.exists(f"./results/task{args.task}"):
        os.makedirs(f"./results/task{args.task}")

    if args.task=="1":
        with open(f"./results/task{args.task}/result_{args.pun_type}_{args.model}.json", 'w', encoding='utf-8') as file:
            json.dump(result_data, file, indent="\t", ensure_ascii=False)
    elif args.task=="2":
        with open(f"./results/task{args.task}/result_{args.pun_type}_{args.model}_{args.lang}.json", 'w', encoding='utf-8') as file:
            json.dump(result_data, file, indent="\t", ensure_ascii=False)
    elif args.task=="3":
        with open(f"./results/task{args.task}/result_{args.pun_type}_{args.model}_{args.lang}.json", 'w', encoding='utf-8') as file:
            json.dump(result_data, file, indent="\t", ensure_ascii=False)
    elif args.task=="4": # Added statement to write results to the file below
        with open(f"./results/task{args.task}/result_meme_sentiments.json", 'w', encoding='utf-8') as file:
            json.dump(result_data, file, indent = "\t", ensure_ascii=False)

    