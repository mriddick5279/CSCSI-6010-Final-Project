import json
import openai
import time
import yaml
import sacrebleu
#HAD TO ADJUST THIS DUE TO RATE LIMIT ERROR
import time

def get_data(pun_type, task):
    annotation_file=json.load(open('annotations.json'))
    caption_file=json.load(open('captions.json'))
    annotation=[annotation_file[str(i)] for i in range(1000) if annotation_file[str(i)]["category"]==pun_type]
    caption=[caption_file[str(i)] for i in range(1000) if caption_file[str(i)]["category"]==pun_type]

    if task==1:
        data=[{"id":i, 
               "pun_type":annotation[i]["category"],
               "pun_sentence":annotation[i]["pun_sentence"], 
               "pun_phrase":annotation[i]["pun_phrase"], 
               "pun_explanation_image_caption":caption[i]["pun_explanation_image_caption"],     
               } for i in range(len(annotation))]
        return data
    elif task==2:
        data=[{"id":i, 
               "pun_type":annotation[i]["category"],
               "pun_sentence":annotation[i]["pun_sentence"], 
               "pun_phrase":annotation[i]["pun_phrase"], 
               "pun_disambiguator_image_caption":caption[i]["pun_disambiguator_image_caption"],
               "disambiguator_image_caption":caption[i]["disambiguator_image_caption"],
               "translation":{
                    "de":{
                        "meaning 1": annotation[i]["meaning_1"]["translations"]["de"],
                        "meaning 2": annotation[i]["meaning_2"]["translations"]["de"]
                        },
                    "fr":{
                        "meaning 1": annotation[i]["meaning_1"]["translations"]["fr"],
                        "meaning 2": annotation[i]["meaning_2"]["translations"]["fr"]
                        },
                    "ko":{
                        "meaning 1": annotation[i]["meaning_1"]["translations"]["ko"],
                        "meaning 2": annotation[i]["meaning_2"]["translations"]["ko"]
                        },
                    } 
                }for i in range(len(annotation))]
        return data
    elif task==3:
        data=[{"id":i, 
               "pun_type":annotation[i]["category"],
               "pun_sentence":annotation[i]["pun_sentence"], 
               "pun_phrase":annotation[i]["pun_phrase"], 
               "pun_explanation_image_caption":caption[i]["pun_explanation_image_caption"],     
               "translation":{
                    "de":{
                        "meaning 1": annotation[i]["meaning_1"]["translations"]["de"],
                        "meaning 2": annotation[i]["meaning_2"]["translations"]["de"]
                        },
                    "fr":{
                        "meaning 1": annotation[i]["meaning_1"]["translations"]["fr"],
                        "meaning 2": annotation[i]["meaning_2"]["translations"]["fr"]
                        },
                    "ko":{
                        "meaning 1": annotation[i]["meaning_1"]["translations"]["ko"],
                        "meaning 2": annotation[i]["meaning_2"]["translations"]["ko"]
                        },
                    } 
                }for i in range(len(annotation))]
        if pun_type=="homographic":
            for i in range(len(annotation)):
                data[i]["description"]={
                        "meaning 1": annotation[i]["meaning_1"]["description"],
                        "meaning 2": annotation[i]["meaning_2"]["description"],
                        }
        elif pun_type=="heterographic":
            for i in range(len(annotation)):
                data[i]["pun_phrase_alternative"]=annotation[i]["meaning_2"]["heterographic"]
        return data
    else:
        raise ValueError("task must be 1, 2, or 3")
    
"""
EXTENSION PART

Created new function to load meme dataset we have and return it to
the new function in run.py to run our extension task
"""
def get_meme_data():
    meme_file=json.load(open('meme_labels.json', encoding='utf-8'))
    meme=[meme_file[str(i)] for i in range(50)]

    data=[{"id":i, 
               "sentiment":meme[i]["sentiment"],
               "text":meme[i]["text"], 
               "description":meme[i]["description"], 
               "image_name":meme[i]["image_name"],     
               } for i in range(len(meme))]
    return data

"""
EXTENSION PART

Added new statement to allow for a task 4 prompt to be read (look below)
"""
def get_prompt(task, model="", lang=""):
    if task==1:
        with open(f'prompts/task1_{model}.txt') as f:
            prompts = f.read()
        return prompts
    elif task==2:
        with open(f'prompts/task2.txt') as f:
            prompts = f.read()
        return prompts
    elif task==3:
        with open(f'prompts/task3_{model}.txt') as f:
            prompts = f.read()
        return prompts
    elif task=="3_eval_homographic":
        with open(f'prompts/task3_eval_homographic.txt') as f:
            prompts = f.read()
        return prompts
    elif task=="3_eval_heterographic":
        with open(f'prompts/task3_eval_heterographic.txt') as f:
            prompts = f.read()
        return prompts
    elif task==4: # Added stateement to read the prompt for our extension task
        with open(f'prompts/task4.txt') as f:
            prompts = f.read()
        return prompts
    else:
        raise ValueError("task must be 1, 2, 3, '3_eval_homographic' or '3_eval_heterographic'.")

def gpt4_response_base(query='', temperature=None):
    with open('.env.yaml') as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    openai.api_key = configs["openai_key"]

    if temperature is not None: #added is not None
        while True:
            try:
                time.sleep(20)  # Add this line to prevent rate-limiting
                message = [
                    {'role':'user', 'content':query}
                    ]
                completion = openai.ChatCompletion.create(
                        model="gpt-4",
                        #model="gpt-3.5-turbo",
                        messages=message,
                        temperature=temperature
                    )
                return completion["choices"][0]["message"]["content"]
            #add this next block in
            except openai.error.RateLimitError as e:
                print("Rate limit hit. Waiting 30 seconds...")
                time.sleep(30)
                continue

            except Exception as e:
                print(f"Error:{e}")
                time.sleep(10) #change from 5 to 10
                continue
    else:
        while True:
            try:
                time.sleep(20)  #add this line to prevent rate-limiting
                message = [
                    {'role':'user', 'content':query}
                    ]
                completion = openai.ChatCompletion.create(
                        model="gpt-4",
                        #model="gpt-3.5-turbo",
                        messages=message
                    )
                return completion["choices"][0]["message"]["content"]

            #add this next block in
            except openai.error.RateLimitError as e:
                print("Rate limit hit. Waiting 30 seconds...")
                time.sleep(30)
                continue

            except Exception as e:
                print(e)
                time.sleep(10)
                continue


def get_bertscore(bertscore, x, y, lang):
    
    results = bertscore.compute(predictions=[x], references=[y], lang=lang)
    return results['precision'][0]


def get_bleuscore(hyps, refs):
    """
    Given a file of hypothesis and reference files,
    evaluate the BLEU score using Moses scripts.
    """
    score = sacrebleu.corpus_bleu(hyps, [refs]).score
    return score





if __name__ == "__main__":


    

    data=get_data(pun_type="homographic", task=2)
    sample=data[0]
    print(sample["translation"]["de"]["meaning 1"])
    print(sample["translation"]["de"]["meaning 2"])
    print(sample["translation"]["fr"]["meaning 1"])
    print(sample["translation"]["fr"]["meaning 2"])
    
