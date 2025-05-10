import argparse
from evaluate import load
import json
import os
from utils import get_data, get_prompt, gpt4_response_base, get_bertscore, get_bleuscore, get_meme_data
language_dict={"de":"German", "fr":"French", "ko":"Korean"}
def run_task1(pun_type, model):
    data = get_data(pun_type, task=1)[:10]
    #data=get_data(pun_type, task=1)
    acc=0
    cnt=0
    for sample in data:
        cnt+=1
        print(f"▶️ Processing example {cnt}/{len(data)}")
        if model=="gpt4_text":
            prompt=get_prompt(task=1, model="gpt4_text").format(sample["pun_sentence"])
        elif model=="gpt4_caption":
            prompt=get_prompt(task=1, model="gpt4_caption").format(sample["pun_sentence"], sample["pun_explanation_image_caption"])
        response=gpt4_response_base(prompt, temperature=0)

        sample["prompt"]=prompt
        sample["response"]=response
        if sample["pun_phrase"].lower() in response.lower():
            sample["correct"]=True
            acc+=1
        else:
            sample["correct"]=False
        
    print("Accuracy:", acc/cnt)
    return data


def run_task2(pun_type, model,lang):
    bertscore = load("bertscore")
    data = get_data(pun_type, task=2)[:10]  # 👈 LIMIT TO 10 FOR TESTING
    #data=get_data(pun_type, task=2)
    acc=0
    cnt=0
    for sample in data:
        cnt+=1
        print(f"▶️ Processing example {cnt}/{len(data)}")
        prompt=get_prompt(task=2, lang=lang).format(sentence=sample["pun_sentence"], caption=sample["pun_disambiguator_image_caption"], lang=language_dict[lang])
        pun_disambiguator_translation=gpt4_response_base(prompt, temperature=0)
        
        sample["prompt_pun_disambiguator_translation"]=prompt
        sample["response_pun_disambiguator_translation"]=pun_disambiguator_translation

        hypo=pun_disambiguator_translation
        refs=[sample["translation"][lang]["meaning 1"], sample["translation"][lang]["meaning 2"]] 
        
        x_ref_1 = get_bertscore(bertscore, hypo, refs[0], lang)
        x_ref_2 = get_bertscore(bertscore, hypo, refs[1], lang)

        if x_ref_1>x_ref_2:
            sample["correct"]=True
            acc+=1
        else:
            sample["correct"]=False

    print("Accuracy:", acc/cnt)
    return data 

def run_task3(pun_type, model, lang):
    meteor = load('meteor')
    data = get_data(pun_type, task=3)[:10]
    #data=get_data(pun_type, task=3)
    acc=0
    cnt=0
    reconstructions=[]
    for sample in data:
        cnt+=1
        print(f"▶️ Processing example {cnt}/{len(data)}")
        # Reconstruction

        if model=="gpt4_text":
            prompt_recon_1=get_prompt(task=3, model=model, lang=lang).format(sentence=sample["translation"][lang]["meaning 1"], lang=language_dict[lang])
            prompt_recon_2=get_prompt(task=3, model=model, lang=lang).format(sentence=sample["translation"][lang]["meaning 2"], lang=language_dict[lang])
        elif model=="gpt4_caption":
            prompt_recon_1=get_prompt(task=3, model=model, lang=lang).format(sentence=sample["translation"][lang]["meaning 1"], caption=sample["pun_explanation_image_caption"], lang=language_dict[lang])
            prompt_recon_2=get_prompt(task=3, model=model, lang=lang).format(sentence=sample["translation"][lang]["meaning 2"], caption=sample["pun_explanation_image_caption"], lang=language_dict[lang])
        
        response_pun_reconstruction_1=gpt4_response_base(prompt_recon_1, temperature=0)
        response_pun_reconstruction_2=gpt4_response_base(prompt_recon_2, temperature=0)
        sample["prompt_pun_reconstruction_1"]=prompt_recon_1
        sample["response_pun_reconstruction_1"]=response_pun_reconstruction_1
        sample["prompt_pun_reconstruction_2"]=prompt_recon_2
        sample["response_pun_reconstruction_2"]=response_pun_reconstruction_2
        
        for i, response_pun_reconstruction in enumerate([response_pun_reconstruction_1, response_pun_reconstruction_2]):
            if "[English]:" in response_pun_reconstruction:
                response_pun_reconstruction=response_pun_reconstruction.split("[English]:")[1].strip()
            reconstructions.append([sample["pun_sentence"], response_pun_reconstruction])

            # Evaluation
            if pun_type=="homographic":
                prompt_eval=get_prompt(task="3_eval_homographic").format(
                pun_sentence=sample["pun_sentence"], 
                pun_phrase=sample["pun_phrase"],
                description_1 = sample["description"]["meaning 1"], 
                description_2 = sample["description"]["meaning 2"],
                recon_sentence=response_pun_reconstruction,
                )
            elif pun_type== "heterographic":
                prompt_eval=get_prompt(task="3_eval_heterographic").format(
                pun_sentence=sample["pun_sentence"], 
                pun_phrase=sample["pun_phrase"],
                pun_phrase_alternative=sample["pun_phrase_alternative"]["pun_phrase"],
                recon_sentence=response_pun_reconstruction
                )
            response_recon_evaluate=gpt4_response_base(prompt_eval, temperature=0)            
            sample[f"prompt_recon_evaluation_{i+1}"]=prompt_eval
            sample[f"response_recon_evaluate_{i+1}"]=response_recon_evaluate


            if "Answer2" in response_recon_evaluate:
                response_recon_evaluate=response_recon_evaluate.split("Answer2")[1].strip()
            if "Answer 2" in response_recon_evaluate:
                response_recon_evaluate=response_recon_evaluate.split("Answer 2")[1].strip()



            if "yes" in response_recon_evaluate.lower():
                sample[f"correct_{i+1}"]=True
                acc+=1
            else:
                sample[f"correct_{i+1}"]=False
        
    hypos, target=zip(*reconstructions)
    hypos=list(hypos)
    target=list(target)

    bleuscore = get_bleuscore(hypos, target)
    meteorscore = meteor.compute(predictions=hypos, references=target)['meteor']


    
    print("Accuracy:", acc/(cnt*2))
    print("BLEU:", bleuscore)
    print("METEOR:", meteorscore)
    return data

"""
EXTENSION PART

Added new function to do sentiment analysis on meme dataset we collected
"""
def run_task4():
    data = get_meme_data()
    # data = get_meme_data()[:10]
    # data = get_data(pun_type, task=1)[:10]
    #data=get_data(pun_type, task=1)
    acc=0
    cnt=0
    # testing = []
    for sample in data:
        cnt+=1
        print(f"▶️ Processing example {cnt}/{len(data)}")
        prompt = get_prompt(task=4).format(sample["description"], sample["text"])
        # testing.append(prompt)
        # print(f"\t{prompt}")
        response=gpt4_response_base(prompt, temperature=0)

        sample["prompt"] = prompt
        sample["response"] = response
        if sample["sentiment"].lower() in response.lower():
            sample["correct"] = True
            acc+=1
        else:
            sample["correct"] = False
        
    print("Accuracy:", acc/cnt)
    return data

if __name__ == "__main__":


    pun_type="heterographic"
    model="gpr4_text"
    lang="fr"

    result_data=run_task3(pun_type=pun_type, model=model, lang=lang)

    with open(f"./results/task3/result_{pun_type}_{model}_{lang}.json", 'w', encoding='utf-8') as file:
        json.dump(result_data, file, indent="\t", ensure_ascii=False)
    
