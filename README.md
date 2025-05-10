#  Understanding Pun with Image Explanations (UNPIE) Benchmark
​
Our dataset includes 500 homographic and 500 heterographic pun sentences, each accompanied by visual context and translations. We conduct three distinct tests as part of our study.
We currently provide implementations of two baselines in this code: text-only GPT-4 (`gpt4_text`) and GPT-4 with BLIP-2 caption (`gpt4_caption`).
​
## Preparation
​
Install the required python packages with the following command:
​
```
pip install -r requirements.txt
```
​
# Task 1. Pun Grounding
​
The task challenges machines to identify the phrase that contains the pun within sentences.
Given the English sentence and optionally the corresponding pun-explanation image caption, models should generate a potential pun phrase. 
​
The following command runs both the inference and evaluation process:
​
```
python main.py --task 1 --pun_type $PUN_TYPE --model $MODEL
```
​
Inputs:
```
{
    $PUN_TYPE: ['homographic', 'heterographic'],
    $MODEL: ['gpt4_text', 'gpt4_caption']
}
```
​
The above command `print`s the accuracy metric on termination.
The prompts and responses generated within the process will be saved in  `./task1`.
​
# Task 2. Pun Disambiguation
​
In this task, machines should disambiguate the incongruity within a pun given the image as cue.
Given the English sentence containing a pun and the pun-disambiguator image aligned with one of the meanings constructing the pun, the model outputs a sentence translated into the target language. 
​
The following command runs both the inference and evaluation process:
​
```
python main.py --task 2 --pun_type $PUN_TYPE --model $MODEL --lang $LANG
```
​
Inputs:
​
```
{
    $PUN_TYPE: ['homographic', 'heterographic'],
    $MODEL: ['gpt4_caption'],  # 'gpt4_text' incurs an error, since the pun disambiguation task necessitates the usage of visual context
    $LANG: ['de', 'fr', 'ko']
}
```
​
The above command `print`s the accuracy metric on termination.
The prompts and responses generated within the process will be saved in  `./task2`.
​
# Task 3. Pun Reconstruction
​
Our pun reconstruction task requires machines to reconstruct the full English pun sentence from the translated versions.
Given a disambiguated translation of the pun sentence and optionally the corresponding pun-explanation image caption, the model should generate the original English sentence.
​
The following command runs both the inference and evaluation process:
​
```
python main.py --task 3 --pun_type $PUN_TYPE --model $MODEL --lang $LANG
```
​
Inputs:
​
```
{
    $PUN_TYPE: ['homographic', 'heterographic'],
    $MODEL: ['gpt4_text', 'gpt4_caption'],
    $LANG: ['de', 'fr', 'ko']
}
```
​
The above command `print`s the accuracy metric and automatic text scores (Bleu-4 and METEOR) on termination.
The prompts and responses generated within the process will be saved in  `./task3`.