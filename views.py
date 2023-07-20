import openai
import json
import numpy as np
import textwrap
import re
from time import time,sleep


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

prompt_template = """
Use the following passage to give detailed answer of the question:

QUESTION:{}

PASSAGE:{}


DETAILED ANSWER:
"""

prompt_template_sum = 'Write a detailed summary of the following:\n\n<<SUMMARY>>\n\nDETAILED SUMMARY:'

openai.api_key = "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"


def gpt3_embedding(content, engine='text-similarity-ada-001'):
    content = content.encode(encoding='ASCII',errors='ignore').decode()
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


def similarity(v1, v2):  # return dot product of two vectors
    return np.dot(v1, v2)


def search_index(text, data, count=5):
    vector = gpt3_embedding(text)
    scores = list()
    for i in data:
        score = similarity(vector, i['vector'])
        scores.append({'content': i['content'], 'score': score})
    ordered = sorted(scores, key=lambda d: d['score'], reverse=True)
    return ordered[0:count]


def gpt3_completion(prompt, engine='text-davinci-003', temp=0.6, top_p=1.0, tokens=2000, freq_pen=0.25, pres_pen=0.0, stop=['<<END>>']):
    max_retry = 5
    retry = 0
    prompt = prompt.encode(encoding='ASCII',errors='ignore').decode()
    while True:
        try:
            response = openai.Completion.create(
                engine=engine,
                prompt=prompt,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['text'].strip()
            text = re.sub('\s+', ' ', text)
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % oops
            print('Error communicating with OpenAI:', oops)
            sleep(1)


def ask_question(query):
    with open('index.json', 'r') as infile:
        index_data = json.load(infile)
        results = search_index(query, index_data)
        answers = []
        for result in results:
            prompt = prompt_template.format(query, result['content'])
            answer = gpt3_completion(prompt)
                            # Combine all answers into one string
        final_answer = ""
        for i, result in enumerate(results):
            prompt = prompt_template.format(query, result['content'])
            answer = gpt3_completion(prompt)
            final_answer += f"{i+1}. {answer}\n\n"
           
    return final_answer

