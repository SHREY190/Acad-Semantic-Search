import openai
import json
import textwrap


def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

OPENAI_API_KEY = ''
with open('OPENAI_API_KEY.txt', 'r') as file_to_read:
    OPENAI_API_KEY = file_to_read.read()

openai.api_key = OPENAI_API_KEY


def gpt3_embedding(content, engine='text-similarity-ada-001'):
    response = openai.Embedding.create(input=content,engine=engine)
    vector = response['data'][0]['embedding']  # this is a normal list
    return vector


if __name__ == '__main__':
    alltext = open_file('case.txt')
    chunks = textwrap.wrap(alltext, 4000)
    result = list()
    for chunk in chunks:
        embedding = gpt3_embedding(chunk.encode(encoding='ASCII',errors='ignore').decode())
        info = {'content': chunk, 'vector': embedding}
        print(info, '\n\n\n')
        result.append(info)
    with open('index.json', 'w') as outfile:
        json.dump(result, outfile, indent=2)