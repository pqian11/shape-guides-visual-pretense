import os
import openai
import requests
import pandas as pd
import json

with open('openai_key.txt') as f:
    api_key = f.readline().strip()

openai.api_key = api_key

df = pd.read_csv('data/stimuli_info/exp4_subpart_alignment.csv')

obj_names = df['real_object'].tolist()
print(obj_names)

item2pretense = {}
pretense2article = {}
item2subparts = {}

for row_idx, row in df.iterrows():
    obj_name = row['real_object']
    pretense_article = row['pretense_article']
    pretense = row['pretense']
    subparts = [row['subpart_1'], row['subpart_2']]

    item2pretense[obj_name] = pretense
    pretense2article[pretense] = pretense_article
    item2subparts[obj_name] = subparts


n_instance = 3
n_sample = 10
img_output_size = "512x512"

inpainting_rs = {}

for obj_name in obj_names:
    pretense = item2pretense[obj_name]
    prompt = '{} {}'.format(pretense2article[pretense].title(), pretense)
    for index in range(n_instance):
        print('Inpaint {}_{} given prompt "{}"'.format(obj_name, index, prompt))

        response = openai.Image.create_edit(
            image=open("materials/study4/inpainting_input/img_with_mask/{}_{}.png".format(obj_name, index), "rb"),
            prompt=prompt,
            n=n_sample,
            size=img_output_size
        )

        # print(response)
        inpainting_rs['{}_{}_{}'.format(obj_name, index, pretense)] = response

        for k, img_url in enumerate(response['data']):
            img_data = requests.get(img_url['url']).content
            with open('dalle_output/{}_{}_{}_{}.png'.format(obj_name, index, pretense, k), 'wb') as handler:
                handler.write(img_data)

with open('exp4_dalle_inpainting_responses.json', 'w') as f:
    json.dump(inpainting_rs, f, indent=4)

