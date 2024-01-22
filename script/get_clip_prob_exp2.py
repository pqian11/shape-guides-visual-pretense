from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import torch
import json

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="pretrained")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="pretrained")

# Load stimuli information
df = pd.read_csv('data/stimuli_info/exp2_pretend-to-real.csv')

stimuli_all = []
item2options = {}
item2option_set = {}
obj2article = {}
obj_names = []

for row_idx, row in df.iterrows():
    op1 = row['real_object1']
    op2 = row['real_object2']
    article = row['article_of_pretend_object']
    pretend_obj = row['pretend_object']

    options = [op1, op2]

    stimulus = {}
    stimulus['pretend_object'] = pretend_obj
    stimulus['options'] = options
    stimuli_all.append(stimulus)

    obj_names.append(pretend_obj)
    item2options[pretend_obj] = options
    item2option_set[pretend_obj] = set(options)
    obj2article[pretend_obj] = article

print(len(obj_names), 'objects selected')
print(df)
print()

pref_data = {}

for item in obj_names:
    pref_data[item] = {}
    options = item2options[item]
    images = [Image.open('materials/study2/img/{}_faded.png'.format(op)) for op in options]
    inputs = processor(text=item, images=images, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=0)

    for op, prob in zip(options, torch.squeeze(probs).tolist()):
        pref_data[item][op] = prob

json.dump(pref_data, open('exp2_clip_pref_data.json','w'), indent=4)
