from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import pandas as pd
import json


model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", cache_dir="pretrained")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", cache_dir="pretrained")

# Load stimuli info
df = pd.read_csv('data/stimuli_info/exp1_real-to-pretend.csv')

stimuli_all = []
item2options = {}
item2option_set = {}
item2option_words = {}
obj_names = []

for row_idx, row in df.iterrows():
    label = row['label']
    op1 = row['option1']
    op2 = row['option2']
    a1 = row['article_of_option1']
    a2 = row['article_of_option2']

    obj_names.append(label)

    options = ['{} {}'.format(a1, op1), '{} {}'.format(a2, op2)]

    stimulus = {}
    stimulus['label'] = label
    stimulus['options'] = options
    stimuli_all.append(stimulus)

    item2options[label] = options
    item2option_set[label] = set(options)
    item2option_words[label] = [op1, op2]

print(len(obj_names), 'objects selected')
print(df)
print()

pref_data = {}

for item in obj_names:
    pref_data[item] = {}
    image = Image.open('materials/study1/img/{}_faded.png'.format(item))
    options = item2options[item]
    inputs = processor(text=options, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    for op, prob in zip(options, probs.tolist()[0]):
        pref_data[item][op] = prob

json.dump(pref_data, open('exp1_clip_pref_data.json','w'), indent=4)
