import PIL.Image
import torch
from io import BytesIO
import pandas as pd
import numpy as np
from diffusers import StableDiffusionInpaintPipeline


np.random.seed(1101)


def load_image(path):
    return PIL.Image.open(path)


df = pd.read_csv('data/stimuli_info/exp4_subpart_alignment.csv')

obj_names = df['real_object'].tolist()

item2pretense = {}
pretense2article = {}
item2subparts = {}
object2prompt = {}

for row_idx, row in df.iterrows():
    obj_name = row['real_object']
    pretense_article = row['pretense_article']
    pretense = row['pretense']
    subparts = [row['subpart_1'], row['subpart_2']]

    item2pretense[obj_name] = pretense
    pretense2article[pretense] = pretense_article
    item2subparts[obj_name] = subparts
    object2prompt[obj_name] = '{} {}'.format(pretense_article.title(), pretense)


pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16, cache_dir='pretrained',
)
pipe = pipe.to("cuda")


output_img_width = 512
output_img_height = 512

DIR_PATH = 'exp4_stimuli'

for obj_name in obj_names:
    for instance_idx in range(3):
        img_path = '{}/img/{}_{}.png'.format(DIR_PATH, obj_name, instance_idx)
        mask_path = '{}/mask_with_buffer/{}_{}_mask.png'.format(DIR_PATH, obj_name, instance_idx)

        init_image = load_image(img_path)
        mask_image = load_image(mask_path)

        img_width, img_height = init_image.size
        mask_width, mask_height = mask_image.size

        assert img_width == mask_width
        assert img_height == mask_height

        init_image = init_image.resize((output_img_width, output_img_height))
        mask_image = mask_image.resize((output_img_width, output_img_height))

        prompt = object2prompt[obj_name]
        pretense = item2pretense[obj_name]

        print('Inpainting {} {} as {}'.format(obj_name, instance_idx, pretense))

        images = []

        n_images = 10
        n_sample_per_batch = 5

        while len(images) < n_images:
            output_dict = pipe(prompt=prompt, image=init_image, mask_image=mask_image, height=init_image.size[1], width=init_image.size[0], 
                generator=torch.Generator("cuda").manual_seed(int(np.random.random()*10000)), num_images_per_prompt=n_sample_per_batch,
                strength=1, num_inference_steps=65, return_dict=True
                )

            sample_images = output_dict.images
            nsfw_flags = output_dict.nsfw_content_detected

            for sample_image, nsfw_detected in zip(sample_images, nsfw_flags):
                if nsfw_detected:
                    continue
                images.append(sample_image)

        images = images[:n_images]

        for k, image in enumerate(images):
            image.save("stable_diffusion_output/{}_{}_{}_{}.png".format(obj_name, instance_idx, pretense, k))

        torch.cuda.empty_cache()