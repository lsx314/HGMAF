# -*- coding: utf-8 -*-

"""
@description  : generated_entity_image_demo
@Author       : lsx
@Email        : lsx314159@163.com
"""
import json
import os
import re

import torch
from PIL import Image
from tqdm import tqdm

"""
 SDXL-Turbo Model Card
 reference :https://huggingface.co/stabilityai/sd-turbo and https://huggingface.co/stabilityai/sdxl-turbo
"""
""

# from diffusers import AutoPipelineForText2Image
# from modelscope import snapshot_download
#
# model_dir = snapshot_download("AI-ModelScope/sdxl-turbo")
# pipe = AutoPipelineForText2Image.from_pretrained(model_dir, torch_dtype=torch.float16, variant="fp16")
# pipe.to("cuda")

from diffusers import AutoPipelineForText2Image
import torch

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")


#
# prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
# image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]


# import torch
# from diffusers import AutoPipelineForText2Image
# from modelscope import snapshot_download
#
# model_dir = snapshot_download("AI-ModelScope/sdxl-turbo")
#
# pipe = AutoPipelineForText2Image.from_pretrained(model_dir, torch_dtype=torch.float16, variant="fp16")
# pipe.to("cuda")
#
# prompt = "Cat"
#
# image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
# image.save("cat.png")

def process_entity(_dir):
    imgs_ids = []
    entity_list = []
    for filename in os.listdir(_dir):
        file_path = os.path.join(_dir, filename)
        if os.path.isfile(file_path):
            imgs_ids.append(filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                entity_list.append(content)
        else:
            raise ValueError("has wrong,please check the path")

    return imgs_ids, entity_list


def process_files(_dir, model="train"):
    imgs_ids, entity_list = process_entity(_dir)

    for image, words in tqdm(zip(imgs_ids, entity_list)):
        image_name = image
        filtered_words = [word for word in words if
                          not (word in ["RT", "_", ":"] or word.startswith('@') or word.startswith(
                              'http:') or word.startswith("https") or word.startswith("//t")
                               or word.startswith("co/"))]

        filtered_words = [word.replace("#", "") for word in filtered_words]
        filtered_words = " ".join(filtered_words)
        filtered_words = re.sub(r'[^\w\s,]', '', filtered_words)
        filtered_words = re.sub(r'\s{2,}', ' ', filtered_words)
        prompt = filtered_words
        if prompt == '':
            white_image = Image.new('RGB', (512, 512), 'white')
            white_image.save(f"./you_path/{model}/sd_{image_name}.png")
            print("prompt=None,we get white_picture")
        else:
            # # xl_1.0
            # output = pipe({'text': prompt})
            # cv2.imwrite(f"./NER_data/sd_twitter2015/{model}/sd_{image_name}.png", output['output_imgs'][0])
            # xl_turbo
            print(prompt)
            image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
            image.save(f".you_path/{model}/sd_{image_name}.png")

    print("process done！")


if __name__ == '__main__':
    # Please replace it with the appropriate path yourself.
    train_15_dir = "./train_entity.txt"
    valid_15_dir = "./valid_entity.txt"
    test_15_dir = "./test_entity.txt"
    process_files(train_15_dir, model="train")
    process_files(valid_15_dir, model="valid")
    process_files(test_15_dir, model="test")
