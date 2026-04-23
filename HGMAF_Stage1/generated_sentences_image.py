# -*- coding: utf-8 -*-

"""
@description  : generated_sentences_image_demo
@Author       : lsx
@Email        : lsx314159@163.com
"""
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


def process_files(_dir, model="train"):
    load_file = _dir
    with open(load_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        raw_words, raw_targets = [], []
        raw_word, raw_target = [], []
        imgs = []
        for line in lines:
            if line.startswith("IMGID:"):
                img_id = line.strip().split('IMGID:')[1] + '.jpg'
                imgs.append(img_id)
                continue
            if line != "\n":
                raw_word.append(line.split('\t')[0])
                label = line.split('\t')[1][:-1]
                if 'OTHER' in label:
                    label = label[:2] + 'MISC'
                raw_target.append(label)
            else:
                raw_words.append(raw_word)
                raw_targets.append(raw_target)
                raw_word, raw_target = [], []

    assert len(raw_words) == len(raw_targets) == len(imgs), f"{len(raw_words)}, {len(raw_targets)}, {len(imgs)}"
    entity_list = [[word for word, target in zip(words, targets) if not target == 'O'] for words, targets in
                   zip(raw_words, raw_targets)]

    for image, words in tqdm(zip(imgs, entity_list), total=len(raw_targets)):
        image_name, _ = os.path.splitext(image)
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
            white_image.save(f".you_path/{model}/sd_{image_name}.png")
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
    train_15_dir = "./train.txt"
    valid_15_dir = "./valid.txt"
    test_15_dir = "./test.txt"
    process_files(train_15_dir, model="train")
    process_files(valid_15_dir, model="valid")
    process_files(test_15_dir, model="test")
