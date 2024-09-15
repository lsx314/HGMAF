# -*- coding: utf-8 -*-

"""
@description  : entity interpretation prompt demo
@Author       : lsx
@Email        : lsx314159@163.com
"""
import os
import json

from openai import OpenAI

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    # You can obtain the api by yourself. For details, see the openai official website: https://openai.com/api/
    # 请自行获取api，操作方法参考open ai官网：https://openai.com/api/
    api_key="",
    base_url=""
)


# 非流式响应
def gpt_35_api(messages: list):
    """为提供的对话消息创建新的回答

    Args:
        messages (list): 完整的对话消息
    """
    completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    print(completion.choices[0].message.content)


def gpt_35_api_stream(messages: list):
    """为提供的对话消息创建新的回答 (流式传输)

    Args:
        messages (list): 完整的对话消息
    """
    stream = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=messages,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="")
            return chunk.choices[0].delta.content


def process_text(file):
    _processed_text = []
    with open(file, "r", encoding="utf-8") as f:
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

    for raw_word in raw_words:
        filtered_list = [e for e in raw_word if
                         e != 'RT' and e != ":" and e != "_" and 'http' not in e and \
                         '@' not in e and "//t" not in e and "co/" not in e]
        _processed_text.append(filtered_list)

    return _processed_text, imgs


def process_entity(es_dir):
    e_list = []
    for ed in os.listdir(es_dir):
        with open(ed, 'r', encoding='utf-8') as file:
            data = json.load(file)
            # 提取"entity"的值
            # 我们采用正则匹配处理无法提取的实体，或者prompt不正确的prompt output
            e_list.append(data.get("entity", "-100"))
    return e_list


def main(o_data_dir, p_data_dir, e_dir):
    texts, item_id = process_text(o_data_dir)
    entity_list = process_entity(e_dir)
    # 非流式调用
    # gpt_35_api(messages)
    for text, sent_id, entity in zip(texts, item_id, entity_list):
        sent_id = sent_id.rstrip(".jpg")
        sent = ' '.join(text)
        # Prompt demo
        content = f"""As a graduate student specializing in entity extraction, I request you to assume the role of an 
        experienced linguistics expert to perform entity extraction on the following input statement. This statement 
        comes from the Twitter platform, reflecting content shared by users on the platform. Your task is to thoroughly
         analyze the original statement and identify its entities. Please ignore any text following the '@' symbol .
         There are only four types of entities: PER, LOC, ORG, and OTHER(Different Type for different Datasets).
         Output format as follows：
         {{
          "input" : " {sent}",
         "output" : {{
         "Entity" : "{entity}"
             }}
          }}
         """
        prompt_emotion = [{'role': 'user', 'content': content}, ]
        # 流式调用，获取内容
        gpt_response = gpt_35_api_stream(prompt_emotion)
        #
        output_file_path = f"{p_data_dir}/{sent_id}.txt"
        #
        with open(output_file_path, 'w', encoding='utf-8') as file:
            file.write(gpt_response)

        print(f"{sent_id} has process {output_file_path}")


if __name__ == '__main__':
    original_data_dir = "."
    processed_data_dir = ""
    e_dir = ""

    main(original_data_dir, processed_data_dir, e_dir)
