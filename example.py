### Code for auto annotation of ontology using GPT-3.5+ and few-shot learning
### contact : aria1th@github / etc
### 2023-06-09
### TODO : model selection exposure, local model handling, Async or multi-threading for faster annotation (maybe 20/s?)

import openai
import pandas as pd
import numpy as np
import re
from collections import defaultdict
import time
from tqdm import tqdm
import csv
import os
import json

#openai에서 발급받은 api key를 입력 → https://platform.openai.com/account/api-keys 참조
openai.api_key = input("Input API Key")
# defined as function and returns the result

def getChatCompletionFromTemplate(template:str, inputs:str):
    completion = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": template},
            {"role": "user", "content": inputs}
        ],
        #max_tokens = 100,
        temperature = 0.05,
    )
    return completion

# for generating ids
CACHE = defaultdict(dict) 

# Dictionary for translating keys
en_to_kor = {
    "text": "텍스트",
    "entities": "개체들",
    "id": "순차",
    "label": "레이블",
    "start_offset": "시작_오프셋",
    "end_offset": "끝_오프셋",
    "relations": "관계들",
    "type": "유형",
    "head": "머리",
    "tail": "꼬리",
    "result": "결과",
    "annotated_text" : "개체_확인_텍스트",

}
# inverse dictionary, Korean to English
convert_table = {v:k for k, v in en_to_kor.items()}

def translate_keys_recursive(iterable):
    # using convert_table, translate keys of iterable recursively
    if isinstance(iterable, dict):
        return {convert_table.get(k, k): translate_keys_recursive(v) for k, v in iterable.items()}
    elif isinstance(iterable, list):
        return [translate_keys_recursive(element) for element in iterable]
    else:
        return iterable
    

# English is better for GPT-3.5+ but for local korean models, its made with korean
# example : Polyglot-ko-5.8B, kullm 12.8b, etc for local

# For local models, use llama-cpp-python server or langchain



few_shot_template = r"""
Prompt: "당신은 주어진 자연어 문장에서 주어진 엔티티 레이블 및 관계 레이블을 사용해 개체와 관계를 추출해야 합니다."

엔티티는 반드시 다음 중 정확히 하나의 레이블만을 가져야 합니다.

엔티티 레이블 타입:
["해군조직", "기타조직", "군인", "군사계급", "군무원", "예비역", "민간인", "사관생도", "사람", "부사관", "준사관", "장교", "병과", "직책", "임무"]

관계는 반드시 다음 중 정확히 하나의 레이블만을 가져야 합니다.
관계 레이블은 엔티티 레이블과 겹치지 않습니다.
관계 레이블 타입:
["hasSubOrganization", "memberOf", "orgCanControlByPosition", "orgCanControlledByPerson","orgCanControlPerson", "hasRank"]

형식은 json 형식으로 주의를 기울여 작성해야 합니다.
반드시 다음과 같은 형식을 따라야 합니다.
형식:
{
    "개체_확인_텍스트" : "%<텍스트_내_레이블>%(레이블)",
    "개체들": [
        {
            "순차": "<숫자>",
            "레이블": "<엔티티_레이블>"
        },
        {

        }
    ],
    "관계들": [
        {
            "순차": "<숫자>",
            "유형": "<사전_정의_관계_유형>",
            "머리": "<텍스트_내_레이블>",
            "꼬리": "<텍스트_내_레이블>"
        },
        {

        }
    ]
}

#예시
입력 : "박성현 주무관은 국방부에서 근무하고 있습니다."

결과 : {
    "개체_확인_텍스트" : "%박성현%(사람) %주무관%(군무원)은 %국방부%(해군조직)에서 근무하고 있습니다.",
    "개체들" : [
        {
            "순차": 1,
            "레이블": "사람"
        },
        {
            "순차": 2,
            "레이블": "군무원"
        },
        {
            "순차": 3,
            "레이블": "해군조직"
        }
    ],
    "관계들" : [
        {
            "순차": 1,
            "유형": "hasRank",
            "머리": "박성현",
            "꼬리": "군무원"
        },
        {
            "순차": 2,
            "유형": "orgCanControlPerson",
            "머리": "국방부",
            "꼬리": "군무원"
        },
        {
            "순차": 3,
            "유형": "memberOf",
            "머리": "박성현",
            "꼬리": "국방부"
        }
    ]
}

입력: 
    
    """

# one shot currently


def create_relations(entity_list:list, relation_found:list) -> list:
    """
    create relations from entity_list and relation_found
    param entity_list : [{"id" : int, "in_text_name" : entity_original_name, "label" : entity_name, "start_offset" : int, "end_offset" : int}, ...]
    param relation_found : [{"id" : int, "type" : relation_name, "head" : entity_original_name, "tail" : entity_original_name}, ...]
    return : [{"id" : int, "type" : relation_name, "from_id" : int, "to_id" : int}, ...]
    """
    # relation_found : [{"id" : int, "type" : relation_name, "head" : entity_original_name, "tail" : entity_original_name}, ...]
    # result : [{"id" : int, "type" : relation_name, "from_id" : int, "to_id" : int}, ...]
    # entity_list : [{"id" : int, "in_text_name" : entity_original_name, "label" : entity_name, "start_offset" : int, "end_offset" : int}, ...]
    # from_id and to_id is found from entity_list by entity_original_name matching
    result = []
    for relation in relation_found:
        head_entity = relation["head"]
        tail_entity = relation["tail"]
        head_id = -1
        tail_id = -1
        for entity in entity_list:
            if entity["in_text_name"] == head_entity:
                head_id = entity["id"]
            elif entity["in_text_name"] == tail_entity:
                tail_id = entity["id"]
            # if found, break
            if head_id != -1 and tail_id != -1:
                break
        result_dict = {"id" : relation["id"], "type" : relation["type"], "from_id" : head_id, "to_id" : tail_id}
        if head_id == -1:
            result_dict["inferred_head_name"] = head_entity
        if tail_id == -1:
            result_dict["inferred_tail_name"] = tail_entity
        result.append(result_dict)
        # if head or tail is not found, add "head_name" or "tail_name" to result

    return result


    # result is "%박성현%(사람) %주무관%(군무원)은 %국방부%(해군조직)에서 근무하고 있습니다." - like format
    # detect entity + category from result %박성현%(사람) -> 박성현, 사람, start_offset, end_offset
def detect(annotated_text:str) -> list:
    """
    detect entity and category from annotated_text
    example: "%박성현%(사람) %주무관%(군무원)은 %국방부%(해군조직)에서 근무하고 있습니다."
    result_list = [["박성현", "사람", 0, 4], ["주무관", "군무원", 6, 10], ["국방부", "해군조직", 13, 17]]
    :param annotated_text: annotated text
    :return: list of entity, category, start_offset, end_offset
    """
    # detect entity
    entities = re.findall(r"%(.+?)%", annotated_text)
    # detect category
    categories = re.findall(r"\((.+?)\)", annotated_text)
    # check if categories is multilabeled, if so, split and select first category
    for i in range(len(categories)):
        if len(categories[i].split(",")) > 1:
            categories[i] = categories[i].split(",")[0]
    # make list of entity and category
    result = []
    detected_end_offsets = {} # store end_offset of previous entity
    for entity, category in zip(entities, categories):
        # detect start_offset, end_offset
        # if there are multiple entities, use previous entity's end_offset as start_offset
        start_offset = annotated_text.find(entity, detected_end_offsets.get(entity, 0))
        end_offset = start_offset + len(entity)
        result.append([entity, category, start_offset, end_offset])
    return result

# using defaultdict, CACHE = {
#   "url" : {
#    "entity_id" : int,},
# }


# for entity, relation, sentence, make function

entity_id = lambda url: get_by_arg(url, "entity_id")
relation_id = lambda url: get_by_arg(url, "relation_id")
sentence_id = lambda url: get_by_arg(url, "sentence_id")


# {article_url : article_id} incrementally added
ARTICLES = {}

# Debug flag, if used, it will use fixed result instead of calling openAI

DEBUG = False
DEBUG_COMPLETIONS = True

def get_by_arg(url, arg) -> int: # generalized version
    """
    get <arg> id from url
    :param url: url of entity
    :param arg: arg name
    :return: arg id

    :example: get_by_arg("https://ko.wikipedia.org/wiki/박성현", "entity_id") -> 1
    """
    arg_id = CACHE[url].get(arg, 0)
    if arg_id == 0:
        CACHE[url][arg] = 1
        return 0
    else:
        CACHE[url][arg] += 1
        return arg_id 


def get_article_id(url) -> int:
    if url in ARTICLES:
        return ARTICLES[url]
    else:
        ARTICLES[url] = len(ARTICLES)
        return ARTICLES[url]


debug_completion_path = "./debug_completion.jsonl"
def debug_completions(value:dict):
    """
    write debug completion to debug_completion_path
    :param value: value to write
    """
    if not DEBUG_COMPLETIONS:
        return
    with open(debug_completion_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")

def getAnalyzation(text:str, base_url:str, article_index = -1) -> dict:
    """
    get analyzation result from text
    :param text: text to analyze
    :param base_url: base url of text 
    :return: analyzation result

    :warning: base_url should be specified, because it is used to make url of entity
    :warning: article_index should be modified externally.

    :example: getAnalyzation("박성현은 국방부에서 근무하고 있습니다.", url)
    """
    if article_index == -1:
        article_index = get_article_id(base_url)
    try:
      if DEBUG:
          result = "\uacb0\uacfc : {\n    \"\uac1c\uccb4_\ud655\uc778_\ud14d\uc2a4\ud2b8\" : \"%\uae40\ub3d9\ud658%(\uc0ac\ub78c, \uad70\uc0ac\uacc4\uae09) %\uc548\uc911\uadfc\ud568\uc7a5%(\uad70\ubb34\uc6d0)\uc740 \u201c%\uc548\uc911\uadfc%(\uc0ac\ub78c, \uc758\uc0ac) \uc758 \ub73b\uc744 \uc774\uc5b4\ubc1b\uc740 %\uc548\uc911\uadfc\ud568%(\ud574\uad70\uc870\uc9c1)\uc740 \uad6d\uac00\uc548\ubcf4\uc758 \ud575\uc2ec \ube44\uc218\ub85c\uc11c \uc5b8\uc81c\u00b7\uc5b4\ub514\uc11c\ub4e0 \uc801\uc744 \uaca9\uce68\ud560 \uc218 \uc788\ub294 \uc804\ud22c\uc900\ube44\ud0dc\uc138\ub97c \uc644\ube44\ud588\ub2e4\u201d\uba70 \u201c\uc548 \uc758\uc0ac\uac00 \uac15\uc870\ud588\ub358 \uc704\uad6d\ud5cc\uc2e0 \uad70\uc778\ubcf8\ubd84\uc758 \uc790\uc138\ub97c \uacac\uc9c0\ud574 \uc2f8\uc6b0\uba74 \ubc18\ub4dc\uc2dc \uc774\uae30\ub294 \ud544\uc2b9\ud574\uad70 \uc804\ud1b5\uc744 \uc774\uc5b4\uac00\uaca0\ub2e4\u201d\uace0 \ub9d0\ud588\ub2e4.\u00a0\",\n    \"\uac1c\uccb4\ub4e4\" : [\n        {\n            \"\uc21c\ucc28\": 1,\n            \"\ub808\uc774\ube14\": \"\uc0ac\ub78c\"\n        },\n        {\n            \"\uc21c\ucc28\": 2,\n            \"\ub808\uc774\ube14\": \"\uad70\uc0ac\uacc4\uae09\"\n        },\n        {\n            \"\uc21c\ucc28\": 3,\n            \"\ub808\uc774\ube14\": \"\uad70\ubb34\uc6d0\"\n        },\n        {\n            \"\uc21c\ucc28\": 4,\n            \"\ub808\uc774\ube14\": \"\uc0ac\ub78c\"\n        },\n        {\n            \"\uc21c\ucc28\": 5,\n            \"\ub808\uc774\ube14\": \"\uc758\uc0ac\"\n        },\n        {\n            \"\uc21c\ucc28\": 6,\n            \"\ub808\uc774\ube14\": \"\ud574\uad70\uc870\uc9c1\"\n        }\n    ],\n    \"\uad00\uacc4\ub4e4\" : [\n        {\n            \"\uc21c\ucc28\": 1,\n            \"\uc720\ud615\": \"hasRank\",\n            \"\uba38\ub9ac\": \"\uae40\ub3d9\ud658\",\n            \"\uaf2c\ub9ac\": \"\uad70\uc0ac\uacc4\uae09\"\n        },\n        {\n            \"\uc21c\ucc28\": 2,\n            \"\uc720\ud615\": \"orgCanControlByPosition\",\n            \"\uba38\ub9ac\": \"\uc548\uc911\uadfc\ud568\",\n            \"\uaf2c\ub9ac\": \"\uad70\ubb34\uc6d0\"\n        },\n        {\n            \"\uc21c\ucc28\": 3,\n            \"\uc720\ud615\": \"hasSubOrganization\",\n            \"\uba38\ub9ac\": \"\uc548\uc911\uadfc\",\n            \"\uaf2c\ub9ac\": \"\uc548\uc911\uadfc\ud568\"\n        },\n        {\n            \"\uc21c\ucc28\": 4,\n            \"\uc720\ud615\": \"orgCanControlPerson\",\n            \"\uba38\ub9ac\": \"\uc548\uc911\uadfc\ud568\",\n            \"\uaf2c\ub9ac\": \"\ud574\uad70\uc870\uc9c1\"\n        }\n    ]\n}"
      else:
        result = getChatCompletionFromTemplate(few_shot_template, text)
    except Exception as e:
      # if KeyboardInterrupt, raise again
      if isinstance(e, KeyboardInterrupt):
        raise e
      print("RateLimitError")
      return None
    #print(result.choices[0].text)
    debug_completions(result)
    if not DEBUG:
        result = result.choices[0]["message"]["content"]
    # start from { to }
    result = result[result.find("{"):result.rfind("}")+1]
    # using ast.literal_eval parse string to dict
    import ast
    result_dict = ast.literal_eval(result)
    assert type(result_dict) == dict, "result was not valid json format"
    # convert keys and values by given dictionary if found, else use original value

    annotation_dict = translate_keys_recursive(result_dict)
    # add original text to annotation_dict
    annotation_dict["text"] = text
    # convert list of entity, category, start_offset, end_offset to dictionary
    # using result, get list first
    entity_list = detect(annotation_dict["annotated_text"])
    # convert list to dictionary
    # "entities":[{“id”:0,“label”:군무원,“start_offset”:#단어시작위치#,“end_offset”:#단어마지막위치#},…]
    annotation_dict["entities"] = [{"id": entity_id(base_url), "in_text_name" : entity[0], "label": entity[1], "start_offset": entity[2], "end_offset": entity[3]} for i, entity in enumerate(entity_list)]
    # now use relation from annotation_dict["relations"]
    # use create_relations(entity_list:list[dict], relation_found:list) -> list:
    relations = create_relations(annotation_dict["entities"], annotation_dict["relations"])
    if DEBUG:
      print(relations)
      # [{'id': 1, 'type': 'hasRank', 'from_id': 0, 'to_id': -1}]
    annotated_relation_dict = []
    for relation in relations:
        relation_obj = {}
        for keys in relation:
            if keys not in {"type", "from_id", "to_id"}:
                relation_obj[keys] = relation[keys]
        relation_obj["id"] = relation_id(base_url)
        relation_obj["label"] = relation["type"]
        relation_obj["head"] = relation["from_id"]
        relation_obj["tail"] = relation["to_id"]
        annotated_relation_dict.append(relation_obj)
    # now add article id and local sentence id to annotation_dict
    annotation_dict["relations"] = annotated_relation_dict
    annotation_dict["article_id"] = article_index
    annotation_dict["local_sentence_id"] = sentence_id(base_url)
    return annotation_dict


# pipeline
# for each (sentence, url) pair, get annotation
# then append or save as csv

def process_file_to_jsonl(file_dir : str, result_file_name: str, columns = [1,3], limit = -1, use_rows = False, target_rows = (1,)):
    """
    process file to jsonl format (listed json)
    :param file_dir: file directory(str of csv)
    :param result_file_name: result file name(can be string or file.jsonl)
    :param columns: rows to use (for example, if columns = [1,3], then use column 1 and 3 as url and sentence)
    :param limit: limit of rows to process
    :param use_rows: use rows or not
    :param target_rows: rows to use
    """
    # check if result_file_name ends with .jsonl
    if not result_file_name.endswith(".jsonl"):
        result_file_name += ".jsonl"

    SKIPPED_ROWS = []
    
    with open(result_file_name, "w", encoding="utf-8") as resultfile:
        with open(file_dir, "r", encoding="utf-8") as readfile:
            total_rows = sum(1 for row in readfile)
        with open(file_dir, "r", encoding="utf-8") as readfile:
            reader = csv.reader(readfile)
            # skip header
            next(reader)
            # for each row, get annotation
            for i, row in tqdm(enumerate(reader), total = limit if limit != -1 else total_rows - 1):
                if use_rows:
                    if i not in target_rows:
                        continue
                # columns = [1,3] -> select idx 0 = 1, idx 1 = 3 -> url, sentence
                url = row[columns[0]]
                sentence = row[columns[1]]
                # check if sentence or url is empty
                if sentence == "" or url == "":
                    continue
                # get annotation
                try:
                    annotation = getAnalyzation(sentence, url)
                except Exception as e:
                    # if KeyboardInterrupt, raise again
                    if isinstance(e, KeyboardInterrupt):
                        print("Annotated {} rows".format(i))
                        return SKIPPED_ROWS
                    print("Failed annotation at row {}".format(i))
                    SKIPPED_ROWS.append(i)
                    annotation = None
                if annotation is not None:
                    # append to list
                    #result.append(annotation)
                    # write to file
                    json.dump(annotation, resultfile, ensure_ascii=False)
                    resultfile.write("\n")
                else:
                    # sleep 50ms
                    time.sleep(0.05)
                    print("Failed annotation at row {}".format(i))
                # debug
                if DEBUG:
                    print(annotation)
                    if i > limit:
                        break
                if limit != -1 and i > limit:
                    break
    return SKIPPED_ROWS

# TODO : Add argparse
if __name__ == "__main__":
    # Path to file
    path = r"PATH_TO_FILE_DIR"
    file_name = "FILE_NAME.csv"
    failed_rows = process_file_to_jsonl(os.path.join(path, file_name), os.path.join(path, "result.jsonl"))
    import pickle
    with open(os.path.join(path, "failed_rows.pkl"), "wb") as f:
        pickle.dump(failed_rows, f)







    
