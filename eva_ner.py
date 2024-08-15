# !/usr/bin/env python3

import os
from typing import List

import torch
from transformers import AutoTokenizer

from modelv1 import convert_inputs, get_bool_ids_greater_than, get_span
# from modelv1 import convert_inputs, get_bool_ids_greater_than, get_span

import json


def inference(
    model,
    tokenizer,
    device: str,
    contents: List[str], 
    prompts: List[str], 
    max_length=512, 
    prob_threshold=0.5
    ) -> List[str]:
    """
    输入 promot 和 content 列表，返回模型提取结果。    

    Args:
        contents (List[str]): 待提取文本列表, e.g. -> [
                                                    '《琅琊榜》是胡歌主演的一部电视剧。',
                                                    '《笑傲江湖》是一部金庸的著名小说。',
                                                    ...
                                                ]
        prompts (List[str]): prompt列表，用于告知模型提取内容, e.g. -> [
                                                                    '主语',
                                                                    '类型',
                                                                    ...
                                                                ]
        max_length (int): 句子最大长度，小于最大长度则padding，大于最大长度则截断。
        prob_threshold (float): sigmoid概率阈值，大于该阈值则二值化为True。

    Returns:
        List: 模型识别结果, e.g. -> [['琅琊榜'], ['电视剧']]
    """
    inputs = convert_inputs(tokenizer, prompts, contents, max_length=max_length)
    model_inputs = {
        'input_ids': inputs['input_ids'].to(device),
        'token_type_ids': inputs['token_type_ids'].to(device),
        'attention_mask': inputs['attention_mask'].to(device),
    }
    output_sp, output_ep = model(**model_inputs)
    output_sp, output_ep = output_sp.detach().cpu().tolist(), output_ep.detach().cpu().tolist()
    start_ids_list = get_bool_ids_greater_than(output_sp, prob_threshold)
    end_ids_list = get_bool_ids_greater_than(output_ep, prob_threshold)

    res = []                                                    # decode模型输出，将token id转换为span text
    offset_mapping = inputs['offset_mapping'].tolist()
    for start_ids, end_ids, prompt, content, offset_map in zip(start_ids_list, 
                                                            end_ids_list,
                                                            prompts,
                                                            contents,
                                                            offset_mapping):
        span_set = get_span(start_ids, end_ids)                 # e.g. {(5, 7), (9, 10)}
        current_span_list = []
        for span in span_set:
            if span[0] < len(prompt) + 2:                       # 若答案出现在promot区域，过滤
                continue
            span_text = ''                                      # 答案span
            input_content = prompt + content                    # 对齐token_ids
            for s in range(span[0], span[1] + 1):               # 将 offset map 里 token 对应的文本切回来
                span_text += input_content[offset_map[s][0]: offset_map[s][1]]
            current_span_list.append(span_text)
        res.append(current_span_list)
    return res


def event_extract_example(
    model,
    tokenizer,
    device: str,
    sentence: str, 
    schema: dict, 
    prob_threshold=0.6,
    max_seq_len=128,
    ) -> dict:
    """
    UIE事件抽取示例。

    Args:
        sentence (str): 待抽取句子, e.g. -> '5月17号晚上10点35分加班打车回家，36块五。'
        schema (dict): 事件定义字典, e.g. -> {
                                            '加班触发词': ['时间','地点'],
                                            '出行触发词': ['时间', '出发地', '目的地', '花费']
                                        }
        prob_threshold (float, optional): 置信度阈值（0~1），置信度越高则召回结果越少，越准确。
    
    Returns:
        dict -> {
                '触发词1': {},
                '触发词2': {
                    '事件属性1': [属性值1, 属性值2, ...],
                    '事件属性2': [属性值1, 属性值2, ...],
                    '事件属性3': [属性值1, 属性值2, ...],
                    ...
                }
            }
    """
    rsp = {}
    trigger_prompts = list(schema.keys())

    for trigger_prompt in trigger_prompts:
        rsp[trigger_prompt] = {}
        triggers = inference(
            model,
            tokenizer,
            device,
            [sentence], 
            [trigger_prompt], 
            max_length=128, 
            prob_threshold=prob_threshold)[0]
        
        for trigger in triggers:
            if trigger:
                arguments = schema.get(trigger_prompt)
                contents = [sentence] * len(arguments)
                prompts = [f"{trigger}的{a}" for a in arguments]
                res = inference(
                    model,
                    tokenizer,
                    device,
                    contents, 
                    prompts,
                    max_length=max_seq_len, 
                    prob_threshold=prob_threshold)
                for a, r in zip(arguments, res):
                    rsp[trigger_prompt][a] = r
    print('[+] Event-Extraction Results: ', rsp)


def information_extract_example(
    model,
    tokenizer,
    device: str,
    sentence: str, 
    schema: dict, 
    prob_threshold=0.5, 
    max_seq_len=256
    ) -> dict:
    """
    UIE信息抽取示例。

    Args:
        sentence (str): 待抽取句子, e.g. -> '麻雀是几级保护动物？国家二级保护动物'
        schema (dict): 事件定义字典, e.g. -> {
                                            '主语': ['保护等级']
                                        }
        prob_threshold (float, optional): 置信度阈值（0~1），置信度越高则召回结果越少，越准确。
    
    Returns:
        dict -> {
                '麻雀': {
                        '保护等级': ['国家二级']
                    },
                ...
            }
    """
    rsp = {}
    subject_prompts = list(schema.keys())

    for subject_prompt in subject_prompts:
        subjects = inference(
            model,
            tokenizer,
            device,
            [sentence], 
            [subject_prompt], 
            max_length=256, 
            prob_threshold=prob_threshold)[0]
        
        for subject in subjects:
            if subject:
                rsp[subject] = {}
                predicates = schema.get(subject_prompt)
                contents = [sentence] * len(predicates)
                prompts = [f"{subject}的{p}" for p in predicates]
                res = inference(
                    model,
                    tokenizer,
                    device,
                    contents, 
                    prompts,
                    max_length=max_seq_len, 
                    prob_threshold=prob_threshold
                )
                for p, r in zip(predicates, res):
                    rsp[subject][p] = r
    # print('[+] Information-Extraction Results: ', rsp)
    # print(type(rsp))
    return rsp


def ner_example(
    model,
    tokenizer,
    device: str,
    sentence: str, 
    schema: list, 
    prob_threshold=0.6
    ) -> dict:
    """
    UIE做NER任务示例。

    Args:
        sentence (str): 待抽取句子, e.g. -> '5月17号晚上10点35分加班打车回家，36块五。'
        schema (list): 待抽取的实体列表, e.g. -> ['出发地', '目的地', '时间']
        prob_threshold (float, optional): 置信度阈值（0~1），置信度越高则召回结果越少，越准确。
    
    Returns:
        dict -> {
                实体1: [实体值1, 实体值2, 实体值3...],
                实体2: [实体值1, 实体值2, 实体值3...],
                ...
            }
    """
    rsp = {}
    sentences = [sentence] * len(schema)    #  一个prompt需要对应一个句子，所以要复制n遍句子
    res = inference(
        model,
        tokenizer,
        device,
        sentences, 
        schema, 
        max_length=128, 
        prob_threshold=prob_threshold)
    for s, r in zip(schema, res):
        rsp[s] = r
    # print('[+] NER Results: ', rsp)
    return rsp

def calculate_metrics(tp, fp, fn):
    # tp = sum(1 for x in pred if x in real)
    # fp = sum(1 for x in pred if x not in real)
    # fn = sum(1 for x in real if x not in pred)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1

if __name__ == "__main__":
    from rich import print

    device = 'cuda:0'                                       # 指定GPU设备
    saved_model_path = '/2014110093/transformers_tasks/UIE-SIAIB/checkpoints/diakg2-all2/model_best'     # 训练模型存放地址
    # saved_model_path = '/2014110093/transformers_tasks/UIE-SIAIB/checkpoints/UIE-SIAIB-ONLYBRANCH-/model_best'     # 训练模型存放地址

    tokenizer = AutoTokenizer.from_pretrained(saved_model_path) 
    model = torch.load(os.path.join(saved_model_path, 'model.pt'))
    model.to(device).eval()

    json_file_path = "data/diakg/hot.jsonl"
    # json_file_path = "/2014110093/transformers_tasks/UIE-SIAIB/data/UIE-SIAIB-TEST/new_dev.json"
    # schema_list = [
    #     '作物',
    #     '病害',
    #     '病原',
    #     '病体',
    #     '虫害',
    #     '虫体',
    #     '时间',
    #     '场所',
    #     '地区',
    #     '行动',
    #     '外部因素',
    #     '症状',
    #     '后果'
    # ]
    schema_list = [
        "疾病",
        "疾病分期分型",
        "病因",
        "发病机制",
        "临床表现",
        "检查方法",
        "检查指标",
        "检查指标值",
        "药物名称",
        "用药频率",
        "用药剂量",
        "用药方法",
        "非药治疗",
        "手术",
        "不良反应",
        "部位",
        "程度",
        "持续时间",
    ]
    tp_all = 0
    fp_all = 0
    fn_all = 0
    with open(json_file_path, 'r', encoding='utf-8') as file:

        for line in file:
            data = json.loads(line)
            text = data['text']
            relations = data['relations']
            entities = data['entities']

            ent_list = []
            for ent in entities:
                ent_id = ent['id']
                ent_start = ent['start_offset']
                ent_end = ent['end_offset']
                ent_text = text[ent_start:ent_end]
                ent_type = ent['label']
                ent_final = {'id':ent_id, 'text':ent_text, 'label': ent_type}
                ent_list.append(ent_final)


            relation_list = []
            for relation in relations:
                start_idx = relation['from_id']
                end_idx = relation['to_id']
                start = ''
                end = ''
                for entity in ent_list:
                    if start_idx == entity['id']:
                        start = entity['text']
                    if end_idx == entity['id']:
                        end = entity['text']
                re_type = relation['type']
                relation_final = (start, re_type, end)
                relation_list.append(relation_final)
            # print(text)
            # print('real',relation_list)
            
            # ner_list = []
            print(text)
            
            pred = ner_example(
                model,
                tokenizer,
                device,
                sentence=text, 
                schema=schema_list
            )
            real = ent_list
            print('pred', pred)
            # print('real', ent_list)

            # new_real = {'作物': [],
            #     '病害': [],
            #     '病原': [],
            #     '病体': [],
            #     '虫害': [],
            #     '虫体': [],
            #     '时间': [],
            #     '场所': [],
            #     '地区': [],
            #     '行动': [],
            #     '外部因素': [],
            #     '症状': [],
            #     '后果': []}
            new_real = {
                "疾病" : [],
                "疾病分期分型" : [],
                "病因" : [],
                "发病机制" : [],
                "临床表现" : [],
                "检查方法" : [],
                "检查指标" : [],
                "检查指标值" : [],
                "药物名称" : [],
                "用药频率" : [],
                "用药剂量" : [],
                "用药方法" : [],
                "非药治疗" : [],
                "手术" : [],
                "不良反应" : [],
                "部位" : [],
                "程度" : [],
                "持续时间" : [],
            }

            for lab in schema_list:
                for re in real:
                    if re['label'] == lab:
                        print(re)
                        new_real[lab].append(re['text']) 
            
            real = new_real
            print('real', new_real)
            tp_ = 0
            fp_ = 0
            fn_ = 0
            for i in schema_list:
                tp = sum(1 for x in pred[i] if x in real[i])
                fp = sum(1 for x in pred[i] if x not in real[i])
                fn = sum(1 for x in real[i] if x not in pred[i])
                tp_ += tp
                fp_ += fp
                fn_ += fn
            tp_all += tp_
            fp_all += fp_
            fn_all += fn_
            precision, recall, f1 = calculate_metrics(tp_, fp_, fn_)
            pre_all, re_all, f1_all = calculate_metrics(tp_all, fp_all, fn_all)
            print('sub_precision:',precision, 'sub_recall:', recall, 'sub_f1:', f1)
            print('all_precision:',pre_all, 'all_recall:', re_all, 'all_f1:', f1_all)
            
                
                
