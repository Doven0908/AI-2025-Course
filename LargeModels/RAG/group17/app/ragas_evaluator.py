from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .config import Settings
from .logger import get_logger
from .pipeline import RAGPipeline

logger = get_logger(__name__)

try:
    from langchain_core.embeddings import Embeddings
    from langchain_core.outputs import LLMResult
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover
    ChatOpenAI = None
    Embeddings = None
    LLMResult = None

if Embeddings is None:
    class Embeddings:  # type: ignore

        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            raise NotImplementedError("Embeddings backend not available")

        def embed_query(self, text: str) -> List[float]:
            raise NotImplementedError("Embeddings backend not available")

if LLMResult is None:
    class LLMResult:  # type: ignore
        pass

try:
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper as _DeprecatedLangchainLLMWrapper
    from ragas.llms.base import LangchainLLMWrapper as _BaseLangchainLLMWrapper
    from ragas.metrics import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
except ImportError:  # pragma: no cover
    EvaluationDataset = None
    _DeprecatedLangchainLLMWrapper = None
    _BaseLangchainLLMWrapper = None
    evaluate = None
    AnswerRelevancy = ContextPrecision = ContextRecall = Faithfulness = None

LangchainLLMWrapper = _DeprecatedLangchainLLMWrapper or _BaseLangchainLLMWrapper
LangchainLLMWrapperBase = _BaseLangchainLLMWrapper


DEFAULT_DATASET: List[Dict[str, object]] = [
     {
        "question": "根据2022版《中国居民膳食指南》，“食物多样，合理搭配”具体要求：每天膳食至少应包括哪些主要食物类别？从一周角度看，平均每天和每周分别建议吃多少种不同的食物？",
        "user_input": "根据2022版膳食指南说明：每天应该吃哪些大类食物？一周内每天和每周各要吃多少种食物才算食物多样？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "膳食指南提出每天饮食应至少包括谷薯类、蔬菜、水果、畜禽鱼蛋奶和大豆等几大类食物；在种类数量上，平均每天应摄入12种以上不同食物，每周应达到25种以上不同食物，以保证膳食多样化。"
        ],
        "top_k": 5,
    },
    {
        "question": "根据2022版膳食指南，成年人每日谷类食物、全谷物和杂豆以及薯类的推荐摄入量分别是多少？",
        "user_input": "成年人每天主食应该吃多少？其中谷类、全谷物和杂豆、薯类各自的推荐摄入量是多少？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "膳食指南推荐成年人每天摄入谷类食物总量约200～300克，其中全谷物和杂豆应占50～150克；薯类每天建议摄入50～100克，可折算计入主食总量。"
        ],
        "top_k": 5,
    },
    {
        "question": "在“多吃蔬果、奶类、全谷、大豆”的核心推荐中，成年人每天新鲜蔬菜和水果各应吃多少？深色蔬菜有什么比例要求？每天奶类大致要摄入多少，果汁能不能代替新鲜水果？",
        "user_input": "成年人每天应该吃多少蔬菜、水果和奶？深色蔬菜要占多少比例？果汁可以代替新鲜水果吗？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "指南建议成年人每天新鲜蔬菜不少于300克，其中深色蔬菜约占蔬菜总量的1/2；新鲜水果每天摄入量约200～350克，提倡天天吃水果；奶及奶制品每日摄入量相当于300毫升以上液态奶。果汁不能代替鲜果，应优先选择完整水果。"
        ],
        "top_k": 5,
    },
    {
        "question": "根据“适量吃鱼、禽、蛋、瘦肉”的建议，成年人平均每天鱼禽蛋瘦肉合计应吃多少克？从一周来看，鱼类、蛋类和畜禽肉分别推荐多少？在吃鸡蛋和选择肉类方面，指南还有哪些特别强调？",
        "user_input": "鱼、禽、蛋和瘦肉每天、每周分别推荐吃多少？鸡蛋和肉类选择上有什么特别提示？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "膳食指南建议鱼、禽、蛋和瘦肉合计，平均每天摄入约120～200克；按周计算，鱼类每周至少吃2次或约300～500克，蛋类每周约300～350克，畜禽肉每周约300～500克。指南强调鸡蛋营养丰富，吃鸡蛋不弃蛋黄；同时提倡优先选择鱼和瘦肉，少吃肥肉、烟熏腌制肉及各类深度加工肉制品。"
        ],
        "top_k": 5,
    },
    {
        "question": "在“少盐少油，控糖限酒”的建议中，成年人每天食盐、烹调油、添加糖和反式脂肪酸的摄入上限分别是多少？哪些人群不应饮酒？如果饮酒，每日酒精摄入量应该控制在多少克以内？",
        "user_input": "膳食指南中，盐、油、糖、反式脂肪酸和酒精分别有哪些具体控制标准？谁不应该喝酒？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "指南提出成年人每天食盐摄入量不超过5克，烹调油25～30克；添加糖每日不超过50克，最好控制在25克以下；反式脂肪酸每日不超过2克。儿童青少年、孕妇、乳母以及慢性病患者不应饮酒；如成年人饮酒，建议一天摄入的酒精量不超过15克。"
        ],
        "top_k": 5,
    },
    {
        "question": "所谓“平衡膳食模式”在食物结构和营养供能比例上有哪些重要特点？特别是碳水化合物和脂肪的供能比大致处于什么范围？",
        "user_input": "平衡膳食模式强调什么样的食物结构？碳水和脂肪分别应该提供多少比例的能量？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "平衡膳食模式以植物性食物为主、动物性食物为辅，要求食物种类齐全、比例合理，包括充足的谷类、蔬菜水果、奶类和大豆等，并强调少油、少盐、少糖。在营养供能比例上，碳水化合物供能比约为50%～65%，脂肪供能比约为20%～30%，蛋白质和其他营养素供应充足且平衡，有利于预防营养不足和营养过剩相关疾病。"
        ],
        "top_k": 5,
    },
    {
        "question": "文中将无花果称为“树上的糖包子”。新鲜无花果的含糖量范围、GI和GL大约是多少？血糖控制不佳的人群应如何食用？在膳食纤维、钾和硒方面无花果有什么营养优势？",
        "user_input": "无花果的含糖量、GI、GL分别大概是多少？糖尿病或血糖控制不好的人能怎么吃？它在膳食纤维、钾和硒方面有什么特点？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "新鲜无花果含糖量约16%～20%，GI约为60、GL约为11，均属中等水平；血糖控制不佳人群要严格控制量，新鲜无花果每天尝鲜1～2个即可，无花果干因含糖接近50%更要限量，并可视作主食的一部分来折算减少米饭面食。无花果膳食纤维含量约3克/100克，高于香蕉和火龙果，有助通便；钾含量约212毫克/100克，接近香蕉，有助调节血压；同时含有一定量硒及多酚、黄酮等抗氧化成分。"
        ],
        "top_k": 5,
    },
    {
        "question": "根据文章对甜瓜的介绍：甜瓜的含水量大约是多少？它的含糖量和热量水平如何，为什么说对减重人群比较友好？甜瓜中哪种矿物质含量较高，其典型含量范围是多少？",
        "user_input": "甜瓜的含水量有多高？糖和热量水平怎样，为什么适合减肥人群？哪种矿物质特别丰富，大概有多少？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "甜瓜的含水量超过92%，水分非常高；其含糖量大约6%左右，能量不到30千卡/100克，属于低糖、低热量水果，因此在相同体积下获得的能量较少，被视为对减重和控制体重人群比较友好的水果。甜瓜中钾含量相对较高，有些品种可达200～400毫克/100克，有助于维持电解质平衡和调节血压。"
        ],
        "top_k": 5,
    },
    {
        "question": "蜂糖李被称为“低糖低热量却甜度爆表”的水果。它的大致含水量、热量和含糖量分别是多少？GI和GL处于什么水平，对血糖有什么意义？从体重管理角度，一天建议吃多少比较合适，为什么不宜一次吃太多？",
        "user_input": "蜂糖李在水分、热量和含糖量方面有什么特点？GI和GL大概是多少，对血糖友好吗？减肥或控制体重时一天吃多少合适，为什么不能一次吃很多？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "蜂糖李含水量超过90%，能量约38千卡/100克，含糖量约8.2%，整体属于低糖低能量水果；其GI约24～32、GL约2～4，均为较低水平，对血糖影响相对温和，适合作为控糖人群的水果选择之一。体重管理时每天吃四五个蜂糖李（约100～150克）比较合适，不宜一次吃太多，一方面是避免能量累积过多，也要给其他水果留出空间保持多样化；此外只要不咬碎果核，一般不会释放出果仁中的有毒成分。"
        ],
        "top_k": 5,
    },
    {
        "question": "文章中如何评价白芸豆的减肥作用？从能量、蛋白质、膳食纤维和GI来看，白芸豆有哪些有利于体重管理的特点？所谓“白芸豆提取物阻断淀粉减肥”的理论依据是什么？欧洲食品安全局对其减肥功效有什么结论？结合膳食指南，全谷物和杂豆的推荐摄入量与现实摄入情况如何？",
        "user_input": "白芸豆真的能减肥吗？它在能量、蛋白质、膳食纤维和GI上有什么特点？所谓淀粉阻断机制是什么？EFSA对白芸豆提取物减肥功效是怎么评价的？膳食指南推荐的全谷物和杂豆摄入量与国人实际摄入量有什么差距？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "白芸豆干品能量约315千卡/100克，但因蛋白质约23克/100克、膳食纤维约9.8克/100克且GI约24，属于高蛋白、高纤维、低GI食物，饱腹感强、血糖上升缓慢，有利于体重管理。所谓“白芸豆提取物阻断淀粉减肥”的理论是其中含有α-淀粉酶抑制剂，可以抑制淀粉分解和吸收，从而在一定程度上减少能量摄入。不过欧洲食品安全局评估认为现有证据不足以证明白芸豆标准化提取物与体重减轻之间存在明确因果关系，不能把它当成可靠的减肥药物。《中国居民膳食指南（2022）》建议成年人每天摄入50～150克全谷物和杂豆，但调查显示我国成年人实际平均摄入不足30克/天，存在明显不足，因此更现实的做法是把白芸豆等杂豆作为主食的一部分，替代部分精制米面，长期改善结构，而不是指望单一食物快速瘦身。"
        ],
        "top_k": 5,
    },
    {
        "question": "根据红油玉耳的加工工艺，分析影响产品风味与质构的关键工艺参数，并解释辣椒油制作温度、玉耳泡发时间与灭菌条件对最终感官品质和安全性的作用机理。",
        "user_input": "根据红油玉耳的加工工艺，分析影响产品风味与质构的关键工艺参数，并解释辣椒油制作温度、玉耳泡发时间与灭菌条件对最终感官品质和安全性的作用机理。",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "红油玉耳的风味和质构主要受玉耳泡发、辣椒油制作温度、灭菌条件三方面影响。",
            "玉耳泡发：在100℃煮5–8分钟、浸泡60–90分钟可使木耳充分复水、保持脆滑口感。时间过短会偏硬，过长会造成胶质流失、口感松散。",
            "辣椒油制作温度：230℃炸香葱蒜可形成焦香基调；130℃炒辣椒能避免辣椒素分解、保持鲜辣风味。温度分段控制能使香气层次更丰富。",
            "灭菌条件：90–100℃水浴灭菌可杀灭微生物并保持质构。过高会使木耳发硬、汤体浑浊；过低则影响安全与保质期。"
        ],
        "top_k": 5
    },
    {
        "question": "整理出14种调味酱/汁的主要原料、关键工艺步骤、加热或灭菌参数及其成品特点。",
        "user_input": "根据文中描述，整理出14种调味酱/汁的主要原料、关键工艺步骤、加热或灭菌参数及其成品特点。",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "桑葚葡萄复合果酒：主要原料：桑葚、葡萄（玫瑰香）、白砂糖、柠檬酸、果胶酶、偏重亚硫酸钾、BV818酵母；关键工艺：压榨过滤→混合打浆→成分调整（糖22°Bx，pH4.4）→酶解45℃ 2h→加亚硫酸钾→接种酵母→20℃发酵→离心过滤→灭菌；加热/灭菌参数：100℃排气5–10min，沸水灭菌15min；成品特点：酒体澄清，香气浓郁，酸甜协调。",
            "酸枣果酒：主要原料：酸枣、水、白砂糖、果胶酶、柠檬酸钠、酵母菌；关键工艺：挑选清洗→1:3水煮10min→酶解55℃ 3h→过滤→调糖至18%→pH 4.0→酵母活化→26℃发酵5d；加热/灭菌参数：水煮10min；成品特点：果香清新，酒味柔和，酸甜适口。",
            "石榴葡萄枸杞复合果酒：主要原料：石榴、葡萄、枸杞、白砂糖、果胶酶、SO₂、酵母；关键工艺：取汁混合→糖度24.5°Bx、pH3.8→果胶酶酶解45℃ 2h→加SO₂→37℃酵母活化→20℃发酵59h→离心澄清；加热/灭菌参数：90℃水浴10min排气；成品特点：复合果香明显，色泽鲜艳，酒体清亮。",
            "酵母-乳酸菌共发酵低醇百香果酒：主要原料：百香果、白砂糖、果胶酶、纤维素酶、焦亚硫酸钾、酵母、乳酸菌；关键工艺：取汁→调糖18%→pH3.5→酶解50℃ 60min→加硫→酵母乳酸菌活化（30℃ 20min）→发酵4d→过滤→巴氏杀菌；加热/灭菌参数：65℃巴氏灭菌30min；成品特点：低醇度、果香清新、口感柔和。",
            "甘蔗菠萝复合果酒：主要原料：甘蔗、菠萝、白砂糖、果胶酶、偏重亚硫酸钾、柠檬酸、维C、酵母；关键工艺：原料处理→打浆→酶解25℃ 6h→护色→混合1:2→加糖225g/L→杀菌→酵母活化（35℃ 30min）→26℃发酵7d→陈酿2月；加热/灭菌参数：灭菌剂处理；成品特点：酸甜适中，热带风味浓郁，色泽清亮。",
            "蓝莓复合果酒：主要原料：蓝莓、白砂糖、柠檬酸、果胶酶、酵母；关键工艺：清洗→打浆→酶解45℃ 1.5h→糖度调节→酵母发酵→过滤→灭菌；加热/灭菌参数：沸水灭菌15min；成品特点：色泽紫红，果香浓郁，酸甜适口。",
            "草莓果酒：主要原料：草莓、白砂糖、果胶酶、酵母、柠檬酸；关键工艺：清洗→打浆→酶解45℃ 2h→加糖调味→接种酵母→发酵→过滤→灭菌；加热/灭菌参数：100℃排气5–10min；成品特点：果香清新，酒体亮红，口感柔和。",
            "樱桃果酒：主要原料：樱桃、白砂糖、果胶酶、酵母、柠檬酸；关键工艺：挑选→清洗→打浆→酶解50℃ 1.5h→糖度调节→发酵→过滤→灭菌；加热/灭菌参数：95℃水浴10min；成品特点：色泽鲜艳，果香浓郁，酸甜协调。",
            "柠檬复合果酒：主要原料：柠檬、白砂糖、果胶酶、酵母、焦亚硫酸钾；关键工艺：去皮切块→酶解→糖度调节→酵母发酵→过滤→灭菌；加热/灭菌参数：100℃排气5–10min；成品特点：酸香浓郁，清爽宜人。",
            "苹果复合果酒：主要原料：苹果、白砂糖、果胶酶、酵母、柠檬酸；关键工艺：清洗→打浆→酶解45℃ 2h→调糖→酵母发酵→过滤→灭菌；加热/灭菌参数：沸水灭菌15min；成品特点：色泽清亮，果香纯正，口感醇和。",
            "葡萄果酒：主要原料：葡萄、白砂糖、果胶酶、酵母、柠檬酸；关键工艺：清洗→压榨→酶解→糖度调整→发酵→过滤→灭菌；加热/灭菌参数：100℃排气5–10min；成品特点：果香浓郁，酒体澄清。",
            "黑加仑果酒：主要原料：黑加仑、白砂糖、果胶酶、酵母、柠檬酸；关键工艺：清洗→打浆→酶解45℃ 2h→调糖→发酵→过滤→灭菌；加热/灭菌参数：90℃水浴10min排气；成品特点：色泽深紫，果香浓郁，酸甜适口。",
            "草莓蓝莓混合果酒：主要原料：草莓、蓝莓、白砂糖、果胶酶、酵母、柠檬酸；关键工艺：清洗→打浆→酶解50℃ 1.5h→糖度调节→发酵→过滤→灭菌；加热/灭菌参数：100℃排气5–10min；成品特点：果香浓郁，色泽鲜艳，酸甜协调。",
            "复合热带果酒：主要原料：芒果、菠萝、百香果、白砂糖、果胶酶、酵母、柠檬酸；关键工艺：清洗→打浆→酶解45℃ 2h→调糖→发酵→过滤→灭菌；加热/灭菌参数：95–100℃水浴10–15min；成品特点：热带水果香气浓郁，口感酸甜适中。"
        ],
        "top_k": 5
    },
    {
        "question": "这8种脆片的加工工艺在‘脱水方式’和‘成品口感控制’上有哪些共性与差异？",
        "user_input": "这8种脆片的加工工艺在‘脱水方式’和‘成品口感控制’上有哪些共性与差异？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "共性：以脱水为核心环节，无论采用气流膨化、真空油炸、真空干燥还是烘干，脱水过程直接决定产品的酥脆度和膨化结构。",
            "普遍进行预处理，如漂烫、糖渍或预冻，以减少褐变、稳定色泽和组织结构。",
            "后处理环节注重保持脆度，如冷却、分级、充氮包装等，防止吸潮回软。",
            "差异：气流膨化类（南瓜脆片、黑木耳脆片）高温高压瞬时减压形成疏松多孔结构，口感轻脆。",
            "真空油炸类（柿子脆片、白萝卜脆片、小麦脆片）低温脱水，油脂包裹形成酥脆感并带油香。",
            "真空干燥类（猕猴桃脆片）低温保香，脆中略带韧性。",
            "烘干类（红枣酸奶脆片、薯香酥脆片）质地较密，风味浓郁。"
        ],
        "top_k": 5
    },
    {
        "question": "文中提到的几种鸡柳（川香鸡柳、无骨鸡柳、脆皮鸡柳、炼乳脆皮鸡柳）在配方和工艺上的主要差异是什么？",
        "user_input": "文中提到的几种鸡柳（川香鸡柳、无骨鸡柳、脆皮鸡柳、炼乳脆皮鸡柳）在配方和工艺上的主要差异是什么？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "川香鸡柳：香辛料复合调味，辣椒粉、姜粉、白胡椒粉比例高，腌渍12小时，油炸后速冻保存，风味突出。",
            "无骨鸡柳：真空滚揉与速冻工艺，腌渍入味保水性，多种香辛料按口味调整，油炸时间短、速冻温度低，质构细嫩。",
            "脆皮鸡柳：外层裹浆与脆皮糊配方，高筋面粉、玉米淀粉、泡打粉形成酥脆外壳，油炸180℃约2分钟，色泽金黄、口感酥脆。",
            "炼乳脆皮鸡柳：在脆皮基础上加入炼乳和蛋清液，使外壳更酥松、带奶香，油炸条件与脆皮鸡柳相近，口感更细腻。"
        ],
        "top_k": 5
    },
    {
        "question": "文中所述卤鸭翅的加工过程中，哪些环节对产品的风味形成与安全稳定性起关键作用？请简述原因。",
        "user_input": "文中所述卤鸭翅的加工过程中，哪些环节对产品的风味形成与安全稳定性起关键作用？请简述原因。",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "风味形成主要受调卤和卤制两个环节影响。",
            "调卤环节：香辛料炒制与卤汤煮制，辣椒、花椒炒香释放挥发性香气，卤汤加入酱油、糖、味精等构建复合香味。",
            "卤制环节：15–20分钟中火卤制并静置30分钟，使香味渗入肉质，同时改善肉质和色泽。",
            "安全稳定性由高压杀菌（115℃，20min）和真空包装保障，杀灭耐热菌与芽孢，延长货架期，防止氧化和微生物生长。"
        ],
        "top_k": 5
    },
    {
        "question": "无糖减肥糖果的主要成分及其功能是什么？",
        "user_input": "无糖减肥糖果的主要成分及其功能是什么？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "主要成分为麦芽糖醇、藤黄果提取物、瓜拉那提取物、L-酪氨酸、维生素B6、柠檬酸、香料和色素。",
            "麦芽糖醇低热量、不升血糖。",
            "藤黄果的HCA能抑制脂肪合成。",
            "瓜拉那提取物含咖啡因，可提神控食。",
            "L-酪氨酸可抑制食欲并调节情绪。"
        ],
        "top_k": 5
    },
    {
        "question": "不同的食品加工方法会如何影响食品中的主要营养成分？请举例说明。",
        "user_input": "不同的食品加工方法会如何影响食品中的主要营养成分？请举例说明。",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "加热加工：高温煎、炒、炸使蛋白质过度变性，破坏维生素C和B族维生素，可能生成丙烯酰胺；蒸煮温和，较好保留蛋白质结构和维生素。",
            "干燥加工：自然干燥易氧化损失维生素C，热风干燥温度高导致营养流失多，冷冻干燥最大限度保留营养和风味。",
            "腌制加工：盐渍导致水溶性维生素流失，糖渍增加能量摄入。",
            "发酵加工：分解蛋白质生成氨基酸，提高营养价值，并可增加部分维生素含量。",
            "现代技术如超高压和微波加热，可杀菌同时减少维生素损失，更好保留营养。"
        ],
        "top_k": 5
    },
    {
        "question": "虎皮鸡爪在加工过程中，哪两个环节对其“外酥里嫩”的特色口感形成最关键？请说明原因。",
        "user_input": "虎皮鸡爪在加工过程中，哪两个环节对其“外酥里嫩”的特色口感形成最关键？请说明原因。",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "挂糖环节：鸡爪浸泡麦芽糖液后，表面形成糖膜，油炸时促进糖类焦化，形成虎皮状纹理并增加酥脆感。",
            "油炸环节：180℃下油炸5–8分钟，使表皮脱水、收缩、起泡，形成虎皮外壳，同时内部保持水分与胶原蛋白，保证嫩滑弹性。"
        ],
        "top_k": 5
    },
    {
        "question": "在肉丸生产工艺中，为什么要先进行“油炸”再进行“蒸煮”？",
        "user_input": "在肉丸生产工艺中，为什么要先进行“油炸”再进行“蒸煮”？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "油炸：高温快速脱水，使表层蛋白凝固并形成金黄色焦香外壳，提高口感和香气。",
            "蒸煮：均匀加热至中心，保证肉丸熟透，同时保持水分，防止过硬。油炸后蒸煮可兼顾口感酥脆与内部嫩滑。"
        ],
        "top_k": 5
    },
    {
        "question": "鱼丸加工过程中，“搅拌成胶体”和“热处理成型”对成品质地有何影响？",
        "user_input": "鱼丸加工过程中，“搅拌成胶体”和“热处理成型”对成品质地有何影响？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "搅拌成胶体：鱼肉蛋白在高速搅拌下形成网络结构，使鱼丸具有弹性和粘性基础。",
            "热处理成型：通过加热使蛋白凝固固定形态，同时水分均匀分布，保证鱼丸弹性、紧致而不散。"
        ],
        "top_k": 5
    },
    {
        "question": "甜樱桃新的执行标准是什么",
        "user_input": "甜樱桃新的执行标准是什么",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "甜樱桃新的执行标准是 GB/T 26906—2024。"
        ],
        "top_k": 5,
    },
    {
        "question": "商业上制作粉条对于水有什么要求",
        "user_input": "商业上制作粉条对于水有什么要求",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "应符合GB5749的规定。"
        ],
        "top_k": 5,
    },
    {
        "question": "食堂需要对安全人员培训多久？",
        "user_input": "食堂需要对安全人员培训多久？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "食品安全总监和食品安全员每年参加培训的时间不少于40小时"
        ],
        "top_k": 5,
    },
    {
        "question": "那些用餐单位需要配备安全总监？",
        "user_input": "那些用餐单位需要配备安全总监？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "（一）每餐次平均用餐人数300人以上的幼儿园食堂、承包经营企业；（二）每餐次平均用餐人数500人以上的学校食堂、承包经营企业；（三）每餐次平均用餐人数300人以上的养老机构食堂、承包经营企业；（四）每餐次平均用餐人数1000人以上的其他单位食堂、承包经营企业；（五）每餐次平均供餐人数1000人以上的供餐单位。符合前款条件的学校、幼儿园委托承包经营的，学校、幼儿园也应当配备食品安全总监，承担相应责任。县级以上地方市场监督管理部门应当结合实际，指导本辖区具备条件的单位食堂、承包经营企业、供餐单位配备食品安全总监。"
        ],
        "top_k": 5,
    },
    {
        "question": "餐饮服务提供者未按规定索证索票的会怎样？",
        "user_input": "餐饮服务提供者未按规定索证索票的会怎样？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "纳入省重点监管食品电子追溯系统的食品生产经营企业未按规定传送数据或者上传虚假电子凭证的，由县级以上人民政府食品安全监督管理部门责令改正，给予警告；拒不改正的，处五千元以上五万元以下罚款；情节严重的，责令停产停业，直至吊销许可证。"
        ],
        "top_k": 5,
    },
    {
        "question": "学校食品安全实行什么安全制度？",
        "user_input": "学校食品安全实行什么安全制度？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "学校食品安全实行校长（园长）负责制"
            ],
        "top_k": 5,
    },
    {
        "question": "学校食堂采购、贮存亚硝酸盐会怎样？",
        "user_input": "学校食堂采购、贮存亚硝酸盐会怎样？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "由县级以上人民政府食品安全监督管理部门责令改正，给予警告，并处5000元以上3万元以下罚款"
            ],
        "top_k": 5,
    },
    {
        "question": "违规食品安全法有哪些严重情形？",
        "user_input": "违规食品安全法有哪些严重情形？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "(一)违法行为涉及的产品货值金额2万元以上或者违法行为持续时间3个月以上； (二)造成食源性疾病并出现死亡病例，或者造成30人以上食源性疾病但未出现死亡病例； (三)故意提供虚假信息或者隐瞒真实情况； (四)拒绝、逃避监督检查； (五)因违反食品安全法律、法规受到行政处罚后1年内又实施同一性质的食品安全违法行为，或者因违反食品安全法律、法规受到刑事处罚后又实施食品安全违法行为； (六)其他情节严重的情形。 "
            ],
        "top_k": 5,
    },
    {
        "question": "学校食堂采购、贮存亚硝酸盐会怎样？",
        "user_input": "学校食堂采购、贮存亚硝酸盐会怎样？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "由县级以上人民政府食品安全监督管理部门责令改正，给予警告，并处5000元以上3万元以下罚款"
            ],
        "top_k": 5,
    },
    {
        "question": "当境外发生食品安全事件时，国家出入境检应对相关的食品采取什么控制措施？",
        "user_input": "当境外发生食品安全事件时，国家出入境检应对相关的食品采取什么控制措施？",
        "retrieved_contexts": [],
        "response": "",
        "reference": "",
        "ground_truths": [
            "(一)退货或者销毁处理； (二)有条件地限制进口； (三)暂停或者禁止进口。"
            ],
        "top_k": 5,
    },
]


class DashScopeEmbeddings(Embeddings):
    """调用DashScope/OpenAI兼容嵌入API"""

    def __init__(
        self,
        client: OpenAI,
        *,
        model: str,
        encoding_format: str = "float",
        dimensions: Optional[int] = None,
    ):
        self.client = client
        self.model = model
        self.encoding_format = encoding_format
        self.dimensions = dimensions

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        normalized = [str(text) for text in texts if str(text).strip()]
        if not normalized:
            return []
        payload: Dict[str, object] = {
            "model": self.model,
            "encoding_format": self.encoding_format,
        }
        if len(normalized) == 1:
            payload["input"] = normalized[0]
        else:
            payload["input"] = normalized
        if self.dimensions:
            payload["dimensions"] = self.dimensions

        response = self.client.embeddings.create(**payload)
        return [item.embedding for item in response.data]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embed_batch(list(texts))

    def embed_query(self, text: str) -> List[float]:
        embeddings = self._embed_batch([text])
        return embeddings[0] if embeddings else []


if LangchainLLMWrapperBase is not None and LLMResult is not None:
    class SequentialLangchainLLMWrapper(LangchainLLMWrapperBase):
        """通过重复的单次调用模拟多次LangchainLLMWrapper请求"""

        def generate_text(
            self,
            prompt,
            n: int = 1,
            temperature: Optional[float] = 0.01,
            stop: Optional[List[str]] = None,
            callbacks=None,
        ):
            if n <= 1:
                return super().generate_text(
                    prompt=prompt,
                    n=1,
                    temperature=temperature,
                    stop=stop,
                    callbacks=callbacks,
                )

            combined_generations: List = []
            combined_runs: List = []
            llm_output = None

            for _ in range(n):
                partial = super().generate_text(
                    prompt=prompt,
                    n=1,
                    temperature=temperature,
                    stop=stop,
                    callbacks=callbacks,
                )
                if partial.generations:
                    combined_generations.extend(partial.generations[0])
                if partial.run:
                    combined_runs.extend(partial.run)
                llm_output = partial.llm_output or llm_output

            return LLMResult(
                generations=[combined_generations],
                llm_output=llm_output,
                run=combined_runs or None,
            )

        async def agenerate_text(
            self,
            prompt,
            n: int = 1,
            temperature: Optional[float] = 0.01,
            stop: Optional[List[str]] = None,
            callbacks=None,
        ):
            if n <= 1:
                return await super().agenerate_text(
                    prompt=prompt,
                    n=1,
                    temperature=temperature,
                    stop=stop,
                    callbacks=callbacks,
                )

            combined_generations: List = []
            combined_runs: List = []
            llm_output = None

            for _ in range(n):
                partial = await super().agenerate_text(
                    prompt=prompt,
                    n=1,
                    temperature=temperature,
                    stop=stop,
                    callbacks=callbacks,
                )
                if partial.generations:
                    combined_generations.extend(partial.generations[0])
                if partial.run:
                    combined_runs.extend(partial.run)
                llm_output = partial.llm_output or llm_output

            return LLMResult(
                generations=[combined_generations],
                llm_output=llm_output,
                run=combined_runs or None,
            )
else:
    SequentialLangchainLLMWrapper = None


@dataclass
class EvaluationResult:
    """封装一次 RAGAS 评估的结果，方便序列化与展示。"""

    variant: str
    run_at: datetime
    metrics: List[Dict[str, object]]
    samples: List[Dict[str, object]]

    def to_serializable(self) -> Dict[str, object]:
        return {
            "variant": self.variant,
            "run_at": self.run_at.isoformat(),
            "metrics": self.metrics,
            "samples": self.samples,
        }


class RagasEvaluationManager:
    """负责读取评测数据集、执行 RAGAS 评估并缓存结果。"""

    def __init__(self, dataset_path: Path, settings: Settings):
        self.dataset_path = dataset_path
        self.settings = settings
        self.results_dir = dataset_path.parent / "ragas_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.eval_model = settings.eval_llm_model
        self.eval_base_url = settings.eval_llm_base_url
        self.eval_api_key = settings.eval_api_key
        self._configure_llm_environment()
        self._eval_client = self._init_eval_client()
        self._ragas_llm = self._init_ragas_llm()
        self._ragas_embeddings = self._init_ragas_embeddings()
        self._ensure_dataset_exists()

    def _ensure_dataset_exists(self) -> None:
        if self.dataset_path.exists():
            return
        logger.info("Creating default RAGAS dataset at %s", self.dataset_path)
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        with self.dataset_path.open("w", encoding="utf-8") as f:
            json.dump(DEFAULT_DATASET, f, ensure_ascii=False, indent=2)

    def _configure_llm_environment(self) -> None:
        """确保 ragas 所依赖的 OpenAI 客户端能读取到正确的密钥与地址。"""
        if self.eval_api_key:
            os.environ["OPENAI_API_KEY"] = self.eval_api_key
        if self.eval_base_url:
            os.environ["OPENAI_BASE_URL"] = self.eval_base_url

    def _init_ragas_llm(self):
        if (
            LangchainLLMWrapper is None
            or SequentialLangchainLLMWrapper is None
            or ChatOpenAI is None
        ):
            logger.warning("缺少 langchain-openai 或 ragas.llms，RAGAS 无法调用 LLM。")
            return None
        if not (self.eval_api_key and self.eval_model):
            logger.warning("缺少评估模型配置，RAGAS 将无法调用 LLM 指标。")
            return None
        try:
            llm = ChatOpenAI(
                model=self.eval_model,
                openai_api_key=self.eval_api_key,
                openai_api_base=self.eval_base_url,
                # 尝试关闭推理
                extra_body={"enable_thinking": False},
            )
            logger.info(
                "RAGAS LLM initialised: model=%s base=%s",
                self.eval_model,
                self.eval_base_url,
            )
            return SequentialLangchainLLMWrapper(langchain_llm=llm, bypass_n=True)
        except Exception as exc:
            logger.warning("初始化 RAGAS LLM 失败: %s", exc)
            return None

    def _init_ragas_embeddings(self):
        if Embeddings is None:
            logger.warning("缺少 langchain-core embeddings 接口，RAGAS 无法配置嵌入模型。")
            return None
        if not (self.eval_api_key and self.eval_base_url):
            logger.warning("缺少评估嵌入配置，RAGAS 将无法调用 DashScope Embedding。")
            return None

        model_name = self.settings.eval_embedding_model or "text-embedding-3-small"
        try:
            embedding_client = OpenAI(
                api_key=self.eval_api_key,
                base_url=self.eval_base_url,
            )
        except Exception as exc:
            logger.warning("初始化 DashScope Embedding 客户端失败: %s", exc)
            return None

        dimensions_value: Optional[int] = None
        dimensions_raw = os.getenv("EVAL_EMBED_DIMENSIONS")
        if dimensions_raw:
            try:
                dimensions_value = int(dimensions_raw)
            except ValueError:
                logger.warning("EVAL_EMBED_DIMENSIONS=%s 无法解析为整数，已忽略。", dimensions_raw)

        try:
            embeddings = DashScopeEmbeddings(
                client=embedding_client,
                model=model_name,
                encoding_format=os.getenv("EVAL_EMBED_ENCODING", "float"),
                dimensions=dimensions_value,
            )
            logger.info(
                "RAGAS Embeddings initialised via DashScope SDK: model=%s base=%s",
                model_name,
                self.eval_base_url,
            )
            return embeddings
        except Exception as exc:
            logger.warning("初始化 RAGAS Embeddings 失败: %s", exc)
            return None

    def _metric_instances(self, include_reference: bool) -> List[object]:
        if Faithfulness is None:
            raise RuntimeError("ragas 未安装，请安装 ragas 后再试。")
        metrics: List[object] = [Faithfulness(), ContextPrecision()]
        if include_reference:
            metrics.extend([AnswerRelevancy(), ContextRecall()])
        logger.debug(
            "RAGAS metric instances prepared (include_reference=%s): %s",
            include_reference,
            [m.__class__.__name__ for m in metrics],
        )
        return metrics

    def _build_evaluation_dataset(self, records: List[Dict[str, object]]):
        if EvaluationDataset is None:
            raise RuntimeError("ragas 未安装，请安装 ragas 后再试。")
        logger.debug("Building RAGAS EvaluationDataset with records: %s", records)
        return EvaluationDataset.from_list(records)
    def load_dataset(self) -> List[Dict[str, object]]:
        with self.dataset_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("RAGAS dataset must be a list of question objects.")
        return data

    def _init_eval_client(self) -> Optional[OpenAI]:
        if not (self.eval_model and self.eval_base_url and self.eval_api_key):
            logger.info(
                "Evaluation LLM not fully configured; inline RAGAS 将跳过自动生成参考答案。"
            )
            return None
        try:
            client = OpenAI(api_key=self.eval_api_key, base_url=self.eval_base_url)
            logger.info(
                "Evaluation LLM client initialised (model=%s)", self.eval_model
            )
            return client
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning(
                "Failed to initialise evaluation LLM client: %s. Inline RAGAS 将不生成参考答案。",
                exc,
            )
            return None

    def _build_ragas_dataset(self, records: List[Dict[str, object]]):
        return self._build_evaluation_dataset(records)

    def _serialize_metrics(
        self,
        ragas_result,
        metric_defs: Optional[List[object]] = None,
    ) -> List[Dict[str, object]]:
        """兼容不同版本 RAGAS 的指标输出结构。"""

        def _to_float(value) -> Optional[float]:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                value = float(value)
                if math.isnan(value):
                    return None
                return value
            try:
                value = float(value)
                if math.isnan(value):
                    return None
                return value
            except Exception:
                return None

        if isinstance(ragas_result, dict) and ragas_result:
            metrics_summary: List[Dict[str, object]] = []
            for name, value in ragas_result.items():
                metrics_summary.append(
                    {
                        "name": name,
                        "score": _to_float(value),
                        "ci_low": None,
                        "ci_high": None,
                    }
                )
            logger.debug("RAGAS metrics parsed from dict: %s", metrics_summary)
            return metrics_summary

        repr_dict = getattr(ragas_result, "_repr_dict", None)
        if isinstance(repr_dict, dict) and repr_dict:
            metrics_summary = [
                {
                    "name": name or f"metric_{idx}",
                    "score": _to_float(value),
                    "ci_low": None,
                    "ci_high": None,
                }
                for idx, (name, value) in enumerate(repr_dict.items())
            ]
            logger.debug("RAGAS metrics parsed from _repr_dict: %s", metrics_summary)
            return metrics_summary

        scores_rows = getattr(ragas_result, "scores", None)
        if isinstance(scores_rows, list) and scores_rows:
            aggregates: Dict[str, Dict[str, float]] = {}
            for row in scores_rows:
                if not isinstance(row, dict):
                    continue
                for key, value in row.items():
                    score_value = _to_float(value)
                    if score_value is None:
                        continue
                    bucket = aggregates.setdefault(key, {"sum": 0.0, "count": 0.0})
                    bucket["sum"] += score_value
                    bucket["count"] += 1
            if aggregates:
                metrics_summary = []
                for idx, (name, payload) in enumerate(aggregates.items()):
                    count = payload["count"]
                    avg = payload["sum"] / count if count else None
                    metrics_summary.append(
                        {
                            "name": name or f"metric_{idx}",
                            "score": avg,
                            "ci_low": None,
                            "ci_high": None,
                        }
                    )
                logger.debug(
                    "RAGAS metrics computed from raw scores rows: %s", metrics_summary
                )
                return metrics_summary

        # 直接读取 metrics 列表或字典
        entries = getattr(ragas_result, "metrics", None)
        metrics_summary: List[Dict[str, object]] = []

        def _extract(metric_name, payload, fallback_idx: int) -> Dict[str, object]:
            score = None
            ci_low = None
            ci_high = None

            if isinstance(payload, dict):
                score = _to_float(payload.get("score"))
                if score is None:
                    score = _to_float(payload.get("value"))
                ci = payload.get("confidence_interval") or payload.get("confidence_interval_")
                if isinstance(ci, dict):
                    ci_low = _to_float(ci.get("low"))
                    ci_high = _to_float(ci.get("high"))
            else:
                score = _to_float(getattr(payload, "score", None) or getattr(payload, "value", payload))
                ci = getattr(payload, "confidence_interval", None) or getattr(payload, "confidence_interval_", None)
                if isinstance(ci, dict):
                    ci_low = _to_float(ci.get("low"))
                    ci_high = _to_float(ci.get("high"))
                elif hasattr(ci, "low"):
                    ci_low = _to_float(ci.low)
                    ci_high = _to_float(getattr(ci, "high", None))

            label = metric_name
            if not label and metric_defs and fallback_idx < len(metric_defs):
                label = getattr(metric_defs[fallback_idx], "name", None) or metric_defs[fallback_idx].__class__.__name__
            if not label:
                label = f"metric_{fallback_idx}"

            return {"name": label, "score": score, "ci_low": ci_low, "ci_high": ci_high}

        if isinstance(entries, dict):
            for idx, (name, payload) in enumerate(entries.items()):
                metrics_summary.append(_extract(name, payload, idx))
        elif isinstance(entries, (list, tuple)):
            for idx, payload in enumerate(entries):
                metrics_summary.append(_extract(None, payload, idx))

        if metrics_summary:
            logger.debug("RAGAS metrics parsed successfully: %s", metrics_summary)
            return metrics_summary

        # 回退：dict()/to_dict()
        for attr in ("dict", "as_dict", "to_dict"):
            if hasattr(ragas_result, attr):
                try:
                    data = getattr(ragas_result, attr)()
                    metrics = isinstance(data, dict) and data.get("metrics")
                    if isinstance(metrics, dict):
                        for idx, (name, payload) in enumerate(metrics.items()):
                            metrics_summary.append(_extract(name, payload, idx))
                        break
                except Exception:
                    continue

        if not metrics_summary:
            logger.warning("RAGAS returned empty metrics result")
        return metrics_summary

    @staticmethod
    def _serialize_raw_result(ragas_result) -> Dict[str, object]:
        """将 RAGAS 原始输出转成可序列化字典，便于调试。"""
        for attr in ("dict", "as_dict", "to_dict"):
            if hasattr(ragas_result, attr):
                try:
                    data = getattr(ragas_result, attr)()
                    if isinstance(data, dict):
                        return data
                except Exception:
                    continue
        try:
            return {"repr": repr(ragas_result)}
        except Exception:
            return {"repr": "unavailable"}

    @staticmethod
    def _normalise_contexts(raw_contexts: Any) -> List[str]:
        if not raw_contexts:
            return []
        if isinstance(raw_contexts, (str, dict)):
            iterable = [raw_contexts]
        else:
            try:
                iterable = list(raw_contexts)
            except TypeError:
                iterable = [raw_contexts]
        contexts_list: List[str] = []
        for item in iterable:
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content") or item.get("page_content") or ""
            else:
                text = str(item)
            text = str(text)
            if text.strip():
                contexts_list.append(text)
        return contexts_list

    @staticmethod
    def _diagnose(metrics_summary: List[Dict[str, object]]) -> List[Dict[str, str]]:
        scores = {item["name"].lower(): item.get("score") for item in metrics_summary}
        findings: List[Dict[str, str]] = []

        faithfulness = scores.get("faithfulness")
        if faithfulness is not None and faithfulness < 0.5:
            findings.append(
                {
                    "type": "生成错误",
                    "detail": "回答与检索到的上下文不一致，建议检查生成提示词或引用内容。",
                }
            )

        context_precision = scores.get("context_precision")
        if context_precision is not None and context_precision < 0.5:
            findings.append(
                {
                    "type": "检索错误",
                    "detail": "检索返回的片段与问题相关性较低，可以优化向量索引或检索策略。",
                }
            )

        context_recall = scores.get("context_recall")
        if context_recall is not None and context_recall < 0.5:
            findings.append(
                {
                    "type": "检索错误",
                    "detail": "检索可能遗漏关键片段，可增加 Top-K 或补充文档覆盖度。",
                }
            )

        answer_relevancy = scores.get("answer_relevancy")
        if answer_relevancy is not None and answer_relevancy < 0.5:
            findings.append(
                {
                    "type": "生成错误",
                    "detail": "回答与参考答案差距较大，建议检查生成模型或提供更详尽的上下文。",
                }
            )

        return findings

    def _fallback_llm_metrics(
        self,
        *,
        question: str,
        answer: str,
        contexts: List[str],
        reference: Optional[str],
    ) -> Tuple[List[Dict[str, object]], List[Dict[str, str]]]:
        if self._eval_client is None:
            logger.warning("Fallback LLM metrics skipped: evaluation client not configured.")
            return [], []

        context_block = "\n".join(f"{idx+1}. {ctx}" for idx, ctx in enumerate(contexts) if ctx.strip())
        if not context_block:
            context_block = "无"
        reference_text = reference or "无"

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个负责 RAG 评估的审查员，请根据提供的问题、回答、检索上下文与参考答案，"
                    "给出四项指标的评分，并输出 JSON。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请评估以下内容，输出 JSON，字段包括：\n"
                    "faithfulness, answer_relevancy, context_precision, context_recall（取值 0 到 1 之间的小数）。\n"
                    "如果无法判断，请给出 null。\n\n"
                    f"问题：{question}\n"
                    f"回答：{answer}\n"
                    f"参考答案：{reference_text}\n"
                    f"检索上下文：\n{context_block}\n"
                ),
            },
        ]

        try:
            response = self._eval_client.chat.completions.create(
                model=self.eval_model,
                messages=messages,
            )
            content = response.choices[0].message.content.strip()
        except Exception as exc:  # pragma: no cover
            logger.warning("Fallback LLM metrics failed: %s", exc)
            return [], [
                {
                    "type": "评估提示",
                    "detail": "评估模型调用失败，无法生成得分。",
                }
            ]

        json_text = content
        if not content.startswith("{"):
            candidates = re.findall(r"\{.*\}", content, flags=re.DOTALL)
            if candidates:
                json_text = candidates[-1]

        try:
            data = json.loads(json_text)
        except Exception as exc:
            logger.warning("Fallback LLM metrics解析失败: %s | content=%s", exc, content)
            return [], [
                {
                    "type": "评估提示",
                    "detail": "评估模型返回的得分格式无法解析。",
                }
            ]

        metrics_map = {
            "faithfulness": "Faithfulness",
            "answer_relevancy": "Answer Relevancy",
            "context_precision": "Context Precision",
            "context_recall": "Context Recall",
        }

        metrics: List[Dict[str, object]] = []
        for key, label in metrics_map.items():
            score = data.get(key)
            if isinstance(score, (int, float)):
                score = max(0.0, min(1.0, float(score)))
            else:
                score = None
            metrics.append(
                {
                    "name": key,
                    "score": score,
                    "ci_low": None,
                    "ci_high": None,
                    "label": label,
                }
            )

        diagnosis = [
            {
                "type": "评估提示",
                "detail": "RAGAS 未返回结果，已改用评估模型直接打分。",
            }
        ]

        return metrics, diagnosis

    def run(self, pipeline: RAGPipeline, *, variant: str) -> EvaluationResult:
        if evaluate is None:
            raise RuntimeError(
                "ragas 未安装，请运行 `pip install ragas` 或检查依赖。"
            )

        dataset = self.load_dataset()
        evaluations: List[Dict[str, object]] = []
        ragas_records: List[Dict[str, object]] = []

        for index, sample in enumerate(dataset):
            raw_question = sample.get("user_input") or sample.get("question") or sample.get("query")
            question = str(raw_question).strip() if raw_question is not None else ""
            if not question:
                logger.warning("RAGAS dataset[%d] 缺少问题字段，已跳过。", index)
                continue

            top_k = sample.get("top_k")
            provided_contexts = sample.get("retrieved_contexts") or sample.get("contexts")
            contexts_from_sample = self._normalise_contexts(provided_contexts)
            provided_response = sample.get("response")

            use_provided_answer = provided_response is not None and bool(contexts_from_sample)

            if use_provided_answer:
                contexts_text = contexts_from_sample
                answer_text = str(provided_response)
                logger.debug(
                    "Using precomputed RAG sample for question='%s' (contexts=%d).",
                    question,
                    len(contexts_text),
                )
            else:
                logger.info("Running pipeline '%s' for RAGAS sample: %s", variant, question)
                answer = pipeline.answer(question, top_k=top_k)
                contexts_text = self._normalise_contexts(answer.contexts)
                answer_text = answer.answer

            ground_truths_raw = sample.get("ground_truths")
            ground_truths: List[str] = []
            if ground_truths_raw:
                if isinstance(ground_truths_raw, str):
                    ground_truths = [ground_truths_raw]
                elif isinstance(ground_truths_raw, (list, tuple)):
                    ground_truths = [str(item) for item in ground_truths_raw if str(item).strip()]

            reference_text = sample.get("reference")
            reference_text = str(reference_text).strip() if reference_text else None
            if not reference_text:
                reference_text = self._generate_reference_answer(
                    question=question, contexts=contexts_text
                )
            if not reference_text and ground_truths:
                reference_text = ground_truths[0]
            display_ground_truths = ground_truths if ground_truths else ([reference_text] if reference_text else [])

            evaluations.append(
                {
                    "question": question,
                    "answer": answer_text,
                    "contexts": contexts_text,
                    "ground_truths": display_ground_truths,
                    "reference": reference_text or "",
                }
            )

            ragas_record = {
                "user_input": question,
                "retrieved_contexts": contexts_text,
                "response": answer_text,
            }
            if reference_text:
                ragas_record["reference"] = reference_text
            ragas_records.append(ragas_record)

        include_reference = all(bool(record.get("reference")) for record in ragas_records)
        logger.debug(
            "Batch RAGAS evaluation for variant=%s, include_reference=%s", variant, include_reference
        )
        ragas_dataset = self._build_evaluation_dataset(ragas_records)
        metrics_to_eval = self._metric_instances(include_reference)
        eval_kwargs: Dict[str, object] = {}
        if self._ragas_llm is not None:
            eval_kwargs["llm"] = self._ragas_llm
        if self._ragas_embeddings is not None:
            eval_kwargs["embeddings"] = self._ragas_embeddings
        logger.debug(
            "Calling RAGAS evaluate (batch) metrics=%s llm=%s embeddings=%s",
            [m.__class__.__name__ for m in metrics_to_eval],
            type(self._ragas_llm).__name__ if self._ragas_llm else None,
            type(self._ragas_embeddings).__name__ if self._ragas_embeddings else None,
        )
        ragas_result = evaluate(
            dataset=ragas_dataset,
            metrics=metrics_to_eval,
            **eval_kwargs,
        )
        logger.debug("Batch RAGAS raw result: %s", ragas_result)
        metrics_summary = self._serialize_metrics(ragas_result)
        if not metrics_summary or all(m.get("score") is None for m in metrics_summary):
            logger.warning("Batch RAGAS metrics empty for variant %s. Falling back to LLM scoring.", variant)
            fallback_scores: Dict[str, List[float]] = {
                "faithfulness": [],
                "answer_relevancy": [],
                "context_precision": [],
                "context_recall": [],
            }
            for sample in evaluations:
                metrics, _ = self._fallback_llm_metrics(
                    question=sample["question"],
                    answer=sample["answer"],
                    contexts=sample["contexts"],
                    reference=sample.get("reference") or None,
                )
                for metric in metrics:
                    if metric["score"] is not None:
                        fallback_scores.setdefault(metric["name"], []).append(metric["score"])
            aggregated: List[Dict[str, object]] = []
            for key, label in [
                ("faithfulness", "Faithfulness"),
                ("context_precision", "Context Precision"),
                ("answer_relevancy", "Answer Relevancy"),
                ("context_recall", "Context Recall"),
            ]:
                values = fallback_scores.get(key, [])
                score = sum(values) / len(values) if values else None
                aggregated.append(
                    {
                        "name": key,
                        "score": score,
                        "ci_low": None,
                        "ci_high": None,
                        "label": label,
                    }
                )
            metrics_summary = aggregated
        samples_summary = [
            {
                "question": item["question"],
                "answer": item["answer"],
                "ground_truths": item["ground_truths"],
                "contexts": item["contexts"],
                "reference": item.get("reference"),
            }
            for item in evaluations
        ]

        result = EvaluationResult(
            variant=variant,
            run_at=datetime.utcnow(),
            metrics=metrics_summary,
            samples=samples_summary,
        )

        self._store_result(result)
        return result

    def _generate_reference_answer(
        self, *, question: str, contexts: List[str]
    ) -> Optional[str]:
        """调用独立大模型生成参考答案，用于缺失标注时的评估。"""
        if self._eval_client is None:
            return None

        context_snippets = "\n\n".join(contexts[:5]) or "（上下文为空）"
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位负责生成精炼参考答案的助手。请基于提供的上下文，总结出一个可靠的事实性回答，"
                    "语言保持中文，避免夸大或编造。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"问题：{question}\n\n"
                    f"检索到的上下文如下：\n{context_snippets}\n\n"
                    "请输出一段 1-3 句的参考答案，尽量客观准确。"
                ),
            },
        ]

        try:
            response = self._eval_client.chat.completions.create(
                model=self.eval_model,
                messages=messages,
                # 尝试关闭推理
                extra_body={"enable_thinking": False},
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            logger.warning(
                "Failed to generate reference answer via %s: %s",
                self.eval_model,
                exc,
            )
            return None

        try:
            content = response.choices[0].message.content.strip()
        except Exception as exc:  # pragma: no cover
            logger.warning("Invalid response from evaluation model: %s", exc)
            return None

        return content or None

    def evaluate_inline(
        self,
        *,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truths: Optional[List[str]] = None,
        reference: Optional[str] = None,
    ) -> Dict[str, object]:
        """对单次问答执行 RAGAS 评估，返回指标摘要。"""
        if evaluate is None:
            raise RuntimeError(
                "ragas 未安装，请运行 `pip install ragas` 或检查依赖。"
            )

        contexts_clean = self._normalise_contexts(contexts)

        user_provided_gt = bool(ground_truths) or bool(reference)
        gt_list: List[str] = []
        if ground_truths:
            if isinstance(ground_truths, str):
                gt_list = [str(ground_truths)]
            else:
                gt_list = [str(item) for item in ground_truths if str(item).strip()]
        reference_text: Optional[str] = str(reference).strip() if reference else None
        generated_reference: Optional[str] = None

        if not reference_text and not gt_list:
            generated_reference = self._generate_reference_answer(
                question=question, contexts=contexts_clean
            )
            if generated_reference:
                reference_text = generated_reference
                gt_list = [generated_reference]

        if not reference_text and gt_list:
            reference_text = gt_list[0]

        include_reference = bool(reference_text)

        record = {
            "user_input": question,
            "retrieved_contexts": contexts_clean,
            "response": answer,
        }
        if reference_text:
            record["reference"] = reference_text

        logger.debug("RAGAS inline input: %s", record)

        ragas_dataset = self._build_evaluation_dataset([record])
        metrics_instances = self._metric_instances(include_reference)
        eval_kwargs: Dict[str, object] = {}
        if self._ragas_llm is not None:
            eval_kwargs["llm"] = self._ragas_llm
        if self._ragas_embeddings is not None:
            eval_kwargs["embeddings"] = self._ragas_embeddings
        logger.debug(
            "Calling RAGAS evaluate with metrics=%s llm=%s embeddings=%s",
            [m.__class__.__name__ for m in metrics_instances],
            type(self._ragas_llm).__name__ if self._ragas_llm else None,
            type(self._ragas_embeddings).__name__ if self._ragas_embeddings else None,
        )

        ragas_result = evaluate(
            dataset=ragas_dataset,
            metrics=metrics_instances,
            **eval_kwargs,
        )
        logger.debug("RAGAS raw result: %s", ragas_result)
        metrics_summary = self._serialize_metrics(ragas_result)
        diagnosis = self._diagnose(metrics_summary)
        raw_payload = ragas_result if isinstance(ragas_result, dict) else self._serialize_raw_result(ragas_result)
        if not metrics_summary or all(metric.get("score") is None for metric in metrics_summary):
            logger.warning("Inline RAGAS metrics empty. Raw payload: %s", raw_payload)
            diagnosis.append(
                {
                    "type": "评估提示",
                    "detail": "RAGAS 未返回有效得分，请检查依赖或稍后重试。",
                }
            )

        ground_truth_source = "user" if user_provided_gt else (
            "generated" if generated_reference else "none"
        )

        primary_reference = None
        if reference_text:
            primary_reference = reference_text

        return {
            "metrics": metrics_summary,
            "used_ground_truths": bool(gt_list),
            "ground_truth_source": ground_truth_source,
            "reference": primary_reference,
            "references": gt_list if gt_list else [],
            "diagnosis": diagnosis,
            "raw": raw_payload,
        }

    def _store_result(self, result: EvaluationResult) -> None:
        path = self.results_dir / f"{result.variant}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(result.to_serializable(), f, ensure_ascii=False, indent=2)

    def load_cached(self, variant: str) -> Optional[Dict[str, object]]:
        path = self.results_dir / f"{variant}.json"
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
