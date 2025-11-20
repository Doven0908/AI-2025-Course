import math
import random
import time
import re
import json
from transformers import AutoTokenizer

from ..base_tree import BaseTree, TreeNode
from .examples import construct_prompt,get_rethinking_token,get_stop_tokens,get_rechecking_prompt,get_rechecking_pattern

from utils.utils import print_time_exceed_warning
from utils.strip_string import extract_answer
from utils.serve_vllm import completion_request,get_token_usage
from utils.grader import math_equal

# class TreeNode:
#     def __init__(self, parent,action,args):
#         self.state = ""
#         self.action = action if action else ""
#         self.parent = parent 
#         self.depth = parent.depth + 1 if parent else 0
#         self.children = {}
#         self.is_terminal = False
#         self.num_visits = 0
#         self.value = 0.0
#     def get_q(self):
#         return self.value
#     def get_state(self):
#         return self.state
#     def refine_state(self,state):
#         self.state = state
#     def create_child(self,action,args):
#         child = TreeNode(self,action,args)
#         if action not in self.children.keys():
#             self.children[action] = child
# class BaseTree:
#     def __init__(self, tree_id, question, gt_ans, gt_cot ,args):
#         self.tree_id = tree_id
#         self.question = question
#         self.gt_ans = gt_ans
#         self.gt_cot = gt_cot
        
#         self.final_answer = None
#         self.final_scores = []

#         self.tree_status = []
#         self.tree_status_index = 0
#         self.tree_status.append("select")
#         self.tree_status.append("expand")
#         self.tree_status.append("simulate")
#         self.tree_status.append("back_propagate")

#         root = TreeNode(None,None,args)
#         self.curnode:TreeNode = root
#         self.root:TreeNode = root
#         self.final_node = None

#         self.starttime = None

#     def get_cur_state(self):
#         return self.tree_status[self.tree_status_index % len(self.tree_status)]
    
#     def step(self):
#         self.tree_status_index += 1
    
#     def get_final_ans(self): # 返回为list
#         return [self.final_answer] if self.final_answer is not None else [extract_answer(self.final_node.get_state())]


class CON_MCTS_Tree(BaseTree):
    def __init__(self, tree_id, question, gt_ans, gt_cot ,args):
        super().__init__(tree_id, question, gt_ans, gt_cot ,args)
        root = CON_MCTS_TreeNode(None,None,args)
        self.curnode: CON_MCTS_TreeNode = root
        self.root: CON_MCTS_TreeNode = root
        prompt = construct_prompt(args.data_name,question,args)
        self.root.state = prompt

        self.new_child_list= []

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

class CON_MCTS_TreeNode(TreeNode):
    def __init__(self, parent, action,args):
        super().__init__(parent, action,args)
        self.answer = extract_answer(self.action,args.data_name) if self.action else ""
        self.token_use = 0
        self.child_answer_list = []
        self.child_answer_count = {}

    
    def add_child_value(self, answer): 
        """把子节点返回的 answer 记录到列表和计数字典中。"""
        if answer:
            self.child_answer_list.append(answer)
            self.num_visits += 1
            for rep in self.child_answer_count.keys():
                if math_equal(answer, rep):
                    self.child_answer_count[rep] += 1
                    break
            else:
                self.child_answer_count[answer] = 1

    def get_q(self) -> float:
        # 可能用到：MESSAGE:信息熵,想好了可能会用到信息熵的想法做
        """
        返回节点的“混乱度”Q∈[0,1]:
        - 0  : 全部子答案一致（无混乱）
        - 1  : 子答案分布最均匀（最大混乱）
        """
        if not self.child_answer_list:          # 尚无数据
            return 0.0
        if len(self.child_answer_count) == 1: # 数据过少或者过于一致
            return 0.0

        counts = list(self.child_answer_count.values())
        total  = sum(counts)
        k      = len(counts)                    # 不同答案类别数

        if k == 1:
            return 0.0                          # 熵=0，完全一致

        # ① 计算香农熵  H = -Σ p_i log₂ p_i
        H = 0.0
        for c in counts:
            p = c / total
            H -= p * math.log(p, 2)

        # ② 归一化到 [0,1]：H_max = log₂ k
        H_max = math.log(k, 2)
        Q = H / H_max                           # 越大越混乱
        return Q
    
    def create_child(self, action, args):
        child = CON_MCTS_TreeNode(self,action,args)
        if action not in self.children.keys():
            self.children[action] = child
            self.add_child_value(child.answer)
            print(f"new child, depth is {child.depth}, answer is {child.answer}\n")


def con_mcts(tree:BaseTree, args):
    
    def follow_max(tree:BaseTree,args):
        root = tree.root
        if not root.child_answer_count and root.child_answer_list:
            for ans in root.child_answer_list:
                for rep in root.child_answer_count.keys():
                    if math_equal(ans, rep):
                        root.child_answer_count[rep] += 1
                        break
                else:
                    root.child_answer_count[ans] = 1
        if not root.child_answer_count:
            return None
        best_answer = max(
            root.child_answer_count.items(),
            key=lambda item: item[1]
        )[0]
        return best_answer


    num_iterations = args.num_iterations
    for _ in range(num_iterations):
        for _ in range(len(tree.tree_status)):
            get_next_step(tree,args)
        #TODO:这个地方是需要加入早停机制的，最终的动态推理大部分依靠的是早停机制。
    output_strategy = args.output_strategy
    if output_strategy == "max_reward":
        best_answer = follow_max(tree,args)
    else:
        best_answer = follow_max(tree,args)
    tree.final_answer = best_answer
    print(f"tree {tree.tree_id} is finished.")
    return tree



def get_next_step(tree:BaseTree, args):
    if tree.get_cur_state() == "select":
        select(tree,args)
    elif tree.get_cur_state() == "expand":
        expand(tree,args)
    elif tree.get_cur_state() == "simulate":
        simulate(tree,args)
    elif tree.get_cur_state() == "back_propagate":
        back_propagate(tree,args)
    tree.step()

def select(tree:BaseTree, args):
    def is_select_terminal(curnode:TreeNode):
        if curnode.depth >= depth_limit:
            return True
        elif curnode.is_terminal:
            return True
        elif len(curnode.children.values()) == 0:
            return True
        print("bbbbbbbbbbbbbbbbbbbbb")
        return False
    
    def uct_select(curnode:TreeNode, args):
        best_value = -math.inf
        best_node = []
        for child in curnode.children.values():
            assert isinstance(child, CON_MCTS_TreeNode)
            if child.num_visits > 0:
                node_value = child.get_q() + args.explore_constant * math.sqrt(2 * math.log(max(curnode.num_visits, 1)) / child.num_visits)
            elif child.num_visits == 0 and args.uct_type == "inf":
                node_value = child.get_q() + args.uct_inf
            elif child.num_visits == 0 and args.uct_type == "sqrt_parent":
                node_value = child.get_q() + args.explore_constant * math.sqrt(2 * math.log(max(curnode.num_visits,1)))

            if node_value > best_value:
                best_node = [child]
                best_value = node_value
            elif node_value == best_value:
                best_node.append(child)
        return random.choice(best_node)
    
    depth_limit = args.mcts_depth_limit
    curnode = tree.root
    while not is_select_terminal(curnode):
        print("a")
        curnode = uct_select(curnode, args)
        print(f"tree_id:{tree.tree_id}, is_select_terminal:{is_select_terminal(curnode)}, depth:{curnode.depth}")
        print_time_exceed_warning(cost_time=time.time() - tree.starttime,input=f"tree:{tree.tree_id}, state:select")
    tree.curnode = curnode

def expand(tree:CON_MCTS_Tree,args):
    def get_prompt(node:TreeNode, actions=None):
        input_prompt = ""
        states = []
        cur_node = node
        while cur_node is not None:
            if cur_node.depth == 0:
                states.append(cur_node.state)
            else:
                states.append(cur_node.action)
            cur_node = cur_node.parent
        
        states.reverse()

        for i, s in enumerate(states):
            if not s:   # None, 空容器, 0, False, 空张量(若定义了 bool) 等都会触发
                print(states,flush=True)
                print(f"[collect_states] Empty/falsy element at index {i}: {s!r}", flush=True)
                # raise ValueError(f"states[{i}] is empty.")
        
        for i,s in enumerate(states):
            input_prompt += s
        return input_prompt
    
    def get_action(node:CON_MCTS_TreeNode,args):
        input_prompt = get_prompt(node)
        rethinking_token = get_rethinking_token(args)
        stop_tokens = get_stop_tokens(args)
        n_sampling = args.n_sampling

        model_input = input_prompt + rethinking_token
        outputs = completion_request(question=model_input,args=args,tree_id=tree.tree_id,model_id=args.model_name_or_path,stop_tokens=stop_tokens,n_sampling=n_sampling)
        actions = [rethinking_token + output for output in outputs["texts"][0]]
        print(f"the length of action is {len(actions[0])}")
        recheck_action_list = []
        # 这里或许可以加一个如果extract不到答案重复生成的内容
        
        for i,action in enumerate(actions):
            recheck_bool, recheck_act = recheck_action(tree,node,action,args)
            if recheck_bool == 0:
                model_input = input_prompt + action
                outputs = completion_request(question=model_input,args=args,tree_id=tree.tree_id,model_id=args.model_name_or_path,stop_tokens=stop_tokens,n_sampling=1)
                new_action = action + outputs["texts"][0][0]
                actions[i] = new_action
                recheck_bool, recheck_act = recheck_action(tree,node,new_action,args)
            
            recheck_action_list.append((recheck_bool,recheck_act))

        for i,action in enumerate(actions):
            recheck_bool, recheck_act = recheck_action_list[i]
            print(f"the recheck bool is {recheck_bool}")
            if recheck_bool == 1:
                node.create_child(recheck_act,args)
                child = node.children[recheck_act]
                child.state = input_prompt
                child.token_use = get_token_usage(tree.tokenizer,recheck_act,args)
                tree.new_child_list.append(child)
            elif recheck_bool == -1:
                node.create_child(action,args)
                child = node.children[action]
                child.state = input_prompt
                child.token_use = get_token_usage(tree.tokenizer,action,args)
                tree.new_child_list.append(child)
            elif recheck_bool == -2:
                node.create_child(action,args)
                child = node.children[action]
                child.state = input_prompt
                child.token_use = get_token_usage(tree.tokenizer,action,args)
                tree.new_child_list.append(child)
            else:
                print("one action is not finished")
                continue # TODO:如果recheck_bool == 0就不好处理了，就代表未完成一次回答。
        
    expand_node = tree.curnode
    get_action(expand_node,args)

def recheck_action(tree:CON_MCTS_Tree,node:CON_MCTS_TreeNode,action,args):
    # return is (int,str|None), the first element means whether the action is completed.
    # 0 is not completed.
    # -1 is request timeout.
    # 1 is completed and the second element is the action.
    # -2 only when args.recheck_action == False
    def check_format():
        # TODO:the usage of this function is to check whether the response is formated.
        # use regularization
        #         """
        # 判定 LLM 输出是否符合下列两种格式之一：
        # 1. 以  "Yes, the first completed answer is: {json}"  开头（{json} 必须能被 json.loads 正确解析）
        # 2. 以  "No,"  开头（大小写敏感，后续内容不限）
        # 若满足其一，返回 True；否则返回 False
        # """
        if not text:
            return False
        text = text.strip()
        # —— 情况 1: Yes + JSON ——
        yes_pat = re.compile(
            get_rechecking_pattern(),
            re.S  # 让 "." 匹配换行
        )
        m = yes_pat.match(text)
        if m:
            json_part = m.group(1)
            try:
                json.loads(json_part)  # 验证 JSON 是否有效
                return True
            except json.JSONDecodeError:
                return False
        # —— 情况 2: No, … ——
        no_pat = re.compile(r'^No,\s*.*$', re.S)
        if no_pat.match(text):
            return True
        return False
    
    def extract_format(extract_action):
        text = text.strip()

        # —— 格式 1: "Yes, the first completed answer is: {json}"
        yes_pat = re.compile(
            r'^Yes,\s*the first completed answer is:\s*(\{.*\})\s*$', 
            re.S  # 让 "." 匹配换行
        )
        m = yes_pat.match(text)
        if m:
            json_part = m.group(1)
            try:
                data = json.loads(json_part)
                return 1, data            # 匹配成功 + 返回 JSON
            except json.JSONDecodeError:
                # JSON 解析失败 → 视为格式不合法
                return -1, None

        # —— 格式 2: "No, ..."
        if text.startswith("No,"):
            return 0, None                 # 匹配成功，但无 JSON

        # —— 全部不匹配
        return -1, None
    

    if not args.recheck_action:
        return -2,None
    rechecking_prompt = get_rechecking_prompt(args)
    input_prompt = rechecking_prompt.format(question=tree.question,action=action)
    outputs = completion_request(question=input_prompt,args=args,tree_id=tree.tree_id,model_id=args.model_name_or_path,n_sampling=1)
    output = outputs["text"][0]
    cnt = 3
    while not check_format(output):
        cnt -= 1
        if cnt == 0:
            break
        outputs = completion_request(question=input_prompt,args=args,tree_id=tree.tree_id,model_id=args.model_name_or_path,n_sampling=1)
        output = outputs["text"][0]
    if cnt == 0:
        return -1,None
    else:
        return extract_format(output) 



def simulate(tree:CON_MCTS_Tree,args):
    return

def back_propagate(tree:CON_MCTS_Tree,args):
    print_time_exceed_warning(cost_time=time.time() - tree.starttime,input=f"tree:{tree.tree_id}, state:back_propagate")
    for child in tree.new_child_list:
        curnode = child
        while curnode is not None:
            if curnode.parent is not None:
                curnode.parent.add_child_value(child.answer)
            curnode = curnode.parent
    tree.new_child_list = []



