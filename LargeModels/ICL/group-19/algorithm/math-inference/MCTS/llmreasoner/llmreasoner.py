# MESSAGE: 接下来的内容都是定制化的，

from .examples import get_llm_reasoner_examples
import math
import random
from utils.serve_vllm import completion_request
from utils.grader import math_equal
from utils.utils import print_time_exceed_warning
from typing import Literal,Callable
import numpy as np
import time
from utils.strip_string import extract_answer
from ..base_tree import BaseTree, TreeNode

class LLM_REASONER_Tree(BaseTree):
    def __init__(self, tree_id, question, gt_ans, gt_cot ,args):
        super().__init__(tree_id, question, gt_ans, gt_cot ,args)
        root = LLM_REASONER_TreeNode(None,None,args)
        self.curnode:LLM_REASONER_TreeNode = root
        self.root:LLM_REASONER_TreeNode = root

class LLM_REASONER_TreeNode(TreeNode):
    def __init__(self, parent, action,args):
        super().__init__(parent, action,args)
        self.confidence = 0.8
        self.self_useful_judge = 0.5
        self.answer = ""
        self.fast_reward = 0
        self.cum_rewards = []
    def get_q(self):
        if self.state:
            return np.mean(self.cum_rewards)
        else:
            return self.fast_reward
    
    def get_state(self):
        state_list = []
        curnode = self
        while curnode.parent is not None:
            state_list.append((curnode.action,curnode.state,curnode.confidence))
            curnode = curnode.parent
            
        state_list.reverse()
        return state_list
    def set_fast_reward(self,fast_reward,args):
        self.fast_reward = fast_reward
        # if self.value == 0:
        #     self.value = fast_reward
    def create_child(self,action,args):
        child = LLM_REASONER_TreeNode(self,action,args)
        if action not in self.children.keys():
            self.children[action] = child


def get_next_step(tree: BaseTree, args):
    if tree.get_cur_state() == "select":
        # print("a\n")
        select(tree,args)
    elif tree.get_cur_state() == "expand":
        expand(tree,args)
    elif tree.get_cur_state() == "simulate":
        simulate(tree,args)
    elif tree.get_cur_state() == "back_propagate":
        back_propagate(tree,args)
    tree.step()


def llm_reasoner_mcts(tree:BaseTree, args):
    def dfs_max_reward(node:TreeNode, args):
        cur = node
        if is_terminal(cur,args):
            return cur.value,node
        elif len(node.children.values()) == 0:
            return -math.inf, node
        visited_children = [x for x in cur.children.values() if x.state]
        if len(visited_children) == 0:
            return -math.inf, node
        max_value = -math.inf
        max_node = cur
        for visited_child in visited_children:
            value,dfs_node = dfs_max_reward(visited_child,args)
            if value > max_value:
                max_value = value
                max_node = dfs_node
        max_value += cur.value
        return max_value, max_node
    
    tree.starttime = time.time()
    
    num_iterations = args.num_iterations
    final_reward = -math.inf
    node = None
    output_strategy = args.output_strategy

    for iteration in range(num_iterations):
        for steps in range(len(tree.tree_status)):
            get_next_step(tree,args)
    if output_strategy == "max_reward":
        final_reward,node = dfs_max_reward(tree.root,args)
    tree.final_node = node
    tree.final_answer = extract_answer(pred_str=node.state, data_name=args.data_name)
    print(f"tree {tree.tree_id} is finished.")
    return tree

def should_expand_action_and_state(node:TreeNode, args): # 0代表既不生成子节点也不生成state， 1表示只生成state，2表示生成state也生成action
    if node.depth > args.mcts_depth_limit:
        return 0
    elif node.is_terminal:
        return 0
    elif len(node.children.values()) != 0:
        return 0
    elif get_llm_reasoner_examples()["last_question_query"] in node.action and node.state == "":
        return 1
    elif node.depth == args.mcts_depth_limit:
        return 1
    else:
        return 2

def is_terminal(node:TreeNode, args):
    action = node.action
    if get_llm_reasoner_examples()["last_question_query"] in action or node.depth >= args.mcts_depth_limit:
        if node.state:
            return True
    elif node.is_terminal:
        return True
    else:
        return False
    # TODO:本函数功能是判断当前生成的节点是否是terminal。

def select(tree:BaseTree,args):
    depth_limit = args.mcts_depth_limit    
    tree.curnode = tree.root
    
    def is_select_terminal(curnode:TreeNode):
        if curnode.depth >= depth_limit:
            return True
        elif curnode.is_terminal:
            return True
        elif len(curnode.children.values()) == 0:
            return True
        return False
    
    def uct_select(curnode:TreeNode, args):
        best_value = -math.inf
        best_node = []
        for child in curnode.children.values():
            assert isinstance(child, LLM_REASONER_TreeNode)
            if child.num_visits != len(child.cum_rewards):
                raise AssertionError("num visit is not equal with len of cum_rewards")
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
    
    while not is_select_terminal(tree.curnode):
        tree.curnode = uct_select(tree.curnode, args)
        print_time_exceed_warning(cost_time=time.time() - tree.starttime,input=f"tree:{tree.tree_id}, state:select")
        # print(f"tree_id:{tree.tree_id}, is_select_terminal:{is_select_terminal(tree.curnode)}, depth:{tree.curnode.depth}")

def expand(tree:BaseTree,args):
    def get_prompt(node:TreeNode,type:Literal["action","state"],args):
        num_shots = args.num_shots
        examples = get_llm_reasoner_examples()
        states = node.get_state()
        input_prompt = ""
        input_prompt += examples["format_prompt"]
        input_examples = examples["format_examples"][:num_shots]
        for input_example in input_examples:
            for key,value in input_example.items():
                input_prompt += key
                input_prompt += value
                input_prompt += "\n\n"
            input_prompt += "\n\n"
        input_question = examples["format_question"].format(idx=num_shots + 1, question=tree.question)
        input_prompt += input_question
        input_prompt += "\n\n"
        cnt = 0
        for subid,(question,answer,confidence) in enumerate(states):
            if question:
                f_subquestion = examples["format_subquestion"].format(idx=num_shots + 1, sub_idx=subid + 1)
                input_prompt += f_subquestion
                input_prompt += question
                input_prompt += "\n"
                cnt += 1
            if answer:
                f_subanswer = examples["format_subanswer"].format(idx=num_shots + 1, sub_idx=subid + 1)
                input_prompt += f_subanswer
                input_prompt += answer
                input_prompt += "\n"
                cnt += 1 
            
        if type == "action":
            if cnt != 2 * len(states):
                print(input_prompt)
                # assert cnt == 2 * len(states)
            format_subquestion = examples["format_subquestion"].format(idx=num_shots + 1, sub_idx=len(states) + 1)
            input_prompt += format_subquestion
            if node.depth == args.mcts_depth_limit - 1:
                input_prompt += examples["format_last_question"]
        elif type == "state":
            if cnt != 2 * len(states) - 1:
                print(input_prompt)
                # assert cnt == 2 * len(states) - 1
            format_subanswer = examples["format_subanswer"].format(idx=num_shots + 1, sub_idx=len(states))
            input_prompt += format_subanswer
        return input_prompt
    
    def get_actions(node:TreeNode,args):
        num_shots = args.num_shots
        examples = get_llm_reasoner_examples()
        # print(f"examples: ex")
        input_prompt = get_prompt(node,"action",args)
        
        model_input = input_prompt
        stop_tokens = [f"Answer {num_shots+1}.", "Question"]
        n_actions = 1 if node.depth == args.mcts_depth_limit - 1 else args.n_actions
        n_batch = args.n_batch

        for start1 in range(0,n_actions,n_batch):
            stop1 = min(start1 + n_batch, n_actions)
            n_sampling = stop1 - start1
            outputs = completion_request(question=model_input,args=args,tree_id=tree.tree_id,stop_tokens=stop_tokens,n_sampling=n_sampling)["texts"]
            outputs = outputs[0]
            outputs = [output.strip() for output in outputs]
            if node.depth == args.mcts_depth_limit - 1:
                outputs = [examples["format_last_question"] + ' ' + output for output in outputs]
            outputs = list(dict.fromkeys(outputs))
            
            actions = outputs
            for action in actions:
                if not action:
                    text = {
                        "input":input_prompt,
                        "outputs":outputs,
                        "tree_id": tree.tree_id
                    }
                    print("no question : ",flush=True)
                    print(text,flush=True)
                else:
                    node.create_child(action,args)
        return
        # MESSAGE:本函数功能是为当前节点生成一系列接下来要解决的子问题。输入为节点和args，输出是模型生成的当前子问题。

    def get_state(node:TreeNode,args):
        num_shots = args.num_shots
        input_prompt = get_prompt(node,"state",args)

        model_input = input_prompt
        stop_tokens = []
        # stop_token1 = f"Question {num_shots+2}" if "Now we can answer the final question" in node.action else f"Question {num_shots+1}."
        stop_token1 = ["Question ", "Given a question"]
        stop_tokens.extend(stop_token1)
        n_confidence = args.n_confidence
        n_batch = args.n_batch
        early_stop_confidence = args.early_stop_confidence

        answer_dict = {}  # map from answer to answers per num
        text_dict = {}
        answer_number = 0
        answer_map = {} # 用于和answer_dict之间取得联系，知道每个int键代表什么答案
        for start1 in range(0,n_confidence,n_batch):
            stop1 = min(start1 + n_batch, n_confidence)
            n_sampling = stop1 - start1
            outputs = completion_request(question=model_input,args=args,tree_id=tree.tree_id,stop_tokens=stop_tokens,n_sampling=n_sampling)["texts"]
            outputs = outputs[0]
            for output in outputs:
                result = output.strip()
                answer = extract_answer(pred_str=result,data_name=args.data_name)
                find = False
                for i,answer_i in answer_map.items():
                    if math_equal(answer_i,answer):
                        find = True
                        answer_dict[i] = answer_dict[i] + 1
                        break
                if not find:
                    answer_dict[answer_number] = 1
                    answer_map[answer_number] = answer
                    text_dict[answer_number] = result
                    answer_number += 1
            if len(answer_dict) == 0:  # no answer yet
                continue
            max_value = max(answer_dict.values())
            # 找到所有对应最大值的键
            max_keys = [key for key, value in answer_dict.items() if value == max_value]

            if max_value / stop1 >= early_stop_confidence:
                if len(answer_dict) >= 2 and len(max_keys) != 1:
                    pass  # Tie with the second best answer
                else:
                    break
            
        if len(answer_dict) == 0:
            print("Warning: no answer found")
            confidence, answer,text = 0, result,result
        else:
            max_value = max(answer_dict.values())
            # 找到所有对应最大值的键
            max_keys = next(key for key, value in answer_dict.items() if value == max_value)
            answer = answer_map[max_keys]
            total = sum(answer_dict.values())
            confidence = max_value / total
            text = text_dict[max_keys]
        final_dict = {
            "state": text,
            "action":node.action,
            "answer":answer,
            "confidence":confidence
        }
        assert isinstance(node,LLM_REASONER_TreeNode)
        node.confidence = confidence
        node.refine_state(text)
        if not text:
            print(final_dict,flush=True)
            raise ValueError()
        node.answer = answer
        node.is_terminal = is_terminal(node,args)
        # MESSAGE:本函数功能是根据当前子问题直接生成答案。输入为节点动作和节点和args，输出是模型生成的当前子问题。

    expand_node = tree.curnode
    expand_evalutate_type = ""
    if expand_node.depth == 0:
        get_actions(expand_node,args)
        expand_evalutate_type = "action_only"
    elif should_expand_action_and_state(expand_node,args) == 2:
        get_state(expand_node,args)
        get_actions(expand_node,args)
        expand_evalutate_type = "state_action"
    elif should_expand_action_and_state(expand_node,args) == 1:
        get_state(expand_node,args)
        expand_evalutate_type = "state_only"
    expand_evalutate(tree,expand_evalutate_type,args)


def expand_evalutate(tree:BaseTree,expand_evaluate_type,args):
    def reward(node:LLM_REASONER_TreeNode,args):
        n_confidence = node.confidence
        self_useful_judge = node.self_useful_judge
        reward_alpha = args.reward_alpha
        node.value = self_useful_judge ** reward_alpha * n_confidence ** (1 - reward_alpha)
    
    def fast_reward(node:LLM_REASONER_TreeNode,args):
        num_shots = args.num_shots
        examples = get_llm_reasoner_examples()
        states = node.get_state()
        input_prompt = ""
        input_prompt += examples["useful_prompt"]
        useful_examples = examples["useful_examples"][:num_shots]
        for useful_example in useful_examples:
            for key,value in useful_example.items():
                input_prompt += key
                input_prompt += value
                input_prompt += "\n"
            input_prompt += "\n\n"
        input_question = examples["format_question"].format(idx=num_shots + 1, question=tree.question)
        input_prompt += input_question
        input_prompt += "\n"
        cnt = 0
        for subid,(question,answer,confidence) in enumerate(states):
            if question:
                f_subquestion = examples["format_subquestion"].format(idx=num_shots + 1, sub_idx=subid + 1)
                input_prompt += f_subquestion
                input_prompt += question
                input_prompt += "\n"
                cnt += 1
        # assert cnt == node.depth
        f_subquestion = examples["format_subquestion"].format(idx=num_shots + 1, sub_idx=len(states) + 1)
        input_prompt += f_subquestion

        children_list = list(node.children.values())
        input_prompts = [input_prompt] * len(children_list)
        for child_id, child in enumerate(children_list):
            new_question = child.action
            input_prompts[child_id] = input_prompts[child_id] + new_question + "\n"
            prefix = examples["useful_prefix"].format(idx=num_shots + 1, sub_idx=len(states) + 1)
            input_prompts[child_id] = input_prompts[child_id] + prefix
        
        outputs = completion_request(input_prompts,args=args,tree_id=tree.tree_id,max_tokens=1,need_get_next_token_prob=True)["logits"]
        reward_alpha = args.reward_alpha
        children_list = list(node.children.values())
        for output_id, output in enumerate(outputs):
            output = output[0]
            fast_reward_node = children_list[output_id]
            assert isinstance(fast_reward_node,LLM_REASONER_TreeNode)
            prob_map = {tok.strip().lower(): p for tok, p in output.items()}
            yes_prob = prob_map.get("yes", 0.0)
            no_prob  = prob_map.get("no", 0.0)

            denom = yes_prob + no_prob
            if denom > 0:
                useful_prob = yes_prob / denom
            else:
                # 如果两者都缺失，给个默认值
                useful_prob = 0.5
            
            fast_reward = useful_prob ** reward_alpha * fast_reward_node.confidence ** (1 - reward_alpha)
            fast_reward_node.self_useful_judge = useful_prob
            fast_reward_node.set_fast_reward(fast_reward,args)
        #每个节点都要fast_reward所以这个地方需要for循环改进一下。
        return 
    
    expand_node = tree.curnode
    if expand_evaluate_type == "action_only":
        fast_reward(expand_node,args)
    elif expand_evaluate_type == "state_action":
        reward(expand_node,args)
        fast_reward(expand_node,args)
    elif expand_evaluate_type == "state_only":
        reward(expand_node,args)
    # else:
    #     raise ValueError("Undetected evaluate type")


def simulate(tree:BaseTree,args):  # FUTUREWORK:args增加一个选项是否将rollout后的结果加入到树里面？
    def greedy_policy(node:TreeNode,args):
        return
    # FUTUREWORK: policy是step式的模拟，要经过多步，每一步都根据预估价值选最大的。
    def fast_rollout(node:TreeNode,args):
        return
    # FUTUREWORK: fast_rollout是直接生成最终回答，可以直接用expand_last_step里面的。
    def random_policy(node:TreeNode,args):
        return
    # FUTUREWORK: policy是step式的模拟，要经过多步，每一步都随机选择。

    def sigmoid_sample(x: list[float] | np.ndarray):
        x = np.asarray(x, dtype=float)
        # ---- ① 数值稳定的 sigmoid ----
        # 直接 1/(1+exp(-x)) 在 x 很大或很小时可能溢出，下面做片段处理
        # 大于 20 基本可看作 1，小于 -20 基本看作 0
        z = np.clip(x, -20, 20)
        probs = 1.0 / (1.0 + np.exp(-z))      # ∈ (0,1)
        # ---- ② 归一化为真正的概率 ----
        s = probs.sum()
        if s == 0 or not np.isfinite(s):       # 全 0 或数值异常的兜底
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= s
        # ---- ③ 采样 ----
        return int(np.random.choice(len(probs), p=probs))

    simulate_node = tree.curnode
    default_simulate_strategies: dict[str, Callable[[list[float]], int]] = {
            'max': lambda x: np.argmax(x),
            'sample': sigmoid_sample,
            'random': lambda x: np.random.choice(len(x)),
        }
    simulate_choice: Callable[[list[float]], int] = default_simulate_strategies.get(args.simulate_strategy)
    while True:
        print_time_exceed_warning(cost_time=time.time() - tree.starttime,input=f"tree:{tree.tree_id}, state:simulate")
        if is_terminal(simulate_node,args) or len(simulate_node.children) == 0:
            tree.curnode = simulate_node
            return
        children_list = list(simulate_node.children.values())
        fast_rewards = [child.fast_reward for child in children_list]
        chosen_idx = simulate_choice(fast_rewards)
        simulate_node = children_list[chosen_idx]
        tree.curnode = simulate_node
        expand(tree,args)



def back_propagate(tree:BaseTree,args):
    back_propagate_node = tree.curnode
    assert isinstance(back_propagate_node, LLM_REASONER_TreeNode)
    rewards = []
    while back_propagate_node is not None:
        print_time_exceed_warning(cost_time=time.time() - tree.starttime,input=f"tree:{tree.tree_id}, state:back_propagate")
        back_propagate_node.num_visits += 1
        rewards.append(back_propagate_node.value)
        cum_reward = sum(rewards[::-1])
        back_propagate_node.cum_rewards.append(cum_reward)
        back_propagate_node = back_propagate_node.parent
    #FUTUREWORK: 一个是sum路径上的总reward,一个是和其它子节点取平均值。这个地方先实现的是取sum路径上的总reward，但是我觉得他做的很不对。
    return

