from utils.strip_string import extract_answer
import time
class TreeNode:
    def __init__(self, parent,action,args):
        self.state = ""
        self.action = action if action else ""
        self.parent = parent 
        self.depth = parent.depth + 1 if parent else 0
        self.children = {}
        self.is_terminal = False
        self.num_visits = 0
        self.value = 0.0
    def get_q(self):
        return self.value
    def get_state(self):
        return self.state
    def refine_state(self,state):
        self.state = state
    def create_child(self,action,args):
        child = TreeNode(self,action,args)
        if action not in self.children.keys():
            self.children[action] = child

class BaseTree:
    def __init__(self, tree_id, question, gt_ans, gt_cot ,args):
        self.tree_id = tree_id
        self.question = question
        self.gt_ans = gt_ans
        self.gt_cot = gt_cot
        
        self.final_answer = None
        self.final_scores = []

        self.tree_status = []
        self.tree_status_index = 0
        self.tree_status.append("select")
        self.tree_status.append("expand")
        self.tree_status.append("simulate")
        self.tree_status.append("back_propagate")

        root = TreeNode(None,None,args)
        self.curnode:TreeNode = root
        self.root:TreeNode = root
        self.final_node = None

        self.starttime = time.time()

    def get_cur_state(self):
        return self.tree_status[self.tree_status_index % len(self.tree_status)]
    
    def step(self):
        self.tree_status_index += 1
    
    def get_final_ans(self): # 返回为list
        return [self.final_answer] if self.final_answer is not None else [extract_answer(self.final_node.get_state())]



