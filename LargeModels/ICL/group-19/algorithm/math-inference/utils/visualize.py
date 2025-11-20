from pyvis.network import Network
import networkx as nx
from MCTS.base_tree import TreeNode


def visualize_tree(root: TreeNode,args,html_path="tree.html"):
    from MCTS.ConMCTS.conmcts import CON_MCTS_TreeNode
    from MCTS.llmreasoner.llmreasoner import LLM_REASONER_TreeNode
    if isinstance(root,CON_MCTS_TreeNode):
        visualize_con_mcts_tree_local(root,html_path)
    elif isinstance(root, LLM_REASONER_TreeNode):
        visualize_tree_local(root,html_path)
    else:
        print("Unable to define the of the tree.",flush=True)
        if args.debug:
            print("DEBUG",flush=True)
            raise TypeError()


def visualize_con_mcts_tree_local(root: TreeNode, html_path="tree.html"):
    """
    将 TreeNode 构成的树保存为交互式 HTML 文件（适用于无图形界面服务器）

    :param root: 树的根节点，类型为 TreeNode
    :param html_path: 生成的 HTML 文件保存路径
    """
    g = nx.DiGraph()
    id_map = {}
    id_counter = [0]  # 使用列表做可变整数

    def traverse(node):
        node_id = id_counter[0]
        id_map[node] = node_id
        id_counter[0] += 1

        label = f"depth={node.depth}\nvalue={node.value:.2f}..."  # 可视化 label 简短
        tooltip = (
            f"depth={node.depth}\n"
            f"action={node.action}\n"
            f"value={node.get_q():.2f}\n"
            f"answer={node.answer}\n"
            f"visits={node.num_visits}\n"
            f"state={node.state}"
        )
        g.add_node(node_id, label=label, title=tooltip)
        for child in node.children.values():
            child_id = id_counter[0]
            g.add_edge(node_id, child_id)
            traverse(child)

    traverse(root)

    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.from_nx(g)
    net.show_buttons(filter_=['physics'])
    net.save_graph(html_path)  # 不自动打开浏览器
    print(f"[✅] 树结构已保存为 HTML: {html_path}")


def visualize_tree_local(root: TreeNode, html_path="tree.html"):
    """
    将 TreeNode 构成的树保存为交互式 HTML 文件（适用于无图形界面服务器）

    :param root: 树的根节点，类型为 TreeNode
    :param html_path: 生成的 HTML 文件保存路径
    """
    g = nx.DiGraph()
    id_map = {}
    id_counter = [0]  # 使用列表做可变整数

    def traverse(node):
        node_id = id_counter[0]
        id_map[node] = node_id
        id_counter[0] += 1

        label = f"depth={node.depth}\nvalue={node.value:.2f}..."  # 可视化 label 简短
        tooltip = (
            f"depth={node.depth}\n"
            f"action={node.action}\n"
            f"value={node.value:.2f}\n"
            f"answer={node.answer}\n"
            f"fast_reward={node.fast_reward:2f}\n"
            f"confidence={node.confidence}\n"
            f"self_useful_judge={node.self_useful_judge}\n"
            f"is_terminal={node.is_terminal}\n"
            f"visits={node.num_visits}\n"
            f"state={node.state}"
        )
        g.add_node(node_id, label=label, title=tooltip)
        for child in node.children.values():
            child_id = id_counter[0]
            g.add_edge(node_id, child_id)
            traverse(child)

    traverse(root)

    net = Network(height="800px", width="100%", directed=True, notebook=False)
    net.from_nx(g)
    net.show_buttons(filter_=['physics'])
    net.save_graph(html_path)  # 不自动打开浏览器
    print(f"[✅] 树结构已保存为 HTML: {html_path}")
