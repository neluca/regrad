from regrad.variable import Var


def get_node_info(node: Var) -> tuple[str, ...]:
    node_name = node.name
    node_data = str(node.val)
    node_args = ""
    if node.op is not None:
        node_args = f"{node.op.op_args}"
    return node_name, node_args, node_data
