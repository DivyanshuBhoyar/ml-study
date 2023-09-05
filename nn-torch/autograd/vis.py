from graphviz import Digraph
import numpy as np


def trace(root):
    # Initialize empty sets for nodes and edges
    nodes, edges = set(), set()

    # Recursive function to build the set of nodes and edges
    def build(v):
        # If the current node is not in the set of nodes
        if v not in nodes:
            # Add the current node to the set of nodes
            nodes.add(v)

            # Iterate over the previous nodes (parents) of the current node
            for child in v._prev:
                # Add an edge from the previous node to the current node
                edges.add((child, v))

                # Recursively call build() on the previous node
                build(child)

    # Start building the set of nodes and edges from the root node
    build(root)

    # Return the set of nodes and edges
    return nodes, edges


def draw_dot(root):
    # Create a Digraph object with SVG format and set the graph attribute 'rankdir' to 'LR'
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})

    # Retrieve the nodes and edges from the root
    nodes, edges = trace(root)

    # Iterate over the nodes
    for n in nodes:
        uid = str(id(n))
        # Convert the data array to a string representation
        data_str = np.array2string(
            n.data, precision=4, separator=",", suppress_small=True
        )
        # Convert the gradient array to a string representation
        grad_str = np.array2string(
            n.grad, precision=4, separator=",", suppress_small=True
        )
        # Create the label for the node using the node's label, data, and gradient strings
        label = "{ %s | data %s | grad %s }" % (n.label, data_str, grad_str)
        # Add a node to the graph with a unique ID, label, and shape 'record'
        dot.node(name=uid, label=label, shape="record")

        # If the node has an operation associated with it
        if n._op:
            # Add a node to the graph with a unique ID based on the node's ID and operation, and label it with the operation
            dot.node(name=uid + n._op, label=n._op)
            # Add an edge from the operation node to the current node
            dot.edge(uid + n._op, uid)

    # Iterate over the edges
    for n1, n2 in edges:
        # Add an edge from the first node's ID to the second node's ID concatenated with the operation name
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    # Return the generated graph
    return dot
