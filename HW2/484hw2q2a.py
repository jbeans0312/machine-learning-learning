import pandas as pd
import math as m
import numpy as np

# load data from csv file
data = pd.read_csv('question2.csv')
features = ['y', 'x1', 'x2', 'x3', 'x4']
features.remove('y')


# create node class
class Node:
    def __init__(self):
        self.children = []
        self.value = ""
        self.isLeaf = False
        self.prediction = ""


"""
    get_entropy(curr_data)
    Returns the entropy of a given data set
    Params: pandas dataframe
    Returns: float representing entropy 
"""


def get_entropy(curr_data):
    pos = 0.0
    neg = 0.0

    for _, row in curr_data.iterrows():
        if row["y"] == 1:
            pos += 1
        else:
            neg += 1
    if pos == 0.0 or neg == 0.0:
        return 0.0
    else:
        pos_ratio = pos/(pos+neg)
        neg_ratio = neg/(pos+neg)
        return -(pos_ratio * m.log(pos_ratio, 2)) - neg_ratio * m.log(neg_ratio, 2)


"""
    information_gain(curr_data, attr)
    Params: pandas dataframe and a target attribute to test
    Returns: float representing the information gain of a given attribute
"""


def information_gain(curr_data, attr):
    values = np.unique(curr_data[attr])
    gain = get_entropy(curr_data)

    for val in values:
        subdata = curr_data[curr_data[attr] == val]

        sub_entropy = get_entropy(subdata)
        gain -= (float(len(subdata)) / float(len(curr_data))) * sub_entropy
    return gain


"""
    ID3(curr_data, attrs)
    Runs the ID3 algorithm on a given data set and a list attributes
    Calculates the max information gain at each step and recursively constructs a tree using the Node class
    Params: pandas dataframe and list of attributes
    Returns: the root node of a decision tree
"""


def id3(curr_data, attrs):
    root_node = Node()

    max_gain = 0
    max_feat = ''

    # Calculates max information gain across the current attributes
    for feature in attrs:
        gain = information_gain(curr_data, feature)
        if gain > max_gain:
            max_gain = gain
            max_feat = feature
    root_node.value = max_feat

    print("\n==========================\n")
    print("Current entropy is: ", get_entropy(curr_data))
    print("Max gain is: ", max_gain, " from attribute: ", max_feat)

    # get all values of the attribute with the max information gain
    # if max_feat = 'x2', uniq = [0,1]
    values = np.unique(curr_data[max_feat])

    # for each value that the max_feat could take, we need to find the DT that follows
    # if max_feat = 'x2' we need to find the corresponding DTs for 0 and 1
    for val in values:
        # get the corresponding data for the current value of max_feat
        subdata = curr_data[curr_data[max_feat] == val]
        # recursively find the DT of the given value of the max feature
        if get_entropy(subdata) == 0.0:  # we have arrived at a leaf node
            leaf_node = Node()
            leaf_node.isLeaf = True
            leaf_node.value = val
            leaf_node.prediction = np.unique(subdata["y"])
            root_node.children.append(leaf_node)
        else:  # we have not reached a leaf node yet, and we need to make another decision, i.e. another call to id3
            internal_node = Node()
            internal_node.value = val
            new_attrs = attrs.copy()
            new_attrs.remove(max_feat)  # prune the attributes so the max feature is not present
            child = id3(subdata, new_attrs)  # find the dt of the next level down,
            internal_node.children.append(child)
            root_node.children.append(internal_node)
    return root_node


"""
    Recursively print a given decision tree
    Params: A decision tree represented by its root Node, and the current depth that we are printing at
    Returns: none
"""


def print_tree(decision_tree: Node, depth=0):
    for i in range(depth):  # adds a tab to signify depth
        print("\t", end="")
    print(decision_tree.value, end="")
    if decision_tree.isLeaf:
        print(" -> ", decision_tree.prediction)
    print()
    for child in decision_tree.children:  # for each child of the root node, print the dt
        print_tree(child, depth + 1)


# DRIVER CODE

dt = id3(data, features)
print("\n==========================\n")
print("Decision Tree is: ")
print_tree(dt)
