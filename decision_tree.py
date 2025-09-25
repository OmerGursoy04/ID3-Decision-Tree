import argparse
import math
import csv

class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None
        self.vote = None

#parse .tsv file
def collect_tData(train_input):
    with open(train_input, 'r') as f_in:
        reader = csv.reader(f_in, delimiter = "\t")
        rows = next(reader)
        attribute_names = rows[:-1]
        attributes = list(range(len(attribute_names)))  
        array = []
        for row in reader:
            tmp = []
            for cell in row:
                tmp.append(int(cell))
            array.append(tmp)

    return array, attribute_names, attributes

#counts how many elements with 1 as the feature value 
def ones_features(array, attribute):
    return sum(map(lambda x: x[attribute], array))
#counts how many elements with 0 as the feature value 
def zeros_features(array, attribute): 
    return len(array) - ones_features(array, attribute)

#counts how many elements with 1 as the label 
def ones_labels(array):
    return sum(map(lambda x: x[-1], array))
#counts how many elements with 0 as the label 
def zeros_labels(array):
    return len(array) - ones_labels(array)

def majority_vote(array):
    return 1 if ones_labels(array) >= zeros_labels(array) else 0

def entropy(array):
    if len(array) == 0:
        return 0
    p1 = ones_labels(array) / len(array)
    p0 = 1 - p1
    entropy = -(p0 * math.log2(p0) + p1 * math.log2(p1))

    return entropy

def entropy_by_attribute(array, attribute):
    if len(array) == 0:
        return 0
    feat0_lab0 = sum(1 for row in array if row[attribute] == 0 and row[-1] == 0)
    feat0_lab1 = sum(1 for row in array if row[attribute] == 0 and row[-1] == 1)
    feat1_lab1 = sum(1 for row in array if row[attribute] == 1 and row[-1] == 1)
    feat1_lab0 = sum(1 for row in array if row[attribute] == 1 and row[-1] == 0)

    zeros = feat0_lab0 + feat0_lab1
    ones = feat1_lab1 + feat1_lab0

    p0 = zeros / len(array)
    p1 = ones / len(array)

    h0 = 0
    if feat0_lab0 > 0:
        h0 += feat0_lab0 / zeros * math.log2(feat0_lab0 / zeros) 
    if feat0_lab1 > 0:
        h0 += feat0_lab1 / zeros * math.log2(feat0_lab1 / zeros)
    h0 *= -1

    h1 = 0
    if feat1_lab1 > 0:
        h1 += feat1_lab1 / ones * math.log2(feat1_lab1 / ones)
    if feat1_lab0 > 0:
        h1 += feat1_lab0 / ones * math.log2(feat1_lab0 / ones)
    h1 *= -1

    entropy = (p0 * h0) + (p1 * h1)

    return entropy

def mutual_information(array, attribute):
    MI = entropy(array) - entropy_by_attribute(array, attribute)
    return MI

def is_node_pure(array):
    return ones_labels(array) == 0 or zeros_labels(array) == 0

def split_node(array, attribute):
    left, right = [], []
    for row in array:
        if row[attribute] == 0:
            left.append(row)
        else:
            right.append(row)

    return left, right

def choose_attribute(array, attributes):
    highest_MI = -1
    best_attribute_idx = None
    for att_i in attributes:
        curr_MI = mutual_information(array, att_i)
        if curr_MI > highest_MI:
            highest_MI = curr_MI
            best_attribute_idx = att_i
        
    return best_attribute_idx, highest_MI

def build_tree(array, attributes, curr_depth, max_depth):
    node = Node()
    remaining_attributes = attributes.copy()
    if len(array) == 0 or curr_depth >= max_depth:
        node.vote = majority_vote(array)
        return node
    if is_node_pure(array) or not attributes:
        node.vote = majority_vote(array)
        return node
    else:
        best_attribute_idx, highest_MI = choose_attribute(array, remaining_attributes)
        if best_attribute_idx is None or highest_MI <= 0:
            node.vote = majority_vote(array)
            return node
        remaining_attributes.remove(best_attribute_idx)
        left_branch, right_branch = split_node(array, best_attribute_idx)
        if left_branch == [] or right_branch == []:
            node.vote = majority_vote(array)
            return node
        node.attr = best_attribute_idx
        node.left = build_tree(left_branch, remaining_attributes, curr_depth + 1, max_depth)
        node.right = build_tree(right_branch, remaining_attributes, curr_depth + 1, max_depth)

        return node

def predict_single_node(node, row):
    if node.vote is not None:
        return node.vote
    else:
        if row[node.attr] == 0:
            return predict_single_node(node.left, row)
        else:
            return predict_single_node(node.right, row)

def predict_tree(node, array):
    predictions = []
    for row in array:
        predictions.append(predict_single_node(node, row))
    return predictions

def error_rate(node, array):
    if len(array) == 0:
        return 0

    incorrect = 0
    for row in array:
        prediction = predict_single_node(node, row)
        if row[-1] != prediction:
            incorrect += 1

    return incorrect / len(array)

def print_tree(node, attr_names, array, depth = 0, parent_attr_i = None, branch_val = None, out = None):
    if out is None:
        out = []

    num_zeros = sum(1 for r in array if r[-1] == 0)
    num_ones = len(array) - num_zeros

    if parent_attr_i is None:
        out.append(f"[{num_zeros} 0/{num_ones} 1]")
    else:
        prefix = "| " * depth
        out.append(f"{prefix}{attr_names[parent_attr_i]} = {branch_val}: [{num_zeros} 0/{num_ones} 1]")

    if node.attr is None:
        return out

    left  = [r for r in array if r[node.attr] == 0]
    right = [r for r in array if r[node.attr] == 1]
    print_tree(node.left,  attr_names, left,  depth + 1, node.attr, 0, out)
    print_tree(node.right, attr_names, right, depth + 1, node.attr, 1, out)
    
    return out

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, 
                        help='maximum depth to which the tree should be built')
    parser.add_argument("train_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the training data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the test data should be written')
    parser.add_argument("metrics_out", type=str, 
                        help='path of the output .txt file to which metrics such as train and test error should be written')
    parser.add_argument("print_out", type=str,
                        help='path of the output .txt file to which the printed tree should be written')
    args = parser.parse_args()

    print_out = args.print_out

    train_input = args.train_input
    test_input = args.test_input
    max_depth = args.max_depth
    train_out = args.train_out
    test_out = args.test_out
    metrics_out = args.metrics_out
    print_out = args.print_out

    train_array, attr_names, attributes = collect_tData(train_input)
    test_array, attr_names, attributes = collect_tData(test_input)

    node = build_tree(train_array, attributes, 0, max_depth)

    predictions_train = predict_tree(node, train_array)
    predictions_test  = predict_tree(node, test_array)

    train_error = error_rate(node, train_array)
    test_error  = error_rate(node, test_array)

    with open(train_out, "w") as f_out:
        for y in predictions_train: 
            f_out.write(f"{y}\n")

    with open(test_out, "w") as f_out:
        for y in predictions_test:
            f_out.write(f"{y}\n")

    with open(metrics_out, "w") as f_out:
        f_out.write(f"error(train): {train_error}\n")
        f_out.write(f"error(test): {test_error}\n")

    lines = print_tree(node, attr_names, train_array)
    with open(print_out, "w") as f_out:
        f_out.write("\n".join(lines) + "\n")