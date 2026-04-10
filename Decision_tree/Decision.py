import math

data = [
    ['sunny', 'hot', 'yes'],
    ['sunny', 'cool', 'yes'],
    ['rain', 'hot', 'yes'],
    ['rain','cool', 'no']
]
features = ['outlook', 'temp']

def entropy(data):
    length =len(data)
    yes = sum(1 for row in data if row[-1]=='yes')
    no =  length - yes

    if  yes ==0 or no==0:
        return 0
    
    p_yes = yes/length
    p_no =no/length

    return -(p_yes * math.log2(p_yes) + p_no * math.log2(p_no))

def split_data(data, col, value):
    return[row for row in data if row[col]==value]

def information_gain(data, col):
    best_entropy =entropy(data)

    values = set(row[col] for row in data)

    weigted_entropy =0
    
    for v in values:
        subset =split_data(data, col, v)
        weight = len(subset)/len(data)
        weigted_entropy += weight *entropy(subset)

    return best_entropy - weigted_entropy

def best_feature(data):
    num_features = len(data[0]) - 1
    best_ig =-1
    best_col =-1
   

    for col in range(num_features):
        ig =information_gain(data, col)

        if ig > best_ig:
            best_col =col
            best_ig = ig
    return best_col


def build_tree(data, features):
    lables =[row[-1] for row in data]

    if lables.count(lables[0])==(len(lables)):
        return lables[0]
    
    if len(features) == 0:
        return max(set(lables), key=lables.count)
    
    best_col =best_feature(data)
    best_name= features[best_col]

    tree ={best_name:{}}

    values = set(row[best_col] for row in data)

    for v in values:
         subset = split_data(data, best_col, v)

        # Step 3: remove used feature
         new_features = features[:best_col] + features[best_col+1:]
         new_data = [row[:best_col] + row[best_col+1:] for row in subset]
        

        # Step 4: recursion
         tree[best_name][v] = build_tree(new_data, new_features)

    return tree


tree = build_tree(data, features)
print(tree)