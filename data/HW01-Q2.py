class BinTreeNode:
    def __init__(self, value=0, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def level_order_traversal(root):
    # create queue of nodes to check
    queue = [root]
    
    # will be list of lists, each nested list is a level of the tree
    levels = []
    
    # while queue isn't empty...
    while queue:
        current_level_size = len(queue) # number of nodes in current level)
        current_level = [] # list to store the nodes in current level
        
        # for each node in the current level (for loop so we can add children to queue)
        for _ in range(current_level_size):
            node = queue.pop(0)  # pop from front of queue
            current_level.append(node.value) # add node to current level list
            
            # add children to back of queue
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        # add current level to levels list
        levels.append(current_level)
    
    # print levels in reverse order
    for level in reversed(levels):
        print(" ".join(map(str, level))) # must convert ints to strings

# testing
# root
root = BinTreeNode(1)

# second level
root.left = BinTreeNode(2)
root.right = BinTreeNode(3)

# third level
root.left.left = BinTreeNode(4)
root.left.right = BinTreeNode(5)
root.right.left = BinTreeNode(6)
root.right.right = BinTreeNode(7)

# fourth level
root.left.left.left = BinTreeNode(8)
root.left.left.right = BinTreeNode(9)
root.left.right.left = BinTreeNode(10)

# fifth level
root.left.right.left.left = BinTreeNode(11)
root.left.left.right.right = BinTreeNode(12)

# call algorithm
level_order_traversal(root)
