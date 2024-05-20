from collections import Counter, defaultdict
from constraint import Problem, AllDifferentConstraint
import time

class BugByte:
    def __init__(self, edges):
        """
        Initializes the graph with the provided edges.

        Args:
            edges (list): A list of tuples where each tuple represents an edge in the format (incoming_node, outgoing_node, weight).
        """
        self.edges = edges
        self.edgeKeys = {}
        self.weightKeys = {}
        self.vars = []
        self.visited = [[], []]
        self.problem = None
        self.solutions = None
        self.graphPermutations = []
        self.min_path = None

        for c, (incoming, outgoing, weight) in enumerate(edges):
            self.visited[0].extend([incoming, outgoing])
            var_name = f'x{c}'
            self.vars.append(var_name)
            self.edgeKeys[(incoming, outgoing)] = var_name
            self.weightKeys[var_name] = weight

        # Get unique nodes and their connected edges
        nodes = Counter(self.visited[0]).keys()
        self.nodes = {node: [] for node in nodes}

        for (incoming, outgoing), var_name in self.edgeKeys.items():
            self.nodes[incoming].append(var_name)
            self.nodes[outgoing].append(var_name)

    def sumRuleConstraint(self, show_constraints=False):
        """
        Applies sum rule constraints to the graph's variables based on the weights and nodes.

        The method generates constraints for the problem where the sum of the edge variables
        should be equal to the node values. It also ensures all variables have different values.
        """
        vars = self.vars
        weightKeys = self.weightKeys
        nodes = self.nodes

        problem = Problem()
        availableNumbers = list(range(1, 25))

        preWeights = [weight for weight in weightKeys.values() if weight != -1]
        preEdges = [edge for edge, weight in weightKeys.items() if weight != -1]

        variables = weightKeys.keys()
        problem.addVariables(variables, availableNumbers)

        def generateConstraints(problem, vars, target):
            """
            Generates constraints for the given variables and target value.

            Args:
                problem (Problem): The constraint problem instance.
                vars (list): A list of variable names.
                target (int or str): The target value for the constraint.
            """
            if any(char in str(target) for char in '()fea'):  # Escape for nodes without rules, and for nodes subject to the path rule.
                return None  # These do not need sum rule constraints

            vars_str = ', '.join(vars)
            sum_str = ' + '.join(vars)
            constraint_code = f"problem.addConstraint(lambda {vars_str}: {sum_str} == {target}, {vars})"
            
            if show_constraints:
                print(constraint_code)
            exec(constraint_code)

        if show_constraints:
            print('Constraints:')

        # Generate constraints for pre-weighted edges
        for edge, weight in zip(preEdges, preWeights):
            generateConstraints(problem, [edge], weight)

        # Generate constraints for nodes
        for node, edgeVars in nodes.items():
            generateConstraints(problem, edgeVars, node)

        problem.addConstraint(AllDifferentConstraint())
        self.solutions = problem.getSolutions()

    def reconstructGraph(self):
        """
        Reconstructs the graph for each solution permutation and stores the resulting edges.

        The method iterates through each solution in `self.solutions`, reconstructs the edges 
        with the corresponding weights, and appends the reconstructed edge lists to 
        `self.graphPermutations`.
        """
        def construct_edges(weightKeys):
            """
            Constructs a list of edges with weights from the provided weight keys.

            Args:
                weightKeys (dict): A dictionary mapping variable names to their corresponding weights.

            Returns:
                list: A list of tuples representing the edges with their weights in the format (node1, node2, weight).
            """
            edges = []
            for edge, var in self.edgeKeys.items():
                weight = weightKeys[var]
                edges.append((edge[0], edge[1], weight))
            return edges

        # Get the edge list for each graph permutation.
        for solution in self.solutions:
            edges = construct_edges(solution)
            self.graphPermutations.append(edges)
        
    def check_edge_sums(self, edges):
        """
        Checks if the sum rule is satisfied for each applicable node.

        Args:
            edges (list): A list of edges with weights.

        Returns:
            bool: True if all conditions are met, False otherwise.
        """
        # Create a dictionary to store the adjacency list
        adjacency_list = defaultdict(list)
        for edge in edges:
            node, next_node, weight = edge
            if '(' in node:
                target_sums = [int(num) for num in node.strip('()').split(',')]  # Extract the integers from the tuple string
                for target_sum in target_sums:
                    node_name = f"{node}_{target_sum}"
                    adjacency_list[node_name].append((next_node, weight))
            elif '(' in next_node:
                target_sums = [int(num) for num in next_node.strip('()').split(',')]  # Extract the integers from the tuple string
                for target_sum in target_sums:
                    next_node_name = f"{next_node}_{target_sum}"
                    adjacency_list[next_node_name].append((node, weight))
            else:
                adjacency_list[node].append((next_node, weight))
                adjacency_list[next_node].append((node, weight))

        # Ensure all nodes are in the adjacency list
        for edge in edges:
            node, next_node, _ = edge
            if node not in adjacency_list:
                adjacency_list[node] = []
            if next_node not in adjacency_list:
                adjacency_list[next_node] = []

        # Perform a DFS from each path rule node
        def dfs(node, target_sum, path_sum=0, visited=None):
            if visited is None:
                visited = set()
            visited.add(node)
            if path_sum > target_sum:
                return False
            if path_sum == target_sum:
                return True
            for next_node, weight in adjacency_list[node]:
                if next_node not in visited:
                    if dfs(next_node, target_sum, path_sum + weight, visited.copy()):
                        return True
            return False

        # For each node, check all paths
        all_conditions_met = True
        for node in adjacency_list:
            if '_' in node:
                _, target_sum = node.split('_')
                target_sum = int(target_sum)
                if not dfs(node, target_sum):
                    all_conditions_met = False
                    break  # Early exit if any condition is not met

        return all_conditions_met

    def checkPathRule(self):
        """
        Checks each graph permutation to see if it satisfies the edge sum constraints.

        The method iterates through each graph permutation stored in `self.graphPermutations`,
        checks if the permutation satisfies the edge sum constraints using the `check_edge_sums` method,
        and if a valid permutation is found, sets it as the solution and prints 'Solved'.
        """
        for permutation in self.graphPermutations:
            if self.check_edge_sums(permutation):
                self.solution = permutation
                break  # Stop after finding the first valid solution
        if self.solution == None:
            print('No solution found.')


    def compute_min_weight_path(self, start_node='start', end_node='end'):
        """
        Computes the minimum weight path between the start and end nodes.

        This method uses Depth-First Search (DFS) to find the minimum weight path from the
        start node to the end node. It updates the `min_path` attribute with the sequence of
        weights for the minimum path.

        Args:
            start_node (str): The starting node of the path. Default is 'start'.
            end_node (str): The ending node of the path. Default is 'end'.
        """
        # Create a dictionary to store the adjacency list
        adjacency_list = defaultdict(list)
        edges = self.solution
        for edge in edges:
            node, next_node, weight = edge
            adjacency_list[node].append((next_node, weight))
            adjacency_list[next_node].append((node, weight))  # Add edge in the opposite direction

        min_path = None
        min_weight = float('inf')
        min_path_edges = []

        def dfs(node, path, weight, weights, edges):
            nonlocal min_path, min_weight, min_path_edges
            if node == end_node:
                if weight < min_weight:
                    min_path = weights
                    min_weight = weight
                    min_path_edges = edges
            else:
                for neighbor, edge_weight in adjacency_list[node]:
                    if neighbor not in path:
                        dfs(neighbor, path + [node], weight + edge_weight, weights + [edge_weight], edges + [(node, neighbor, edge_weight)])

        dfs(start_node, [], 0, [], [])
        self.min_path = min_path

        # Print the formatted path
        if min_path_edges:
            path_str = ' -> '.join([f"{node}--({weight})-->{next_node}" for node, next_node, weight in min_path_edges])
            print(f"Minimum weight path: {path_str}")

    def decode_to_word(self):
        """
        Decodes the minimum weight path to a word by converting each weight to its corresponding letter.

        Returns:
            str: The decoded word.
        """
        return ''.join(chr(n + 64) for n in self.min_path)


# Edge list from bugByte graph
edges = [
    ('(19, 23)', '17', -1),
    ('(19, 23)', '54', -1),
    ('(19, 23)', 'start', 12),
    ('(6, 9, 16)', '60', -1),
    ('17', '3', -1),
    ('25', '29', -1),
    ('25', '75', -1),
    ('25', 'end', -1),
    ('29', '39', -1),
    ('29', '79', -1),
    ('3', 'start', -1),
    ('(31)', '54', -1),
    ('39', '79', 7),
    ('39', 'end', -1),
    ('49', '60', -1),
    ('49', '75', 20),
    ('49', '(8)', -1),
    ('49', 'start', -1),
    ('54', '60', -1),
    ('54', '79', -1),
    ('60', '75', -1),
    ('60', '79', 24),
    ('75', 'foot2', -1),
    ('79', 'foot1', -1)
]

def main():
    start_time = time.time()

    bugByteSolver = BugByte(edges)
    bugByteSolver.sumRuleConstraint(show_constraints=False)
    bugByteSolver.reconstructGraph()
    bugByteSolver.checkPathRule()
    bugByteSolver.compute_min_weight_path()
    print(bugByteSolver.decode_to_word())

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

if __name__ == '__main__':
    main()