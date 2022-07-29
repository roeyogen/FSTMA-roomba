import heapdict as heapdict
from heapdict import heapdict
import numpy as np
from termcolor import colored
import copy
from time import process_time as curr_time


class Node:
    OUT = -100
    CHARGE = 100

    def __init__(self, board, agent_loc, parent=None,g_value=0, agent_fuel=20, finishing_side="right"):

        self.board = board
        self.height = board.shape[0]
        self.width = board.shape[1]
        self.agent_loc = agent_loc

        self.parent = parent
        self.g_value = g_value
        self.agent_fuel = agent_fuel
        self.finishing_side = finishing_side

        self.next = {'STAY': None, 'UP': None, 'DOWN': None, 'RIGHT': None, 'LEFT': None}

    def __repr__(self):

        res = ""
        for r in range(self.height):
            res += "|"
            for c in range(self.width):
                val = self.board[r, c]
                if val == self.OUT:
                    res += " " + colored("=======".ljust(7), 'grey') + " |"  # format
                elif [r, c] == self.agent_loc:
                    val = ' Agent '
                    res += " " + colored(str(val).ljust(7), 'blue') + " |"  # format
                elif val == self.CHARGE:
                    res += " " + colored(("CHARGE"+"⚡️"[0]).ljust(7), 'green') + " |"  # format
                else:
                    if val == 0:
                        res += " " + colored("".ljust(7), 'white') + " |"  # format
                    else:
                        res += " " + ("   " + str(val) + "   ").ljust(7) + " |"  # format
            res += "\n"
        return res +"\nagent_fuel = "+str(self.agent_fuel)

    def is_goal(self):

        if self.finishing_side == "right":
            if self.agent_loc != [self.height // 2, -1]:
                return False

        if self.finishing_side == "left":
            if self.agent_loc != [self.height // 2, 0]:
                return False

        # check if all board is zeros (or charge / out)
        to_compare = np.full((self.height, self.width), 0, dtype=int)
        to_compare[:, 0] = self.OUT
        to_compare[self.height // 2, 0] = self.CHARGE
        to_compare[:, -1] = self.OUT
        to_compare[self.height // 2, -1] = self.CHARGE

        if not np.array_equal(to_compare, self.board):
            return False

        return True

    def is_in_board(self, agent_loc):

        if 0 <= agent_loc[0] < self.height and 0 <= agent_loc[1] < self.width and \
                not self.board[tuple(agent_loc)] == self.OUT:
            return True
        return False

    def get_path(self):
        current_node = self
        path = [current_node]
        while current_node.parent is not None:
            path = [current_node.parent] + path  # adding parent to beginning of the path
            current_node = current_node.parent
        return path


class NodesPriorityQueue:
    def __init__(self):
        self.nodes_queue = heapdict()
        self.state_to_node = dict()

    def add(self, node, priority):
        #assert (node.board,node.agent_loc,node.agent_fuel) not in self.state_to_node
        self.nodes_queue[node] = priority
        self.state_to_node[node] = node

    def pop(self):
        if len(self.nodes_queue) > 0:
            node, priority = self.nodes_queue.popitem()
            del self.state_to_node[node]
            return node
        else:
            return None

    def __contains__(self, state):
        return state in self.state_to_node

    def get_node(self, state):
        assert state in self.state_to_node
        return self.state_to_node[state]

    def remove_node(self, node):
        assert node in self.nodes_queue
        del self.nodes_queue[node]
        #assert (node.board,node.agent_loc,node.agent_fuel) in self.state_to_node
        del self.state_to_node[node]

    def __len__(self):
        return len(self.nodes_queue)

class NodesCollection:
    def __init__(self):
        self._collection = dict()

    def add(self, node: Node):
        assert isinstance(node, Node)
        #assert (node.board,node.agent_loc,node.agent_fuel) not in self._collection
        self._collection[node] = node

    def remove_node(self, node):
        #assert (node.board,node.agent_loc,node.agent_fuel) in self._collection
        del self._collection[node]

    def __contains__(self, node):
        return node in self._collection

    def get_node(self, board,agent_loc,agent_fuel):
        #assert (board,agent_loc,agent_fuel) in self._collection
        return self._collection[node]

class Graph:

    ACTIONS = {'STAY': (0, 0), 'UP': (-1, 0), 'DOWN': (1, 0), 'RIGHT': (0, 1), 'LEFT': (0, -1)}
    OUT = -100
    CHARGE = 100


    def __init__(self, height, width):

        self.height = height
        self.width = width + 2

        # create full board
        board = np.full((self.height, self.width), 1, dtype=int)

        # mark floor and charge
        board[:, 0] = self.OUT
        board[self.height // 2, 0] = self.CHARGE

        board[:, -1] = self.OUT
        board[self.height // 2, -1] = self.CHARGE

        # agent starts at left of board
        agent_loc = [self.height // 2, 0]

        self.head = Node(board,agent_loc)


    def successor(self,node):

        # cannot move - out of fuel
        if node.agent_fuel == 0:
            return None

        for action,index in self.ACTIONS.items():

            new_pos = [node.agent_loc[0] + index[0],node.agent_loc[1] + index[1]]

            if node.is_in_board(new_pos):

                new_board = copy.deepcopy(node.board)

                # reduce value from where I stood -  if not charging point
                if new_board[tuple(node.agent_loc)] != self.CHARGE:
                    new_board[tuple(node.agent_loc)] = max(0,new_board[tuple(node.agent_loc)]-1)

                new_node = Node(new_board,new_pos,node,node.g_value+1,node.agent_fuel-1)

                node.next[action] = new_node

        return node.next




class GraphSearchSolution:
    def __init__(self, final_node: Node, solve_time: float, n_node_expanded: int, init_heuristic_time=None,
                 no_solution_reason=None):
        if final_node is not None:
            self.cost = final_node.g_value
            self.path = final_node.get_path()
        else:
            assert no_solution_reason is not None
            self.no_solution_reason = no_solution_reason
            self.cost = float("inf")
            self.path = None
        self.solve_time = solve_time
        self.n_node_expanded = n_node_expanded
        self.init_heuristic_time = init_heuristic_time


class BestFirstSearch():
    def __init__(self):
        super(BestFirstSearch, self).__init__()
        self.open = None
        self.close = None
        self.name = "abstract best first search"

    def _calc_node_priority(self, node):
        
        # calc how clean board is

        return node.g_value

    def solve(self, graph, time_limit=float("inf"), compute_all_dists=False):
        start_time = curr_time()

        self.open = NodesPriorityQueue()
        self.close = NodesCollection()

        """
        if hasattr(self, "_init_heuristic"):  # some heuristics need to be initialized with the maze problem
            init_heuristic_start_time = curr_time()
            self._init_heuristic(graph)
            init_heuristic_time = curr_time() - init_heuristic_start_time
        else:
            init_heuristic_time = None
        """

        initial_node = graph.head
        initial_node_priority = self._calc_node_priority(initial_node)
        self.open.add(initial_node, initial_node_priority)

        n_node_expanded = 0  # count the number of nodes expanded during the algorithm run.

        while True:
            if curr_time() - start_time >= time_limit:
                no_solution_found = True
                no_solution_reason = "time limit exceeded"
                break

            next_node = self.open.pop()
            if next_node is None:
                no_solution_found = True
                no_solution_reason = "no solution exists"
                break

            self.close.add(next_node)
            if next_node.is_goal():
                if not compute_all_dists:  # we will use this later, don't change
                    return GraphSearchSolution(next_node, solve_time=curr_time() - start_time,
                                               n_node_expanded=n_node_expanded, init_heuristic_time=None)
            ############################################################################################################

            n_node_expanded += 1
            print(n_node_expanded)
            for s in graph.successor(next_node).values():
                if s is None:
                    continue
                successor_node = s
                successor_node_priority = self._calc_node_priority(successor_node)
                if s not in self.open and s not in self.close:
                    self.open.add(successor_node, successor_node_priority)
                elif s in self.open:
                    node_in_open = self.open.get_node(s)
                    if successor_node.g_value < node_in_open.g_value:
                        self.open.remove_node(node_in_open)
                        self.open.add(successor_node, successor_node_priority)
                else:  # s is in close
                    node_in_close = self.close.get_node(s)
                    if successor_node.g_value < node_in_close.g_value:
                        self.close.remove_node(node_in_close)
                        self.open.add(successor_node, successor_node_priority)
            ############################################################################################################

        if compute_all_dists:
            return self.close
        else:
            assert no_solution_found
            return GraphSearchSolution(final_node=None, solve_time=curr_time() - start_time,
                                       n_node_expanded=n_node_expanded, no_solution_reason=no_solution_reason,
                                       init_heuristic_time=None)


class UniformCostSearch(BestFirstSearch):
    def __init__(self):
        super(UniformCostSearch, self).__init__()
        self.name = "uniform cost search"

    def _calc_node_priority(self, node):
        return node.g_value



if __name__ == '__main__':


    # graph = Graph(height = 5,width = 3)
    #
    # for action,next_node in graph.successor(graph.head).items():
    #     print(action)
    #     print(next_node)
    #     print()
    #
    #     if next_node is None:
    #         continue
    #
    #     for son_action, son_node in graph.successor(next_node).items():
    #
    #         print(son_action)
    #         print(son_node)
    #         print()

    graph = Graph(height=3, width=3)
    ucs = UniformCostSearch()

    ucs.solve(graph)




