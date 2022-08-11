from heapdict import heapdict
import numpy as np
from termcolor import colored
import copy
from time import process_time as curr_time
import time
import random
import itertools

from environment import Env


class Node:
    OUT = -100
    CHARGE = 100

    def __init__(self, board, agent_loc, parent=None, g_value=0, agent_fuel=20, finishing_side="right"):

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
                    res += " " + colored(("CHARGE").ljust(7), 'green') + " |"  # format
                else:
                    if val == 0:
                        res += " " + colored("".ljust(7), 'white') + " |"  # format
                    else:
                        res += " " + ("   " + str(val) + "   ").ljust(7) + " |"  # format
            res += "\n"
        return res +"\nagent_fuel = "+str(self.agent_fuel)

    def is_goal(self):

        if self.finishing_side == "right":
            if self.agent_loc != [self.height // 2, self.width-1]:
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


class MetaNode:
    OUT = -100
    CHARGE = 100
    ACTIONS = {'STAY': None, 'RIGHT_RIGHT': None, 'RIGHT_LEFT': None, 'LEFT_LEFT': None, 'LEFT_RIGHT': None}

    def __init__(self, board, agents, parent=None, g_value=0, g_path=None):

        self.board = board
        self.height = board.shape[0]
        self.width = 1
        self.length = board.shape[1]

        self.agents = agents
        self.number_of_agents = len(self.agents)

        self.parent = parent
        self.g_value = g_value
        self.g_path = g_path if g_path is not None else {'Agent_{}'.format(i + 1):0 for i in range(self.number_of_agents)}
        self.charging_points = [(0, column) for column in range(0, self.length, 2)]

        action_list = list(itertools.product(self.ACTIONS.keys(), repeat=self.number_of_agents))
        self.next = dict.fromkeys(action_list)

    def __repr__(self):

        res = ""
        for r in range(self.height):
            res += "|"
            for c in range(self.length):
                val = self.board[r, c]
                if val == self.OUT:
                    res += " " + colored("=======".ljust(7), 'grey') + " |"  # format
                elif [r, c] in [agent[0] for agent in self.agents.values()]:
                    for i in range(self.number_of_agents):
                        if [r, c] == self.agents['Agent_{}'.format(i + 1)][0]:
                            val = 'Agent_{}'.format(i + 1)
                    res += " " + colored(str(val).ljust(7), 'blue') + " |"  # format
                elif val == self.CHARGE:
                    res += " " + colored(("CHARGE").ljust(7), 'green') + " |"  # format
                else:
                    if val == 0:
                        res += " " + colored("".ljust(7), 'white') + " |"  # format
                    else:
                        res += " " + ("   " + str(val) + "   ").ljust(7) + " |"  # format
            res += "\n"

        res += "agent \t\tlocation \t\tfuel\n"
        for agent, value in self.agents.items():
            res += agent + "\t\t" + str(value[0]) + "\t\t\t" + str(value[1]) +"\n"
        return res

    def is_goal(self):

        # check if all board is zeros (or charge / out)
        to_compare = np.full((self.height, self.length), 0, dtype=int)
        for column in range(0, self.length, self.width + 1):
            to_compare[:, column] = self.OUT
            to_compare[self.height // 2, column] = self.CHARGE

        if not np.array_equal(to_compare, self.board):
            return False

        # check if all robots are in charging
        for i in range(self.number_of_agents):
            if tuple(self.agents['Agent_{}'.format(i + 1)][0]) not in self.charging_points:
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
        #assert isinstance(node, Node)
        #assert (node.board,node.agent_loc,node.agent_fuel) not in self._collection
        self._collection[node] = node

    def remove_node(self, node):
        #assert (node.board,node.agent_loc,node.agent_fuel) in self._collection
        del self._collection[node]

    def __contains__(self, node):
        return node in self._collection

    def get_node(self, node):
        #assert (board,agent_loc,agent_fuel) in self._collection
        return self._collection[node]


class Graph:
    ACTIONS = {'STAY': (0, 0), 'UP': (-1, 0), 'DOWN': (1, 0), 'RIGHT': (0, 1), 'LEFT': (0, -1)}
    OUT = -100
    CHARGE = 100

    def __init__(self, height, width, max_agent_fuel=20 , finishing_side="right"):

        self.height = height
        self.width = width + 2
        self.max_agent_fuel = max_agent_fuel
        self.finishing_side = finishing_side

        # create full board
        board = np.full((self.height, self.width), 1, dtype=int)

        # mark floor and charge
        board[:, 0] = self.OUT
        board[self.height // 2, 0] = self.CHARGE

        board[:, -1] = self.OUT
        board[self.height // 2, -1] = self.CHARGE

        # agent starts at left of board
        agent_loc = [self.height // 2, 0]

        self.head = Node(board=board, agent_loc=agent_loc, agent_fuel=max_agent_fuel, finishing_side = finishing_side)

    def successor(self, node):

        # cannot move - out of fuel
        if node.agent_fuel == 0:
            return None

        for action, index in self.ACTIONS.items():

            new_pos = [node.agent_loc[0] + index[0], node.agent_loc[1] + index[1]]

            if node.is_in_board(new_pos):

                new_board = copy.deepcopy(node.board)

                # reduce value from where I stood -  if not charging point
                if new_board[tuple(node.agent_loc)] != self.CHARGE:
                    new_board[tuple(node.agent_loc)] = max(0, new_board[tuple(node.agent_loc)]-1)

                if new_board[tuple(new_pos)] == self.CHARGE:
                    new_fuel = self.max_agent_fuel
                else:
                    new_fuel = node.agent_fuel-1

                new_node = Node(new_board, new_pos, node, node.g_value+1, new_fuel, self.finishing_side)

                node.next[action] = new_node

        return node.next


class MetaGraph:
    ACTIONS = {'STAY': (0, 0), 'RIGHT_RIGHT': (0, 2), 'RIGHT_LEFT': (0, 0), 'LEFT_LEFT': (0, -2), 'LEFT_RIGHT': (0, 0)}
    OUT = -100
    CHARGE = 100

    def __init__(self,num_of_solar_panels, height, width,number_of_agents=2, max_agent_fuel={},costs={},fixed_starting=None):

        self.num_of_solar_panels =num_of_solar_panels
        self.height = height
        self.width = width
        self.length = (self.width + 1) * self.num_of_solar_panels + 1

        self.number_of_agents = number_of_agents

        self.max_agent_fuel = max_agent_fuel
        self.costs = costs
        self.agents = {}

        self.charging_points = [(self.height // 2, column) for column in range(0, self.length, self.width + 1)]

        # create full board
        board = np.full((self.height, self.length), 1, dtype=int)
        for column in range(0, self.length, self.width + 1):
            board[:, column] = self.OUT
            board[self.height // 2, column] = self.CHARGE

        # agent starts at left of board
        #agent_loc = [self.height // 2, 0]

        # Initialize the agents positions
        # Random positions
        if fixed_starting is None:
            starting_points = random.sample([*[list(x) for x in self.charging_points]], self.number_of_agents)
            for i in range(self.number_of_agents):
                self.agents['Agent_{}'.format(i + 1)] = [starting_points[i], self.max_agent_fuel['Agent_{}'.format(i + 1)]]

        # Fixed positions
        else:
            for i in range(self.number_of_agents):
                self.agents['Agent_{}'.format(i + 1)] = [list(self.charging_points[fixed_starting[i]]), self.max_agent_fuel['Agent_{}'.format(i + 1)]]

        self.head = MetaNode(board=board, agents=self.agents)


    def is_in_board(self,board, agent_loc):

        if 0 <= agent_loc[0] < self.height and 0 <= agent_loc[1] < self.length and \
                not board[agent_loc] == self.OUT:
            return True
        return False

    def is_legal_step(self, board, new_agents, action, node):
        locs = [tuple(x[0]) for x in new_agents.values()]
        locs2 = [tuple(x[0]) for x in node.agents.values()]
        # check no overlap
        if not len(set(locs)) == len(locs):
            return False

        for i, loc in enumerate(locs):
            # check in in_board
            if not self.is_in_board(board, loc):
                return False
            if action[i] == "RIGHT_LEFT" and loc == (0, self.length-1):
                return False
            if action[i] == "LEFT_RIGHT" and loc == (0, 0):
                return False

        for i, loc in enumerate(locs2):
            if action[i] == "RIGHT_LEFT":
                for i2, loc2 in enumerate(locs2):
                    if action[i2] == "LEFT_RIGHT" and loc2[1] == loc[1] + 2:
                        return False
            if action[i] == "LEFT_RIGHT":
                for i2, loc2 in enumerate(locs2):
                    if action[i2] == "RIGHT_LEFT" and loc2[1] == loc[1] - 2:
                        return False

        return True

    def successor(self, node):

        # cannot move - out of fuel
        #if node.agent_fuel == 0:
        #   return None

        for action, index in node.next.items():

            new_agents = {}

            for i in range(self.number_of_agents):

                x = node.agents['Agent_{}'.format(i + 1)][0][0]
                y = node.agents['Agent_{}'.format(i + 1)][0][1]

                x += self.ACTIONS[action[i]][0]
                y += self.ACTIONS[action[i]][1]
                new_pos = [x, y]
                new_fuel = node.agents['Agent_{}'.format(i + 1)][1]

                new_agents['Agent_{}'.format(i + 1)] = [new_pos, new_fuel]


            if self.is_legal_step(node.board,new_agents,action, node):

                new_board = copy.deepcopy(node.board)

                #total_cost = 0

                new_g_path = copy.deepcopy(node.g_path)

                # mark "clean" panel
                for i in range(self.number_of_agents):

                    # reduce value for transition
                    mid_pos = copy.deepcopy(new_agents['Agent_{}'.format(i + 1)][0])
                    if action[i] == "RIGHT_RIGHT" or action[i] == "LEFT_RIGHT":
                        mid_pos[1] -= 1
                    elif action[i] == "RIGHT_LEFT" or action[i] == "LEFT_LEFT":
                        mid_pos[1] += 1
                    else:
                        #total_cost += self.costs['Agent_{}'.format(i + 1)][action[i]]
                        new_g_path['Agent_{}'.format(i + 1)] += self.costs['Agent_{}'.format(i + 1)][action[i]]
                        continue

                    mid_pos = tuple(mid_pos)
                    new_board[mid_pos] = max(0,new_board[mid_pos]-1)
                    #total_cost += self.costs['Agent_{}'.format(i + 1)][action[i]]
                    new_g_path['Agent_{}'.format(i + 1)] += self.costs['Agent_{}'.format(i + 1)][action[i]]

                #if new_board[tuple(new_pos)] == self.CHARGE:
                #    new_fuel = self.max_agent_fuel

                max_cost = max(new_g_path.values())
                new_node = MetaNode(new_board, new_agents, node, max_cost,new_g_path)

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


class BestFirstSearch:
    def __init__(self):
        super(BestFirstSearch, self).__init__()
        self.open = None
        self.close = None
        self.name = "abstract best first search"

    def _calc_node_priority(self, node):
        # calc how clean board is
        return node.g_value + np.sum(node.board[:, 1:-1])

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
            # if n_node_expanded % 100 == 0:
            #     print(n_node_expanded)

            if graph.successor(next_node) is None:
                continue
            for s in graph.successor(next_node).values():
                if s is None:
                    continue
                successor_node = s
                #print(s)
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
        return node.g_value + np.sum(node.board[:, 1:-1:2])

def get_multi_action_path(path):

    actions = {}
    starting_points = {}

    prev_loc = {}

    for agent,[location,fuel] in path[0].agents.items():

        prev_loc[agent] = location
        actions[agent] = []

    starting_points = copy.deepcopy(prev_loc)

    prev_board = path[0].board

    for p in path[1:]:

        new_loc = {}

        new_board = p.board

        for agent, [location, fuel] in p.agents.items():

            new_loc[agent] = location

            diff_y = new_loc[agent][1] - prev_loc[agent][1]


            if diff_y == 0 and np.array_equal(prev_board,new_board):
                actions[agent].append('STAY')
            elif diff_y == 2:
                actions[agent].append('RIGHT_RIGHT')
            elif diff_y == -2:
                actions[agent].append('LEFT_LEFT')
            elif diff_y == 0:
                changed_RIGHT_LEFT = new_loc[agent][1]+1
                changed_LEFT_RIGHT = new_loc[agent][1]-1
                x_loc = new_loc[agent][0]

                if abs(new_board[(x_loc,changed_RIGHT_LEFT)] - prev_board[(x_loc,changed_RIGHT_LEFT)]) == 1 \
                        and changed_RIGHT_LEFT < new_board.shape[1]:
                    actions[agent].append('RIGHT_LEFT')
                if abs(new_board[(x_loc,changed_LEFT_RIGHT)] - prev_board[(x_loc,changed_LEFT_RIGHT)]) == 1 \
                        and changed_LEFT_RIGHT >=0:
                    actions[agent].append('LEFT_RIGHT')


        prev_loc = new_loc
        prev_board = new_board

    return actions,starting_points

if __name__ == '__main__':

    #costs = {'STAY': 1, 'RIGHT_RIGHT': 5, 'RIGHT_LEFT': 6, 'LEFT_LEFT': 5, 'LEFT_RIGHT': 6}
    #costs = {'STAY': 1, 'RIGHT_RIGHT': 1, 'RIGHT_LEFT': 1, 'LEFT_LEFT': 1, 'LEFT_RIGHT': 1}

    costs = {'Agent_1': {'STAY': 1, 'RIGHT_RIGHT': 5, 'RIGHT_LEFT': 6, 'LEFT_LEFT': 5, 'LEFT_RIGHT': 6},
                    'Agent_2': {'STAY': 1, 'RIGHT_RIGHT': 5, 'RIGHT_LEFT': 6, 'LEFT_LEFT': 5, 'LEFT_RIGHT': 6}}

    max_agent_fuel = {"Agent_1": 20, "Agent_2": 20}

    graph = MetaGraph(num_of_solar_panels=4, height=1, width=1,number_of_agents=2, max_agent_fuel=max_agent_fuel,
                      costs=costs,fixed_starting=(0,3))

    ucs = UniformCostSearch()
    solution = ucs.solve(graph)

    #print(*solution.path)

    for state in solution.path:
        print(state)
        time.sleep(0.5)

    print(solution.cost)
    print(solution.n_node_expanded)
    print(solution.solve_time)

    # meta_actions,meta_starts = get_multi_action_path(solution.path)
    #
    #
    # single_actions =  {'Agent_1': {'STAY': None,
    #                              'RIGHT_RIGHT': ['RIGHT', 'UP', 'RIGHT', 'DOWN', 'RIGHT'],
    #                              'RIGHT_LEFT': ['RIGHT', 'UP', 'RIGHT', 'DOWN', 'LEFT', 'LEFT'],
    #                              'LEFT_LEFT': ['LEFT', 'UP', 'LEFT', 'DOWN', 'LEFT'],
    #                              'LEFT_RIGHT': ['LEFT', 'UP', 'LEFT', 'DOWN', 'RIGHT', 'RIGHT']},
    #                  'Agent_2': {'STAY': None,
    #                              'RIGHT_RIGHT': ['RIGHT', 'UP', 'RIGHT', 'DOWN', 'RIGHT'],
    #                              'RIGHT_LEFT': ['RIGHT', 'UP', 'RIGHT', 'DOWN', 'LEFT', 'LEFT'],
    #                              'LEFT_LEFT': ['LEFT', 'UP', 'LEFT', 'DOWN', 'LEFT'],
    #                              'LEFT_RIGHT': ['LEFT', 'UP', 'LEFT', 'DOWN', 'RIGHT', 'RIGHT']}}
    # single_costs = {'Agent_1': {'STAY': 1, 'RIGHT_RIGHT': 5, 'RIGHT_LEFT': 6, 'LEFT_LEFT': 5, 'LEFT_RIGHT': 6},
    #                 'Agent_2': {'STAY': 1, 'RIGHT_RIGHT': 5, 'RIGHT_LEFT': 6, 'LEFT_LEFT': 5, 'LEFT_RIGHT': 6}}
    #
    # action_paths = {key: [] for key in meta_starts.keys()}
    #
    # for agent,path in meta_actions.items():
    #
    #     for p in path:
    #         action_paths[agent].append(single_actions[agent][p])
    #
    # for agent, path in action_paths.items():
    #
    #     action_paths[agent] = sum(path,[])
    #
    # actions_list = []
    #
    # lengths = [len(actions) for actions in action_paths.values()]
    #
    # for i in range(max(lengths)):
    #     joint_action = []
    #     for agent_actions in action_paths.values():
    #         to_append = agent_actions[i] if i < len(agent_actions) else 'STAY'
    #         joint_action.append(to_append)
    #
    #     actions_list.append(joint_action)
    #
    # env = Env(num_of_solar_panels=4, height=2, width=2, number_of_agents=2, max_fuel=20, fixed_starting=[0, 3])
    # env.render()
    # for a in actions_list:
    #     print(f"Actions: Agent_1={a[0]}, Agent_2={a[1]}")  # , , Agent_3={actions[2]}, , Agent_4={actions[3]}, , Agent_5={actions[4]}, , Agent_6={actions[5]}")
    #     env.step(a)
    #     env.render()
    #     if env.is_done():
    #         print("success")
    #         break
    #
    # print(meta_starts)









