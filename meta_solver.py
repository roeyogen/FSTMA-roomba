import copy
import time

import numpy as np

from offline_graph_path import UniformCostSearch, Graph, MetaGraph, Node


class metaSolver:

    ACTIONS = ['STAY', 'UP', 'DOWN', 'RIGHT', 'LEFT']


    def __init__(self,num_of_solar_panels=4, height=1, width=1,number_of_agents=2, max_agent_fuel={},
                 fixed_starting=(0,3)):

        self.num_of_solar_panels = num_of_solar_panels
        self.height = height
        self.width = width
        self.length = (self.width + 1) * self.num_of_solar_panels + 1

        self.number_of_agents = number_of_agents

        self.max_agent_fuel = max_agent_fuel
        self.fixed_starting = fixed_starting


        single_costs, single_actions = self.calc_cost()

        self.meta_graph = MetaGraph(num_of_solar_panels=num_of_solar_panels,height=height, width=width,
                                    number_of_agents=number_of_agents, max_agent_fuel=max_agent_fuel,
                                    costs=single_costs, fixed_starting=fixed_starting)

        ucs = UniformCostSearch()
        self.meta_solution = ucs.solve(self.meta_graph)

        print(*self.meta_solution)
        print("X")

        meta_actions,meta_starts = self.get_multi_action_path(self.meta_solution.path)

        action_paths = {key: [] for key in meta_starts.keys()}






        # use meta path for performing single agent path





    def calc_cost(self):

        costs = {}
        actions = {}

        for i in range(self.number_of_agents):

            agent_costs = {'STAY': 1, 'RIGHT_RIGHT': None, 'RIGHT_LEFT': None, 'LEFT_LEFT': None, 'LEFT_RIGHT': None}

            agent_actions = {'STAY': None, 'RIGHT_RIGHT': None, 'RIGHT_LEFT': None, 'LEFT_LEFT': None, 'LEFT_RIGHT': None}

            right_graph = Graph(self.height, self.width, self.max_agent_fuel['Agent_{}'.format(i + 1)],
                                finishing_side='right')
            right_ucs = UniformCostSearch()
            right_solution = right_ucs.solve(right_graph)
            print(right_solution.cost)
            print(right_solution.n_node_expanded)
            print(right_solution.solve_time)
            print(*right_solution.path)

            # for state in right_solution.path:
            #     print(state)
            #     time.sleep(1)

            left_graph = Graph(self.height, self.width, self.max_agent_fuel['Agent_{}'.format(i + 1)],
                                finishing_side='left')
            left_ucs = UniformCostSearch()
            left_solution = left_ucs.solve(left_graph)
            print(left_solution.cost)
            print(left_solution.n_node_expanded)
            print(left_solution.solve_time)
            print(*left_solution.path)

            # for state in left_solution.path:
            #     print(state)
            #     time.sleep(1)

            agent_costs['RIGHT_RIGHT'] = right_solution.cost
            agent_costs['LEFT_LEFT'] = right_solution.cost

            agent_costs['RIGHT_LEFT'] = left_solution.cost
            agent_costs['LEFT_RIGHT'] = left_solution.cost

            agent_actions['RIGHT_RIGHT'] = self.get_single_action_path(right_solution.path)
            agent_actions['LEFT_LEFT'] = self.reversed_actions(agent_actions['RIGHT_RIGHT'])

            agent_actions['RIGHT_LEFT'] = self.get_single_action_path(left_solution.path)
            agent_actions['LEFT_RIGHT'] = self.reversed_actions(agent_actions['RIGHT_LEFT'])

            costs['Agent_{}'.format(i + 1)] = agent_costs
            actions['Agent_{}'.format(i + 1)] = agent_actions

        return costs, actions


    def reversed_actions(self,actions):

        copy_actions = []

        for a in actions:
            if a == 'RIGHT':
                copy_actions.append('LEFT')
            elif a == 'LEFT':
                copy_actions.append('RIGHT')
            # elif a == 'RIGHT_RIGHT':
            #     copy_actions.append('LEFT_LEFT')
            # elif a == 'LEFT_LEFT':
            #     copy_actions.append('RIGHT_RIGHT')
            # elif a == 'RIGHT_LEFT':
            #     copy_actions.append('LEFT_RIGHT')
            # elif a == 'LEFT_RIGHT':
            #     copy_actions.append('RIGHT_LEFT')
            else:
                copy_actions.append(a)

        return copy_actions

    def get_single_action_path(self,path):

        actions =[]

        prev_loc = path[0].agent_loc

        for p in path[1:]:

            new_loc = p.agent_loc

            diff_x = new_loc[0] - prev_loc[0]
            diff_y = new_loc[1] - prev_loc[1]

            if diff_x == 0 and diff_y == 0:
                actions.append('STAY')
            elif diff_x == -1 and diff_y == 0:
                actions.append('UP')
            elif diff_x == 1 and diff_y == 0:
                actions.append('DOWN')
            elif diff_x == 0 and diff_y == 1:
                actions.append('RIGHT')
            elif diff_x == 0 and diff_y == -1:
                actions.append('LEFT')

            prev_loc = new_loc

        return actions

    def get_multi_action_path(self,path):

        actions = {}
        starting_points = {}

        prev_loc = {}

        for agent, [location, fuel] in path[0].agents.items():
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

                if diff_y == 0 and np.array_equal(prev_board, new_board):
                    actions[agent].append('STAY')
                elif diff_y == 2:
                    actions[agent].append('RIGHT_RIGHT')
                elif diff_y == -2:
                    actions[agent].append('LEFT_LEFT')
                elif diff_y == 0:
                    changed_RIGHT_LEFT = new_loc[agent][1] + 1
                    changed_LEFT_RIGHT = new_loc[agent][1] - 1
                    x_loc = new_loc[agent][0]

                    if abs(new_board[(x_loc, changed_RIGHT_LEFT)] - prev_board[(x_loc, changed_RIGHT_LEFT)]) == 1 \
                            and changed_RIGHT_LEFT < new_board.shape[1]:
                        actions[agent].append('RIGHT_LEFT')
                    if abs(new_board[(x_loc, changed_LEFT_RIGHT)] - prev_board[(x_loc, changed_LEFT_RIGHT)]) == 1 \
                            and changed_LEFT_RIGHT >= 0:
                        actions[agent].append('LEFT_RIGHT')

            prev_loc = new_loc
            prev_board = new_board

        return actions, starting_points






if __name__ == '__main__':

    meta_solver = metaSolver(num_of_solar_panels=4, height=2, width=2,number_of_agents=2,
                             max_agent_fuel={'Agent_1':20,'Agent_2':20}, fixed_starting=(0,3))

