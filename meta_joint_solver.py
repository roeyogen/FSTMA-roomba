import copy
import pickle
import time
import numpy as np
from environment import Env
from multi_agent_single_panel import MetaJointGraph, JointGraph , WAStart,meta_heuristic,single_heuristic
import multi_agent_single_panel
from offline_graph_path import UniformCostSearch, Graph, MetaGraph, Node
from pathlib import Path


class metaJointSolver:
    ACTIONS = ['STAY', 'UP', 'DOWN', 'RIGHT', 'LEFT']

    def __init__(self, num_of_solar_panels=4, height=1, width=1, number_of_agents=2, max_agent_fuel={},
                 fixed_starting=(0, 3),is_notebook = False):

        self.num_of_solar_panels = num_of_solar_panels
        self.height = height
        self.width = width
        self.length = (self.width + 1) * self.num_of_solar_panels + 1

        self.number_of_agents = number_of_agents

        self.max_agent_fuel = max_agent_fuel
        self.fixed_starting = fixed_starting

        folder = "pickles/joint_agent_per_panel_cost_action/"

        actions_file_path = folder + f"{height}_BY_{width}_actions_for_{str(max_agent_fuel).replace(': ', '_')}.pkl"
        costs_file_path = folder + f"{height}_BY_{width}_costs_for_{str(max_agent_fuel).replace(': ', '_')}.pkl"
        joint_actions_file_path = folder + f"{height}_BY_{width}_joint_actions_for_{str(max_agent_fuel).replace(': ', '_')}.pkl"
        joint_costs_file_path = folder + f"{height}_BY_{width}_joint_costs_for_{str(max_agent_fuel).replace(': ', '_')}.pkl"

        if is_notebook:
            actions_file_path = actions_file_path.replace("pickles","/content/drive/My Drive/Colab Notebooks/pickles")
            costs_file_path = costs_file_path.replace("pickles", "/content/drive/My Drive/Colab Notebooks/pickles")
            joint_actions_file_path = joint_actions_file_path.replace("pickles", "/content/drive/My Drive/Colab Notebooks/pickles")
            joint_costs_file_path = joint_costs_file_path.replace("pickles", "/content/drive/My Drive/Colab Notebooks/pickles")


        actions_file = Path(actions_file_path)
        costs_file = Path(costs_file_path)
        joint_actions_file = Path(joint_actions_file_path)
        joint_costs_file = Path(joint_costs_file_path)

        print('@' * 45)
        print('@' * 5 + " " * 5 + "Part 1 - Cleaning a panel" + " " * 5 + '@' * 5)
        print('@' * 45)

        if actions_file.is_file() and costs_file.is_file():

            print("actions_file and costs_file exist, reading from pickle")

            actions_file = open(actions_file_path, "rb")
            single_actions = pickle.load(actions_file)

            costs_file = open(costs_file_path, "rb")
            single_costs = pickle.load(costs_file)
        else:
            print("no prev calc found")

            single_costs, single_actions = self.calc_cost(actions_file_path, costs_file_path)

        if joint_actions_file.is_file() and joint_costs_file.is_file():

            print("joint_actions_file and joint_costs_file exist, reading from pickle")

            joint_actions_file = open(joint_actions_file_path, "rb")
            joint_single_actions = pickle.load(joint_actions_file)

            joint_costs_file = open(joint_costs_file_path, "rb")
            joint_single_costs = pickle.load(joint_costs_file)
        else:
            print("no prev calc found")

            joint_single_costs, joint_single_actions = self.calc_joint_cost(joint_actions_file_path,\
                                                                            joint_costs_file_path, single_costs)


        all_costs = copy.deepcopy(single_costs)
        for agent, a_costs in joint_single_costs.items():
            for action,value in a_costs.items():
                all_costs[agent][action] = value

        all_actions = copy.deepcopy(single_actions)
        for agent, a_costs in joint_single_actions.items():
            for action, value in a_costs.items():
                all_actions[agent][action] = value

        print()
        print('@' * 45)
        print('@' * 5 + " " + "Part 2 - Running joint meta solution" + " " + '@' * 5)
        print('@' * 45)

        self.meta_graph = MetaJointGraph(num_of_solar_panels=num_of_solar_panels,
                                    number_of_agents=number_of_agents, max_agent_fuel=max_agent_fuel,
                                    costs=all_costs, fixed_starting=fixed_starting)

        ucs = multi_agent_single_panel.UniformCostSearch()

        meta_solution_file_path = "pickles/joint_agent_per_panel_meta_solution/" \
                                  + f"{height}_BY_{width}_meta_solution_for_{str(max_agent_fuel).replace(': ','_')}.pkl"

        if is_notebook:
            meta_solution_file_path = meta_solution_file_path.replace("pickles","/content/drive/My Drive/Colab Notebooks/pickles")


        meta_solution_file = Path(meta_solution_file_path)

        is_file = False
        if meta_solution_file.is_file():
            print("meta_solution_file exists, reading from pickle\n")

            meta_solution_file = open(meta_solution_file_path, "rb")
            self.meta_solution = pickle.load(meta_solution_file)
            is_file = True


        else:
            print("no prev calc found")

            Astar = WAStart(meta_heuristic)
            self.meta_solution = Astar.solve(self.meta_graph)

            #self.meta_solution = ucs.solve(self.meta_graph)



        for state in self.meta_solution.path:
            print(state)
            time.sleep(0.5)
        # print(*self.meta_solution.path)



        meta_actions, meta_starts = self.get_multi_action_path(self.meta_solution.path)

        # use meta path for performing single agent path

        action_paths = self.get_action_path_per_agent(meta_starts, meta_actions, all_actions)

        actions_list = self.get_action_per_timestep(action_paths)

        if not is_file:
            path = self.meta_solution.path
            for i in range(len(path)):
                for next_state in path[i].next.values():
                    del next_state
                    # if next_state is not None:
                    #     if next_state != path[i + 1]:
                    #         del next_state
                    #     else:
                    #         for _state in next_state.next.values():
                    #             if _state not in path:
                    #                 del _state

            meta_solution_file = open(meta_solution_file_path, "wb")
            pickle.dump(self.meta_solution, meta_solution_file)
            meta_solution_file.close()

        print()
        print('@' * 45)
        print('@' * 5 + " " * 6 + "Part 3 - Final solution" + " " * 6 + '@' * 5)
        print('@' * 45)
        print("creating env and printing solution:\n")
        env = Env(num_of_solar_panels=self.num_of_solar_panels, height=self.height,
                  width=self.width, number_of_agents=self.number_of_agents, max_fuel=max_agent_fuel,
                  fixed_starting=list(meta_starts.values()))
        env.render()
        for a in actions_list:
            to_print = ["Agent_{} = {}".format(i + 1, a[i]) for i in range(len(a))]
            print("Actions: ")
            for act in to_print:
                print(act)
            print() # , , Agent_3={actions[2]}, , Agent_4={actions[3]}, , Agent_5={actions[4]}, , Agent_6={actions[5]}")
            env.step(a)
            env.render()
            if env.is_done():
                print("\nSuccess")
                print("-" * 10)
                print("Solution Cost =", self.meta_solution.cost)
                print("Number of Steps =", self.meta_solution.number_of_steps)
                print("Number of Nodes Expanded =", self.meta_solution.n_node_expanded)
                print("Run Time =", int(self.meta_solution.solve_time), "sec")
                break


    def calc_cost(self, actions_file_path, costs_file_path):

        costs = {}
        actions = {}

        for i in range(self.number_of_agents):
            agent_costs = {'STAY': 1, 'RIGHT_RIGHT': None, 'RIGHT_LEFT': None, 'LEFT_LEFT': None, 'LEFT_RIGHT': None}

            agent_actions = {'STAY': ['STAY'], 'RIGHT_RIGHT': None, 'RIGHT_LEFT': None, 'LEFT_LEFT': None,
                             'LEFT_RIGHT': None}

            right_graph = Graph(self.height, self.width, self.max_agent_fuel['Agent_{}'.format(i + 1)],
                                finishing_side='right')
            # right_ucs = UniformCostSearch()
            # right_solution = right_ucs.solve(right_graph)
            right_Astar = WAStart(single_heuristic)
            right_solution = right_Astar.solve(right_graph)

            print(right_solution.cost)
            print(right_solution.n_node_expanded)
            print(right_solution.solve_time)
            print(*right_solution.path)
            for state in right_solution.path:
                del state.next

            left_graph = Graph(self.height, self.width, self.max_agent_fuel['Agent_{}'.format(i + 1)],
                               finishing_side='left')
            # left_ucs = UniformCostSearch()
            # left_solution = left_ucs.solve(left_graph)
            left_Astar = WAStart(single_heuristic)
            left_solution = left_Astar.solve(left_graph)

            print(left_solution.cost)
            print(left_solution.n_node_expanded)
            print(left_solution.solve_time)
            print(*left_solution.path)
            for state in left_solution.path:
                del state.next

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

        actions_file = open(actions_file_path, "wb")
        pickle.dump(actions, actions_file)
        actions_file.close()

        costs_file = open(costs_file_path, "wb")
        pickle.dump(costs, costs_file)
        costs_file.close()

        return costs, actions

    def calc_joint_cost(self, actions_file_path, costs_file_path, single_costs):

        costs = {}
        actions = {}

        agent_costs = {'JRR': {}, 'JRL': {}, 'JLR': {}, 'JLL': {}}

        agent_actions = {'JRR': {}, 'JRL': {}, 'JLR': {}, 'JLL': {}}

        waiting = 0
        swapped = True
        max_waiting = single_costs['Agent_1']['RIGHT_LEFT']
        runner = 0
        fuel = {'Agent_1': self.max_agent_fuel['Agent_1'], 'Agent_2': self.max_agent_fuel['Agent_1']}

        while runner < max_waiting:

            waiting_dict = {'Agent_1': waiting, 'Agent_2': 0}
            swapped_graph = JointGraph(height=self.height, width=self.width, max_agent_fuel=fuel,
                               waiting=waiting_dict, swapped=swapped)

            # swapped_ucs = multi_agent_single_panel.UniformCostSearch()
            # swapped_solution = swapped_ucs.solve(swapped_graph)
            swapped_Astar = WAStart(single_heuristic)
            swapped_solution = swapped_Astar.solve(swapped_graph)

            print(swapped_solution.cost)
            print(swapped_solution.number_of_steps)
            print(swapped_solution.n_node_expanded)
            print(swapped_solution.solve_time)
            print(*swapped_solution.path)
            for state in swapped_solution.path:
                del state.next

            runner = max(swapped_solution.number_of_steps.values())
            if runner >= max_waiting:
                break
            agent_costs['JRR'][waiting] = swapped_solution.number_of_steps['Agent_1']
            agent_costs['JLL'][waiting] = swapped_solution.number_of_steps['Agent_2']

            agent_actions['JRR'][waiting] = self.get_joint_single_action_path(swapped_solution.path,'Agent_1')
            agent_actions['JLL'][waiting] = self.get_joint_single_action_path(swapped_solution.path,'Agent_2')

            waiting += 1



        waiting = 0
        runner = 0
        swapped = False
        max_waiting = single_costs['Agent_1']['RIGHT_LEFT']
        while runner < max_waiting:

            waiting_dict = {'Agent_1': waiting, 'Agent_2': 0}
            not_swapped_graph = JointGraph(height=self.height, width=self.width, max_agent_fuel=fuel,
                                       waiting=waiting_dict, swapped=swapped)

            # not_swapped_ucs = multi_agent_single_panel.UniformCostSearch()
            # not_swapped_solution = not_swapped_ucs.solve(not_swapped_graph)
            not_swapped_Astar = WAStart(single_heuristic)
            not_swapped_solution = not_swapped_Astar.solve(not_swapped_graph)

            print(not_swapped_solution.cost)
            print(not_swapped_solution.number_of_steps)
            print(not_swapped_solution.n_node_expanded)
            print(not_swapped_solution.solve_time)
            print(*not_swapped_solution.path)
            for state in not_swapped_solution.path:
                del state.next

            runner = max(not_swapped_solution.number_of_steps.values())
            if runner >= max_waiting:
                break

            agent_costs['JRL'][waiting] = not_swapped_solution.number_of_steps['Agent_1']
            agent_costs['JLR'][waiting] = not_swapped_solution.number_of_steps['Agent_2']

            agent_actions['JRL'][waiting] = self.get_joint_single_action_path(not_swapped_solution.path, 'Agent_1')
            agent_actions['JLR'][waiting] = self.get_joint_single_action_path(not_swapped_solution.path, 'Agent_2')

            waiting += 1


        for i in range(self.number_of_agents):
            costs['Agent_{}'.format(i + 1)] = agent_costs
            actions['Agent_{}'.format(i + 1)] = agent_actions

        actions_file = open(actions_file_path, "wb")
        pickle.dump(actions, actions_file)
        actions_file.close()

        costs_file = open(costs_file_path, "wb")
        pickle.dump(costs, costs_file)
        costs_file.close()

        return costs, actions

    @staticmethod
    def reversed_actions(actions):

        copy_actions = []

        for a in actions:
            if a == 'RIGHT':
                copy_actions.append('LEFT')
            elif a == 'LEFT':
                copy_actions.append('RIGHT')
            else:
                copy_actions.append(a)

        return copy_actions

    @staticmethod
    def get_single_action_path(path):

        actions = []

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

    @staticmethod
    def get_joint_single_action_path(path,agent):

        actions = []

        prev_loc = path[0].agents[agent][0]

        for p in path[1:]:

            new_loc = p.agents[agent][0]

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

    def get_multi_action_path(self, path):

        actions = []
        starting_points = {}
        for agent, [location, fuel] in path[0].agents.items():
            starting_points[agent] = location

        current_node = path[-1]
        current_steps = current_node.g_path
        while current_node.parent is not None:
            current_action = None
            for action, child in current_node.parent.next.items():
                if child == current_node:
                    current_action = action
                    break
            actions.append([current_action,current_node.parent.next_waiting])
            current_node = current_node.parent

        for agent,sp in starting_points.items():
            starting_points[agent]= self.meta_graph.charging_points.index(tuple(sp))

        return actions[::-1], starting_points

    @staticmethod
    def get_action_path_per_agent(meta_starts, meta_actions, all_paths):

        copied_all_paths = {}

        for agent, successors in all_paths.items():
            copied_all_paths[agent] = {}
            for action,path in successors.items():
                """
                if action in ['JRR','JRL']:
                    copied_all_paths[agent][action] = {}
                    for key,value in path.items():
                        new_path = copy.deepcopy(all_paths["Agent_1"][action][key])
                        copied_all_paths[agent][action][key] = new_path[key::]
                elif action in ['JLR','JLL']:
                    copied_all_paths[agent][action] = {}
                    for key,value in path.items():
                        new_path = copy.deepcopy(all_paths["Agent_2"][action][key])
                        copied_all_paths[agent][action][key] = new_path[key::]
                """
                if action in ['JRR','JRL','JLR','JLL']:
                    copied_all_paths[agent][action] = {}
                    for key,value in path.items():
                        new_path = copy.deepcopy(all_paths[agent][action][key])
                        copied_all_paths[agent][action][key] = new_path[key::]

                else:
                    copied_all_paths[agent][action] = copy.deepcopy(path)


        action_paths = {key: [] for key in meta_starts.keys()}

        for i,agent in enumerate(action_paths.keys()):
            for a in meta_actions:
                if a[0][i] in ['JRR', 'JRL', 'JLR', 'JLL']:
                    # if LR take agent 2
                    waiting = a[1]['Agent_{}'.format(i + 1)]
                    expanded_path = copied_all_paths[agent][a[0][i]][waiting]
                    action_paths[agent].extend(expanded_path)
                else:
                    expanded_path = copied_all_paths[agent][a[0][i]]
                    action_paths[agent].extend(expanded_path)

        return action_paths

    @staticmethod
    def get_action_per_timestep(action_paths):

        actions_list = []

        lengths = [len(actions) for actions in action_paths.values()]

        for i in range(max(lengths)):
            joint_action = []
            for agent_actions in action_paths.values():
                to_append = agent_actions[i] if i < len(agent_actions) else 'STAY'
                joint_action.append(to_append)

            actions_list.append(joint_action)

        return actions_list





if __name__ == '__main__':
    num_of_solar_panels = 5
    height = 2
    width = 2
    number_of_agents = 3
    max_agent_fuel = {'Agent_1': 20, 'Agent_2': 20,'Agent_3': 20}
    fixed_starting = (0,2,4)


    meta_joint_solver = metaJointSolver(num_of_solar_panels=num_of_solar_panels, height=height, width=width,
                             number_of_agents=number_of_agents,
                             max_agent_fuel=max_agent_fuel, fixed_starting=fixed_starting,is_notebook=True)
