from environment import Env
import random
import os
import openpyxl
import datetime
import math
import time

from meta_environment import MetaEnv
from offline_graph_path import Graph, UniformCostSearch

ACTIONS = {'STAY', 'RIGHT', 'LEFT'}


def metaMain():
    graph = Graph(height=3, width=2, max_agent_fuel=45, finishing_side="left")
    ucs = UniformCostSearch()
    solution = ucs.solve(graph)

    # print(*solution.path)

    for state in solution.path:
        print(state)
        time.sleep(0.5)

    print(solution.cost)
    print(solution.n_node_expanded)
    print(solution.solve_time)

    metaEnv = MetaEnv(num_of_solar_panels=4, height=1, width=1, number_of_agents=2, max_fuel=100, fixed_starting=[0, 3])
    metaEnv.render()

    for i in range(10):
        actions = []
        for _ in range(metaEnv.number_of_agents):
            actions.append(random.sample([*metaEnv.ACTIONS.keys()], 1)[0])
        print(f"Actions: Agent_1={actions[0]}, Agent_2={actions[1]}")  # , , Agent_3={actions[2]}, , Agent_4={actions[3]}, , Agent_5={actions[4]}, , Agent_6={actions[5]}")
        metaEnv.step(actions)
        metaEnv.render()
        if metaEnv.is_done():
            print("success")
            break




if __name__ == '__main__':

    # env = Env(num_of_solar_panels=4, height=5, width=3, number_of_agents=2, max_fuel=100, fixed_starting = [0,3])
    # env.render()
    # for i in range(10):
    #     actions = []
    #     for _ in range(env.number_of_agents):
    #         actions.append(random.sample([*env.ACTIONS.keys()], 1)[0])
    #     print(f"Actions: Agent_1={actions[0]}, Agent_2={actions[1]}")  # , , Agent_3={actions[2]}, , Agent_4={actions[3]}, , Agent_5={actions[4]}, , Agent_6={actions[5]}")
    #     env.step(actions)
    #     env.render()
    #     if env.is_done():
    #         print("success")
    #         break


    #metaMain()

    # env = Env(num_of_solar_panels=3, height=3, width=3, number_of_agents=2, max_fuel=10, fixed_starting=None)
    # env.render()

    graph = Graph(height=3, width=3, max_agent_fuel=20, finishing_side="right")

    ucs = UniformCostSearch()
    solution = ucs.solve(graph)

    for state in solution.path:
        print(state)
        time.sleep(0.5)

    print("-" * 10)
    print("Solution Cost =", solution.cost)
    print("Number of Nodes Expanded =", solution.n_node_expanded)
    print("Run Time =", int(solution.solve_time), "sec")










