from environment import Env
import random
import os
# import openpyxl
import datetime
import math
import time
import pickle

from meta_environment import MetaEnv
from meta_solver import metaSolver
from offline_graph_path import *
from meta_solver import *
from multi_agent_single_panel import *
from meta_joint_solver import *






def Part_1_Calculation():
    print("Part One Calculation")

    # # @@@@@@@@@@ 2. 3*3, Left to Right, Fuel = 20, Astar
    print("\n@@@@@@@@@@ 2. 3x3, Left to Right, Fuel = 20, Astar")


    graph = Graph(height=3, width=3, max_agent_fuel=20, finishing_side="right")

    Astar = WAStart(single_heuristic)
    solution = Astar.solve(graph)

    for state in solution.path:
        print(state)
        del state.next
        time.sleep(0.5)

    print("-" * 10)
    print("Solution Cost =", solution.cost)
    print("Number of Nodes Expanded =", solution.n_node_expanded)
    print("Run Time =", int(solution.solve_time), "sec")

    solution_file_path = "pickles\\presentation\\Single Agent Single Panel\\3x3, Left to Right, Fuel = 20, Astar.pkl"
    solution_file = open(solution_file_path, "wb")
    pickle.dump(solution, solution_file)
    solution_file.close()

    # @@@@@@@@@@ 3. 3*3, Left to Left, Fuel = 20, Astar
    print("\n@@@@@@@@@@ 3. 3x3, Left to Left, Fuel = 20, Astar")


    graph = Graph(height=3, width=3, max_agent_fuel=20, finishing_side="left")

    Astar = WAStart(single_heuristic)
    solution = Astar.solve(graph)

    for state in solution.path:
        print(state)
        del state.next
        time.sleep(0.5)

    print("-" * 10)
    print("Solution Cost =", solution.cost)
    print("Number of Nodes Expanded =", solution.n_node_expanded)
    print("Run Time =", int(solution.solve_time), "sec")

    solution_file_path = "pickles\\presentation\\Single Agent Single Panel\\3x3, Left to Left, Fuel = 20, Astar.pkl"
    solution_file = open(solution_file_path, "wb")
    pickle.dump(solution, solution_file)
    solution_file.close()

    # @@@@@@@@@@ 4. 3*3, Left to Left, Fuel = 6, Astar
    print("\n@@@@@@@@@@ 4. 3x3, Left to Left, Fuel = 6, Astar")


    graph = Graph(height=3, width=3, max_agent_fuel=6, finishing_side="left")

    Astar = WAStart(single_heuristic)
    solution = Astar.solve(graph)

    for state in solution.path:
        print(state)
        del state.next
        time.sleep(0.5)

    print("-" * 10)
    print("Solution Cost =", solution.cost)
    print("Number of Nodes Expanded =", solution.n_node_expanded)
    print("Run Time =", int(solution.solve_time), "sec")

    solution_file_path = "pickles\\presentation\\Single Agent Single Panel\\3x3, Left to Left, Fuel = 6, Astar.pkl"
    solution_file = open(solution_file_path, "wb")
    pickle.dump(solution, solution_file)
    solution_file.close()

    # @@@@@@@@@@ 5. 3*3, Left to Left, Fuel = 10, Astar
    print("\n@@@@@@@@@@ 5. 3x3, Left to Left, Fuel = 10, Astar")


    graph = Graph(height=3, width=3, max_agent_fuel=10, finishing_side="left")

    Astar = WAStart(single_heuristic)
    solution = Astar.solve(graph)

    for state in solution.path:
        print(state)
        del state.next
        time.sleep(0.5)

    print("-" * 10)
    print("Solution Cost =", solution.cost)
    print("Number of Nodes Expanded =", solution.n_node_expanded)
    print("Run Time =", int(solution.solve_time), "sec")

    solution_file_path = "pickles\\presentation\\Single Agent Single Panel\\3x3, Left to Left, Fuel = 10, Astar.pkl"
    solution_file = open(solution_file_path, "wb")
    pickle.dump(solution, solution_file)
    solution_file.close()

    # @@@@@@@@@@ 6. 5*10, Left to Right, Fuel = 60, Astar
    print("\n@@@@@@@@@@ 6. 5x10, Left to Right, Fuel = 60, Astar")


    graph = Graph(height=5, width=10, max_agent_fuel=60, finishing_side="right")


    Astar = WAStart(single_heuristic)
    solution = Astar.solve(graph)

    for state in solution.path:
        print(state)
        del state.next
        time.sleep(0.5)

    print("-" * 10)
    print("Solution Cost =", solution.cost)
    print("Number of Nodes Expanded =", solution.n_node_expanded)
    print("Run Time =", int(solution.solve_time), "sec")

    solution_file_path = "pickles\\presentation\\Single Agent Single Panel\\5x10, Left to Right, Fuel = 60, Astar.pkl"
    solution_file = open(solution_file_path, "wb")
    pickle.dump(solution, solution_file)
    solution_file.close()

    # @@@@@@@@@@ 7. Extra Dirty
    print("\n@@@@@@@@@@ 7. Extra Dirty")

    """
    in offline_graph_path
    in Graph innit 
    UNCOMMENT # adding dirty in certain spots 
    and after run re-COMMENT
    
    """

    graph = Graph(height=4, width=4, max_agent_fuel=60, finishing_side="right")

    #change as needed
    for i in range(1,4):
        for j in range(1,5):
            graph.head.board[i,j] = max(i+j,1)

    Astar = WAStart(single_heuristic)
    solution = Astar.solve(graph)

    for state in solution.path:
        print(state)
        del state.next
        time.sleep(0.5)

    print("-" * 10)
    print("Solution Cost =", solution.cost)
    print("Number of Nodes Expanded =", solution.n_node_expanded)
    print("Run Time =", int(solution.solve_time), "sec")

    solution_file_path = "pickles\\presentation\\Single Agent Single Panel\\4x4, Left to Right, Fuel = 60, Astar with Extra Dirty.pkl"
    solution_file = open(solution_file_path, "wb")
    pickle.dump(solution, solution_file)
    solution_file.close()




def Part_1_Testing():
    print("Part One Testing")

    # @@@@@@@@@@ 2. 3*3, Left to Right, Fuel = 20, Astar
    print("\n@@@@@@@@@@ 2. 3x3, Left to Right, Fuel = 20, Astar")

    solution_file_path = "pickles/presentation/Single Agent Single Panel/3x3, Left to Right, Fuel = 20, Astar.pkl"
    solution_file = open(solution_file_path, "rb")
    solution = pickle.load(solution_file)

    for state in solution.path:
        print(state)

    print("-" * 10)
    print("Solution Cost =", solution.cost)
    print("Number of Nodes Expanded =", solution.n_node_expanded)
    print("Run Time =", int(solution.solve_time), "sec")

    # @@@@@@@@@@ 3. 3*3, Left to Left, Fuel = 20, Astar
    print("\n@@@@@@@@@@ 3. 3x3, Left to Left, Fuel = 20, Astar")

    solution_file_path = "pickles/presentation/Single Agent Single Panel/3x3, Left to Left, Fuel = 20, Astar.pkl"
    solution_file = open(solution_file_path, "rb")
    solution = pickle.load(solution_file)

    for state in solution.path:
        print(state)

    print("-" * 10)
    print("Solution Cost =", solution.cost)
    print("Number of Nodes Expanded =", solution.n_node_expanded)
    print("Run Time =", int(solution.solve_time), "sec")

    # @@@@@@@@@@ 4. 3*3, Left to Left, Fuel = 6, Astar
    print("\n@@@@@@@@@@ 4. 3x3, Left to Left, Fuel = 6, Astar")


    solution_file_path = "pickles/presentation/Single Agent Single Panel/3x3, Left to Left, Fuel = 6, Astar.pkl"
    solution_file = open(solution_file_path, "rb")
    solution = pickle.load(solution_file)

    for state in solution.path:
        print(state)

    print("-" * 10)
    print("Solution Cost =", solution.cost)
    print("Number of Nodes Expanded =", solution.n_node_expanded)
    print("Run Time =", int(solution.solve_time), "sec")

    # @@@@@@@@@@ 5. 3*3, Left to Left, Fuel = 10, Astar
    print("\n@@@@@@@@@@ 5. 3x3, Left to Left, Fuel = 10, Astar")


    solution_file_path = "pickles/presentation/Single Agent Single Panel/3x3, Left to Left, Fuel = 10, Astar.pkl"
    solution_file = open(solution_file_path, "rb")
    solution = pickle.load(solution_file)

    for state in solution.path:
        print(state)

    print("-" * 10)
    print("Solution Cost =", solution.cost)
    print("Number of Nodes Expanded =", solution.n_node_expanded)
    print("Run Time =", int(solution.solve_time), "sec")

    # @@@@@@@@@@ 6. 5*10, Left to Right, Fuel = 60, Astar
    print("\n@@@@@@@@@@ 6. 5x10, Left to Right, Fuel = 60, Astar")

    solution_file_path = "pickles/presentation/Single Agent Single Panel/5x10, Left to Right, Fuel = 60, Astar.pkl"
    solution_file = open(solution_file_path, "rb")
    solution = pickle.load(solution_file)

    for state in solution.path:
        print(state)

    print("-" * 10)
    print("Solution Cost =", solution.cost)
    print("Number of Nodes Expanded =", solution.n_node_expanded)
    print("Run Time =", int(solution.solve_time), "sec")

    # @@@@@@@@@@ 7. Extra Dirty
    print("\n@@@@@@@@@@ 7. Extra Dirty")

    solution_file_path = "pickles/presentation/Single Agent Single Panel/4x4, Left to Right, Fuel = 60, Astar with Extra Dirty.pkl"
    solution_file = open(solution_file_path, "rb")
    solution = pickle.load(solution_file)

    for state in solution.path:
        print(state)

    print("-" * 10)
    print("Solution Cost =", solution.cost)
    print("Number of Nodes Expanded =", solution.n_node_expanded)
    print("Run Time =", int(solution.solve_time), "sec")





def Part_2_Calculation():
    print("Part Two Calculation")

    # @@@@@@@@@@ 1. First, a simple problem of a single row

    # print("\n@@@@@@@@@@ 1. First, a simple problem of a single row")
    #
    # costs = {'Agent_1': {'STAY': 1, 'RIGHT_RIGHT': 5, 'RIGHT_LEFT': 6, 'LEFT_LEFT': 5, 'LEFT_RIGHT': 6},
    #          'Agent_2': {'STAY': 1, 'RIGHT_RIGHT': 5, 'RIGHT_LEFT': 6, 'LEFT_LEFT': 5, 'LEFT_RIGHT': 6}}
    #
    # max_agent_fuel = {"Agent_1": 20, "Agent_2": 20}
    #
    # graph = MetaGraph(num_of_solar_panels=4, height=1, width=1, number_of_agents=2, max_agent_fuel=max_agent_fuel,
    #                   costs=costs, fixed_starting=(0, 3))
    #
    # Astar = offline_graph_path.WAStart(offline_graph_path.meta_heuristic)
    # solution = Astar.solve(graph)
    #
    #
    # for state in solution.path:
    #     print(state)
    #     del state.next
    #     time.sleep(0.5)
    #
    # print("-" * 10)
    # print("Solution Cost =", solution.cost)
    # print("Number of Nodes Expanded =", solution.n_node_expanded)
    # print("Run Time =", int(solution.solve_time), "sec")
    #
    # solution_file_path = "pickles\\presentation\\Multi Agent Single Panel\\a simple problem of a single row.pkl"
    # solution_file = open(solution_file_path, "wb")
    # pickle.dump(solution, solution_file)
    # solution_file.close()
    #
    # # @@@@@@@@@@ 2. Bigger board with full solution
    #
    # print("\n@@@@@@@@@@ 2. Bigger board with full solution")
    #
    # num_of_solar_panels = 3
    # height = 5
    # width = 3
    # number_of_agents = 2
    # max_agent_fuel = {'Agent_1': 20, 'Agent_2': 20}
    # fixed_starting = (0, 3)
    #
    # meta_solver = metaSolver(num_of_solar_panels=num_of_solar_panels, height=height, width=width,
    #                          number_of_agents=number_of_agents,
    #                          max_agent_fuel=max_agent_fuel, fixed_starting=fixed_starting)

    # @@@@@@@@@@ 3. Even More Agents!

    print("\n@@@@@@@@@@ 3. Even More Agents!")

    num_of_solar_panels = 5
    height = 2
    width = 2
    number_of_agents = 3
    max_agent_fuel = {'Agent_1': 20, 'Agent_2': 20, 'Agent_3': 20}
    fixed_starting = (0,2,4)

    meta_solver = metaSolver(num_of_solar_panels=num_of_solar_panels, height=height, width=width,
                             number_of_agents=number_of_agents,
                             max_agent_fuel=max_agent_fuel, fixed_starting=fixed_starting)



    # # @@@@@@@@@@ 4. Multiple agents on a joint panel (2 agents 1 board)
    #
    # print("\n@@@@@@@@@@ 4. Multiple agents on a joint panel (2 agents 1 board)")
    #
    # print("not swapping positions")
    #
    # max_agent_fuel = {"Agent_1": 30, "Agent_2": 30}
    # waiting = {"Agent_1": 1, "Agent_2": 0}
    # swapped = False
    # graph = JointGraph(height=5, width=3, max_agent_fuel=max_agent_fuel, waiting=waiting, swapped=swapped)
    #
    # Astar = multi_agent_single_panel.WAStart(multi_agent_single_panel.single_heuristic)
    # solution = Astar.solve(graph)
    #
    # # print(*solution.path)
    #
    # for state in solution.path:
    #     print(state)
    #     del state.next
    #     time.sleep(0.5)
    #
    # print("-" * 10)
    # print("Solution Cost =", solution.cost)
    # print("Number of Steps =", solution.number_of_steps)
    # print("Number of Nodes Expanded =", solution.n_node_expanded)
    # print("Run Time =", int(solution.solve_time), "sec")
    #
    # print("swapping positions")
    #
    # max_agent_fuel = {"Agent_1": 30, "Agent_2": 30}
    # waiting = {"Agent_1": 1, "Agent_2": 0}
    # swapped = True
    # graph = JointGraph(height=5, width=3, max_agent_fuel=max_agent_fuel, waiting=waiting, swapped=swapped)
    #
    # Astar = multi_agent_single_panel.WAStart(multi_agent_single_panel.single_heuristic)
    # solution = Astar.solve(graph)
    #
    # #print(*solution.path)
    #
    # for state in solution.path:
    #     print(state)
    #     time.sleep(0.5)
    #
    # print("-"*10)
    # print("Solution Cost =",solution.cost)
    # print("Number of Steps =",solution.number_of_steps)
    # print("Number of Nodes Expanded =",solution.n_node_expanded)
    # print("Run Time =",int(solution.solve_time),"sec")
    #
    #
    #
    # # @@@@@@@@@@ 5. Multiple agents on joint panels (2 agents 3 boards)
    #
    # print("\n@@@@@@@@@@ 5. Multiple agents on joint panels (2 agents 3 boards)")
    #
    # num_of_solar_panels = 3
    # height = 5
    # width = 3
    # number_of_agents = 2
    # max_agent_fuel = {'Agent_1': 20, 'Agent_2': 20}
    # fixed_starting = (0, 3)
    #
    # meta_joint_solver = metaJointSolver(num_of_solar_panels=num_of_solar_panels, height=height, width=width,
    #                                     number_of_agents=number_of_agents,
    #                                     max_agent_fuel=max_agent_fuel, fixed_starting=fixed_starting)
    #
    # # @@@@@@@@@@ 6. Multiple agents on joint panels (3 agents 3 boards)
    #
    # print("\n@@@@@@@@@@ 6. Multiple agents on joint panels (3 agents 5 boards)")
    #
    # num_of_solar_panels = 5
    # height = 2
    # width = 2
    # number_of_agents = 3
    # max_agent_fuel = {'Agent_1': 20, 'Agent_2': 20, 'Agent_3': 20}
    # fixed_starting = (0, 2, 5)
    #
    # meta_joint_solver = metaJointSolver(num_of_solar_panels=num_of_solar_panels, height=height, width=width,
    #                                     number_of_agents=number_of_agents,
    #                                     max_agent_fuel=max_agent_fuel, fixed_starting=fixed_starting)




def Part_2_Testing():
    print("Part Two Testing")

    # @@@@@@@@@@ 1. First, a simple problem of a single row

    print("\n@@@@@@@@@@ 1. First, a simple problem of a single row")

    costs = {'Agent_1': {'STAY': 1, 'RIGHT_RIGHT': 5, 'RIGHT_LEFT': 6, 'LEFT_LEFT': 5, 'LEFT_RIGHT': 6},
             'Agent_2': {'STAY': 1, 'RIGHT_RIGHT': 5, 'RIGHT_LEFT': 6, 'LEFT_LEFT': 5, 'LEFT_RIGHT': 6}}

    max_agent_fuel = {"Agent_1": 20, "Agent_2": 20}

    graph = MetaGraph(num_of_solar_panels=4, height=1, width=1, number_of_agents=2, max_agent_fuel=max_agent_fuel,
                      costs=costs, fixed_starting=(0, 3))


    solution_file_path = "pickles/presentation/Multi Agent Single Panel/a simple problem of a single row.pkl"
    solution_file = open(solution_file_path, "rb")
    solution = pickle.load(solution_file)

    for state in solution.path:
        print(state)
        time.sleep(0.5)

    print("-" * 10)
    print("Solution Cost =", solution.cost)
    print("Number of Nodes Expanded =", solution.n_node_expanded)
    print("Run Time =", int(solution.solve_time), "sec")

    # @@@@@@@@@@ 2. Bigger board with full solution

    print("\n@@@@@@@@@@ 2. Bigger board with full solution")

    num_of_solar_panels = 3
    height = 4
    width = 4
    number_of_agents = 2
    max_agent_fuel = {'Agent_1': 20, 'Agent_2': 20}
    fixed_starting = (0, 3)


    meta_solver = metaSolver(num_of_solar_panels=num_of_solar_panels, height=height, width=width,
                             number_of_agents=number_of_agents,
                             max_agent_fuel=max_agent_fuel, fixed_starting=fixed_starting)
    #
    #
    #
    # @@@@@@@@@@ 3. Even More Agents!

    print("\n@@@@@@@@@@ 3. Even More Agents!")

    num_of_solar_panels = 5
    height = 2
    width = 2
    number_of_agents = 3
    max_agent_fuel = {'Agent_1': 20, 'Agent_2': 20, 'Agent_3': 20}
    fixed_starting = (0,2,4)

    meta_solver = metaSolver(num_of_solar_panels=num_of_solar_panels, height=height, width=width,
                             number_of_agents=number_of_agents,
                             max_agent_fuel=max_agent_fuel, fixed_starting=fixed_starting)



    # @@@@@@@@@@ 4. Multiple agents on a joint panel (2 agents 1 board)

    print("\n@@@@@@@@@@ 4. Multiple agents on a joint panel (2 agents 1 board)")

    print("not swapping positions")

    max_agent_fuel = {"Agent_1": 30, "Agent_2": 30}
    waiting = {"Agent_1": 1, "Agent_2": 0}
    swapped = False
    graph = JointGraph(height=3, width=3, max_agent_fuel=max_agent_fuel, waiting=waiting, swapped=swapped)

    # Astar = multi_agent_single_panel.WAStart(multi_agent_single_panel.single_heuristic)
    # solution = Astar.solve(graph)
    #
    # for state in solution.path:
    #     print(state)
    #     del state.next
    #     time.sleep(0.5)

    solution_file_path = "pickles/presentation/Multi Agent Single Panel/Multiple agents on a joint panel _not swapping.pkl"
    # solution_file = open(solution_file_path, "wb")
    # pickle.dump(solution, solution_file)
    # solution_file.close()


    solution_file = open(solution_file_path, "rb")
    solution = pickle.load(solution_file)


    # print(*solution.path)

    for state in solution.path:
        print(state)
        time.sleep(0.5)

    print("-" * 10)
    print("Solution Cost =", solution.cost)
    print("Number of Steps =", solution.number_of_steps)
    print("Number of Nodes Expanded =", solution.n_node_expanded)
    print("Run Time =", int(solution.solve_time), "sec")

    print("swapping positions")

    max_agent_fuel = {"Agent_1": 30, "Agent_2": 30}
    waiting = {"Agent_1": 1, "Agent_2": 0}
    swapped = True
    graph = JointGraph(height=3, width=3, max_agent_fuel=max_agent_fuel, waiting=waiting, swapped=swapped)

    # Astar = multi_agent_single_panel.WAStart(multi_agent_single_panel.single_heuristic)
    # solution = Astar.solve(graph)
    #
    # for state in solution.path:
    #     print(state)
    #     del state.next
    #     time.sleep(0.5)

    solution_file_path = "pickles/presentation/Multi Agent Single Panel/Multiple agents on a joint panel _swapping.pkl"
    # solution_file = open(solution_file_path, "wb")
    # pickle.dump(solution, solution_file)
    # solution_file.close()

    solution_file = open(solution_file_path, "rb")
    solution = pickle.load(solution_file)

    #print(*solution.path)

    for state in solution.path:
        print(state)
        time.sleep(0.5)

    print("-"*10)
    print("Solution Cost =",solution.cost)
    print("Number of Steps =",solution.number_of_steps)
    print("Number of Nodes Expanded =",solution.n_node_expanded)
    print("Run Time =",int(solution.solve_time),"sec")



    # @@@@@@@@@@ 5. Multiple agents on joint panels (2 agents 3 boards)

    print("\n@@@@@@@@@@ 5. Multiple agents on joint panels (2 agents 3 boards)")

    num_of_solar_panels = 3
    height = 3
    width = 3
    number_of_agents = 2
    max_agent_fuel = {'Agent_1': 20, 'Agent_2': 20}
    fixed_starting = (0, 3)

    meta_joint_solver = metaJointSolver(num_of_solar_panels=num_of_solar_panels, height=height, width=width,
                                        number_of_agents=number_of_agents,
                                        max_agent_fuel=max_agent_fuel, fixed_starting=fixed_starting)

    # @@@@@@@@@@ 6. Multiple agents on joint panels (3 agents 3 boards)

    print("\n@@@@@@@@@@ 6. Multiple agents on joint panels (3 agents 5 boards)")

    num_of_solar_panels = 5
    height = 2
    width = 2
    number_of_agents = 3
    max_agent_fuel = {'Agent_1': 20, 'Agent_2': 20, 'Agent_3': 20}
    fixed_starting = (0, 2, 4)

    meta_joint_solver = metaJointSolver(num_of_solar_panels=num_of_solar_panels, height=height, width=width,
                                        number_of_agents=number_of_agents,
                                        max_agent_fuel=max_agent_fuel, fixed_starting=fixed_starting)



if __name__ == '__main__':


    # Part_1_Calculation()
    #
    #
    Part_1_Testing()

    # Part_2_Calculation()

    Part_2_Testing()















