import search_209143486_325549681 as search 
import random
import math
import copy
import itertools
from collections import deque

ids = ["209143486", "325549681"]


class State():
    """
    A class which represents a state in the game.
    It holds all the dynamic data of the problem, 
    i.e. ships locations, current collected treasures, etc.
    """
    def __init__(self, pirate_ships, collected_treasures, marine_ships, marine_ships_next_location_index,
                 counter) -> None:
        self.pirate_ships = pirate_ships
        self.collected_treasures = collected_treasures
        self.marine_ships = marine_ships
        self.marine_ships_next_location_index = marine_ships_next_location_index

        self.counter = counter
        self.index = counter[0]# state index

    def copy_state(self):
        """
        @return a copy of the current state
        """
        return State(copy.deepcopy(self.pirate_ships),
                     copy.deepcopy(self.collected_treasures),
                     copy.deepcopy(self.marine_ships),
                     copy.deepcopy(self.marine_ships_next_location_index),
                     self.counter)

    def hash_func(self):
        return str(self.pirate_ships) + ", " + str(sorted(list(self.collected_treasures))) \
               + ", " + str(self.marine_ships_next_location_index)

    def equals(self, state):
        """
        compares between current state and given state
        @return True if same state, else False
        """
        for pirate in self.pirate_ships.keys():
            for key in self.pirate_ships[pirate].keys():
                if state.pirate_ships[pirate][key] != self.pirate_ships[pirate][key]:
                    return False

        for t in self.collected_treasures:
            if t not in state.collected_treasures:
                return False
        for t in state.collected_treasures:
            if t not in self.collected_treasures:
                return False

        for i, index in enumerate(self.marine_ships_next_location_index):
            if index != state.marine_ships_next_location_index[i]:
                return False

        return True

    def __lt__(self, state):
        """
        Tie breaker
        """
        return self.index < state.index


class OnePieceProblem(search.Problem):
    """This class implements a medical problem according to problem description file"""

    def __init__(self, initial):
        """Don't forget to implement the goal test
        You should change the initial to your own representation.
        search.Problem.__init__(self, initial) creates the root node"""
        search.Problem.__init__(self, initial)

        self.counter = [0] # counter used to index the nodes

        self.map = initial['map']
        self.treasures = initial['treasures']
        self.marine_ships_path = initial['marine_ships']

        self.pirate_base = initial['pirate_ships']['pirate_ship_1']

        self.index_to_tresure = {value: key for key, value in self.treasures.items()}

        # s_to_treasure is a dictionary, which each treasure is the key, and the value contains the real distance from every sea tile including base
        # s_to_base is a dictionary, which holds the distance from each sea tile to base
        s_to_treasures, s_to_base = self.calculate_distances(self.treasures, self.pirate_base)

        s_to_base[self.pirate_base] = 0
        self.s_to_treasures = s_to_treasures
        self.s_to_base = s_to_base

        pirate_ships = {}
        for pirate_ship, location in initial['pirate_ships'].items():
            pirate_ships[pirate_ship] = {'location': location,
                                         'hand_1': None,
                                         'hand_2': None}
            
        collected_treasures = set() # a set of treasures that are already deposited

        self.initial_State = State(pirate_ships, collected_treasures, self.marine_ships_path,
                                   [(0, 'r') for i in range(len(initial['marine_ships']))], self.counter)

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""

        actions = {pirate_ship: [] for pirate_ship in state.pirate_ships.keys()}
        for pirate_ship in state.pirate_ships.keys():
            # Wait
            actions[pirate_ship].append(("wait", pirate_ship))

            location = state.pirate_ships[pirate_ship]['location']
            if self.map[location[0]][location[1]] == "B":
                # Deposit treasure
                if state.pirate_ships[pirate_ship]["hand_1"] is not None \
                        or state.pirate_ships[pirate_ship]["hand_2"] is not None:
                    actions[pirate_ship].append(("deposit_treasures", pirate_ship))

            for orientation in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                new_location = (location[0] + orientation[0], location[1] + orientation[1])
                if not self.out_of_bounds(new_location):
                    if self.map[new_location[0]][new_location[1]] == "I":
                        if new_location in self.index_to_tresure.keys():
                            # Collect treasure
                            if state.pirate_ships[pirate_ship]["hand_1"] is None \
                                    or state.pirate_ships[pirate_ship]["hand_2"] is None:
                                for key, loc in self.treasures.items():
                                    if loc == new_location:
                                        actions[pirate_ship].append(("collect_treasure", pirate_ship, key))
                    else:
                        # Sail
                        actions[pirate_ship].append(("sail", pirate_ship, new_location))

        actions_list = [value for value in actions.values()]

        return list(itertools.product(*actions_list))

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        new_state = state.copy_state()
        for act in action:
            if "sail" in act:
                new_state.pirate_ships[act[1]]['location'] = act[2]

            elif "collect_treasure" in act:
                if new_state.pirate_ships[act[1]]['hand_1'] is None:
                    new_state.pirate_ships[act[1]]['hand_1'] = act[2]
                else:
                    if new_state.pirate_ships[act[1]]['hand_1'] < act[2]:
                        new_state.pirate_ships[act[1]]['hand_2'] = act[2]
                    else:
                        new_state.pirate_ships[act[1]]['hand_2'] = new_state.pirate_ships[act[1]]['hand_1']
                        new_state.pirate_ships[act[1]]['hand_1'] = act[2]


            elif "deposit_treasures" in act:
                new_state.collected_treasures.add(new_state.pirate_ships[act[1]]['hand_1'])
                new_state.pirate_ships[act[1]]['hand_1'] = None
                if new_state.pirate_ships[act[1]]['hand_2'] is not None:
                    new_state.collected_treasures.add(new_state.pirate_ships[act[1]]['hand_2'])
                    new_state.pirate_ships[act[1]]['hand_2'] = None

        # Marines movement:
        for i, (marine_path, curr_marine_state) in enumerate(
                zip(new_state.marine_ships.values(), new_state.marine_ships_next_location_index)):
            next_step = self.next_marine_step(marine_path, curr_marine_state[0], curr_marine_state[1])

            marine_location = marine_path[next_step[0]]
            for value in new_state.pirate_ships.values():
                if value["location"] == marine_location:
                    value["hand_1"] = None
                    value["hand_2"] = None

            new_state.marine_ships_next_location_index[i] = next_step

        return new_state

    def goal_test(self, state):
        """ Given a state, checks if this is the goal state.
         Returns True if it is, False otherwise."""
        return len(state.collected_treasures) == len(self.treasures.keys())

    def h(self, node):
        """ This is the heuristic. It gets a node
        and returns a goal distance estimate"""
        treasures_distance = []
        unique_t = []
        for name, t_loc in self.treasures.items():
            best_distance = self.s_to_treasures[name][0] * 2 - 1 # initilaize the furthest distance from base to current treasure and back
            if name in node.state.collected_treasures:
                # Treasure already depostied
                treasures_distance.append(0)
                continue
            for p in node.state.pirate_ships.values():
                # For each pirate check the following 
                if name == p["hand_1"] or name == p["hand_2"]:
                    # if the pirate holds the treasure, check if pirate is closer to base than the current best_distance
                    unique_t.append(name) 
                    temp_distance = self.s_to_base[p["location"]]
                    best_distance = min(temp_distance, best_distance)

                elif None == p["hand_1"] or None == p["hand_2"]:
                    # if the pirate doesn't holds the treasure, and he has an empty hand, check if he is closer to the treasure than best_distance
                    if p["location"] not in self.s_to_treasures[name][1]:
                        # treasure is unreachable
                        best_distance = math.inf 
                    else:
                        temp_distance = self.s_to_treasures[name][1][p["location"]] + self.s_to_treasures[name][0] - 1 # ? -2  # + 1
                        best_distance = min(temp_distance, best_distance)

            treasures_distance.append(best_distance)

        empty_hands = 0
        for p in node.state.pirate_ships.values():
            if None == p["hand_1"]:
                empty_hands += 1
            if None == p["hand_2"]:
                empty_hands += 1

        treasures_distance = sorted(treasures_distance)  
        num_t_to_collect = len(treasures_distance) - len(node.state.collected_treasures)
        max_dist = treasures_distance[-1]

        treasures_distance = [item for item in treasures_distance if item != 0]

        unique_t = set([t for t in unique_t if t not in node.state.collected_treasures]) # treasure that are currenlty held by pirates 
                                                                                         # and are not depostied yet
        deposit_num = math.ceil(num_t_to_collect / (2 * len(node.state.pirate_ships.keys()))) # number of deposits left

        temp = 0
        if empty_hands + len(unique_t) < num_t_to_collect:
            # calculates the number of times all the pirates will need to deposit and go back to collect
            for i in range(math.ceil((num_t_to_collect - len(unique_t) - empty_hands) / (2 * len(node.state.pirate_ships.keys()))) + 1):
                if i < len(treasures_distance):
                    temp += treasures_distance[i]

            max_dist = max(max_dist, temp)

        max_dist = max_dist + deposit_num
        if len(node.state.pirate_ships.keys()) >= 3:
            if len(self.map[0]) * len(self.map) > 16:
                return sum(treasures_distance) + deposit_num
            else:
                return sum(treasures_distance) / len(node.state.pirate_ships.keys()) + deposit_num
        return max_dist

    def L1_distance(self, x, y):
        return abs(x[0] - y[0]) + abs(x[1] +y[1])

    def bfs(self, start, t_or_b):
        '''
        @param start is a starting point in the map, can be treasure or base
        @param t_or_b if start is pointing at base or treasure
        @returns three values:
        distance_t_b is the real distance from base to treasure, including collect turn
        distatnce_b_s is a dictionary of distances from the base to each sea tile
        distance_t_s is a dictionary of distances from the treasure to each sea tile, including collect turn
        '''
        visited = set()
        distances_t_b = math.inf
        distances_b_s = {}
        distances_t_s = {}

        queue = deque([(start, 0)])

        while queue:
            current, distance = queue.popleft()
            x, y = current

            # Mark the current position as visited
            visited.add(current)

            # Store the distance if the current position is a treasure or the base
            if t_or_b == "t":
                if current == self.pirate_base:
                    distances_t_b = distance
                if self.map[x][y] == "S" or self.map[x][y] == "B":
                    distances_t_s[current] = distance

            if t_or_b == "b":
                if self.map[x][y] == "S":
                    distances_b_s[current] = distance

            # Explore all possible directions

            for orientation in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                n_x = x + orientation[0]
                n_y = y + orientation[1]
                new_location = (n_x, n_y)
                if not self.out_of_bounds(new_location) and \
                        (self.map[n_x][n_y] == "S" or self.map[n_x][n_y] == "B") and new_location not in visited:
                    queue.append((new_location, distance + 1))

        return distances_t_b, distances_b_s, distances_t_s

    def calculate_distances(self, treasures, base):
        # Calculate distances from every 'S' to every treasure
        s_to_treasures = {}
        for treasure_name, treasure_position in treasures.items():
            distances_t_b, distances_b_s, distances_t_s = self.bfs(treasure_position, "t")
            s_to_treasures[treasure_name] = [distances_t_b, distances_t_s]

        # Calculate distances from every 'S' to the base
        distances_t_b, distances_b_s, distances_t_s = self.bfs(base, "b")
        s_to_base = distances_b_s

        return s_to_treasures, s_to_base

    def out_of_bounds(self, new_location):
        # Check out of bounds:
        if new_location[0] >= len(self.map) or new_location[1] >= len(self.map[0]) \
                or new_location[0] < 0 or new_location[1] < 0:
            return True

        return False

    def next_marine_step(self, marin_path, current_location, direction):
        # update the marine location and it's direction
        if len(marin_path) == 1:
            return (0, 'r')

        if direction == 'r':
            if current_location + 1 == len(marin_path):
                return (current_location - 1, 'l')
            return (current_location + 1, 'r')

        else:
            if current_location == 0:
                return (current_location + 1, 'r')
            return (current_location - 1, 'l')


def create_onepiece_problem(game):
    return OnePieceProblem(game)
