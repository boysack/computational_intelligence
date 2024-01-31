from copy import deepcopy
import pickle
import random
from random import choice
from game import Game, Move, Player
import numpy as np
from tqdm import tqdm
import os
import sys

# need to compare results: lose < draw < win
# idx 0 for player 0 perspective, idx 1 for player 1 perspective
COMP_RES = [{
    -1:  0, # draw
     0:  1, # win
     1: -1  # lose
},{
    -1:  0, # draw
     0: -1, # lose
     1:  1  # win
}]

ROT_SLIDE = [{
    Move.TOP: 0,
    Move.LEFT: 1,
    Move.BOTTOM: 2,
    Move.RIGHT: 3
},{
    0: Move.TOP,
    1: Move.LEFT,
    2: Move.BOTTOM,
    3: Move.RIGHT
}]

class HumanPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'MyGame') -> tuple[tuple[int, int], Move]:
        col = int(input("insert a number from 0 to 4 to select the column -> "))
        row = int(input("insert a number from 0 to 4 to select the row -> "))
        from_pos = (col, row)
        while True:
            move = input("insert a string with the direction you want to move the piece (TOP, BOTTOM, LEFT, RIGTH) -> ").upper()
            move = [value for name, value in vars(Move).items() if name == move]
            if move:
                break
        return from_pos, move[0]

class RandomPlayer(Player):
    def __init__(self, player: int = 1) -> None:
        super().__init__()
        player = player % 2
        self.player = player

    def make_move(self, game: 'MyGame') -> tuple[tuple[int, int], Move]:
        """ from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT]) """
        available_moves = game.available_moves(self.player)
        from_pos, slide = random.choice(available_moves)
        return from_pos, slide

class MinMaxPlayer(Player):
    def __init__(self, soft: bool = False, player: int = 0, pruning_level: int = 2) -> None:
        super().__init__()
        self.soft = soft
        player = player % 2
        self.player = player
        self.pruning_level = pruning_level

        # recursion limit setting
        import sys
        if sys.getrecursionlimit() < (pruning_level+9):
            sys.setrecursionlimit(pruning_level+9)

    def make_move(self, game: 'MyGame') -> tuple[tuple[int, int], Move]:
        g_copy = deepcopy(game)
        # try to implement something to get equally rewarded moves to be chosen randomly, and not just taken the first
        moves = self.minmax(g_copy, player_id=g_copy.current_player_idx)[0]
        move = choice(moves)
        from_pos = (move[0][0], move[0][1])
        slide = move[1]
        return from_pos, slide

    def minmax(self, game: 'MyGame', level: int = 1, alpha = -np.inf, beta = np.inf, player_id: int = 0) -> tuple[tuple[int, int], Move]:
        available_moves = game.available_moves(player_id)

        if player_id == self.player: # my player plays always as MAX
            best = [[], -np.inf]
        else:
            best = [[], +np.inf]

        if len(available_moves) == 0 or game.check_winner() != -1 or level > self.pruning_level:
            #return [[], COMP_RES[self.player][game.check_winner()]]
            return [[], self.get_reward(game)]

        for move in available_moves: # same level nodes
            from_pos = move[0]
            slide = move[1]

            # backup - save row/column
            if slide == Move.LEFT or slide == Move.RIGHT:
                prev_values = deepcopy(game._board[from_pos[1], :])
            else:
                prev_values = deepcopy(game._board[:, from_pos[0]])

            # make a move
            if game._Game__move(from_pos, slide, player_id) == False:
                raise Exception("Invalid move chosen")
            score = self.minmax(game, level+1, alpha, beta, (player_id+1)%2)

            # restore - restore row/column
            if slide == Move.LEFT or slide == Move.RIGHT:
                game._board[from_pos[1]] = prev_values
            else:
                game._board[:, from_pos[0]] = prev_values

            score[0].append(move)

            if player_id == self.player: # my player plays always as MAX
                if score[1] > best[1]:
                    best = score  # max value
                    alpha = score[1]
                elif score[1] == best[1]:
                    best[0].extend(score[0])
            else:
                if score[1] < best[1]:
                    best = score  # min value
                    beta = score[1]
                elif score[1] == best[1]:
                    best[0].extend(score[0])

            if self.soft:
                if alpha >= beta: # <- PRUNE EVEN IF EQUAL
                    break
            else:
                if alpha > beta:
                    break
        return best

    def get_reward(self, g: 'MyGame'):
        p = self.player
        o = (self.player+1)%2
        b = g.get_board()

        # max number of player blocks for each row
        r_max = np.sum(b == p, axis=1).max()
        # max number of player blocks for each col
        c_max = np.sum(b == p, axis=0).max()
        # number of player blocks in the main diagonal
        md_max = np.sum(np.diagonal(b) == p)
        # number of player blocks in the other diagonal
        od_max = np.sum(np.diagonal(np.fliplr(b)) == p)

        # max number of player blocks for each row
        o_r_max = np.sum(b == o, axis=1).max()
        # max number of player blocks for each col
        o_c_max = np.sum(b == o, axis=0).max()
        # number of player blocks in the main diagonal
        o_md_max = np.sum(np.diagonal(b) == o)
        # number of player blocks in the other diagonal
        o_od_max = np.sum(np.diagonal(np.fliplr(b)) == o)

        return max(r_max, c_max, md_max, od_max) - max(o_r_max, o_c_max, o_md_max, o_od_max)

class QPlayer(Player):
    def __init__(self, alpha = .5, init_epsilon = .8, final_epsilon = .8, gamma = .8, input_filename = "./policies/Q_quixo", output_filename = "./policies/Q_quixo", mode: str = "val", m_state: str = "red", player: int = 0, iterations: int = 1000) -> None:
        super().__init__()
        if input_filename and os.path.isfile(input_filename):
            with open(input_filename, "rb") as f:
                self.Q = pickle.load(f)
        else:
            self.Q = {}

        self.alpha = alpha
        self.init_epsilon = init_epsilon
        self.epsilon = init_epsilon
        self.final_epsilon = final_epsilon
        self.gamma = gamma

        self.mode = mode
        self.m_state = m_state
        player %= 2
        self.player = player
        self.iterations = iterations

        self.output_filename = output_filename

    @staticmethod
    def state_to_hash(state: np.ndarray, m_state: str):
        if m_state == "bug":
            return frozenset(set.union(*map(set, state)))
        return tuple(state.flatten())

    # STATE: board, ACTION: move = ((col, row), slide)
    def get_value(self, state, action):
        hashable_state = QPlayer.state_to_hash(state, self.m_state)
        #print(hashable_state)
        if (hashable_state, action) not in self.Q:
            self.Q[(hashable_state, action)] = 0.0
        return self.Q[(hashable_state, action)]

    def rotate_coord_countclock(self, action, shape):
        _, cols = shape
        from_pos, slide = action
        col, row = from_pos
        rot_row = cols - 1 - col
        rot_col = row
        rot_slide = ROT_SLIDE[0][slide]
        rot_slide += 1
        rot_slide %= 4
        rot_slide = ROT_SLIDE[1][rot_slide]

        return ((rot_col, rot_row), rot_slide)

    def rotate_countclock(self, state: np.ndarray, action: tuple[tuple[int,int], Move]) -> tuple[np.ndarray, tuple]:
        return (np.rot90(state), self.rotate_coord_countclock(action, state.shape))
    
    # MUST RETURN EVEN THE TRANSFORMATION..
    # return r_hashable_state, r_action, real_action, Q-value
    def get_value_wsymmetry(self, state, action):
        hashable_state = QPlayer.state_to_hash(state, self.m_state)
        if (hashable_state, action) in self.Q:
            return hashable_state, action, action, self.Q[(hashable_state, action)]
        #  90 rotation
        r_state, r_action = self.rotate_countclock(state, action)
        r_hashable_state = QPlayer.state_to_hash(r_state, self.m_state)
        if (r_hashable_state, r_action) in self.Q:
            return r_hashable_state, r_action, action, self.Q[(r_hashable_state, r_action)]
        # 180 rotation
        r_state, r_action = self.rotate_countclock(r_state, r_action)
        r_hashable_state = QPlayer.state_to_hash(r_state, self.m_state)
        if (r_hashable_state, r_action) in self.Q:
            return r_hashable_state, r_action, action, self.Q[(r_hashable_state, r_action)]
        # 270 rotation
        r_state, r_action = self.rotate_countclock(r_state, r_action)
        r_hashable_state = QPlayer.state_to_hash(r_state, self.m_state)
        if (r_hashable_state, r_action) in self.Q:
            return r_hashable_state, r_action, action, self.Q[(r_hashable_state, r_action)]
        # not present
        self.Q[(hashable_state, action)] = 0.0
        return hashable_state, action, action, self.Q[(hashable_state, action)]

    # function to reduce dimensionality of the state
    def reduce_state(self, b) -> np.ndarray:
        p = self.player
        o = (self.player+1)%2

        # index of row with max objects
        r_max = np.sum(b == p, axis=1)
        if np.all(r_max == -1):
            r_max = -1
        else:
            r_max = r_max.argmax()
        # index of row with max objects
        c_max = np.sum(b == p, axis=0)
        if np.all(c_max == -1):
            c_max = -1
        else:
            c_max = c_max.argmax()
        # number of player blocks in the main diagonal
        md_max = np.sum(np.diagonal(b) == p)
        # number of player blocks in the other diagonal
        od_max = np.sum(np.diagonal(np.fliplr(b)) == p)

        # index of row with max objects
        o_r_max = np.sum(b == o, axis=1)
        if np.all(o_r_max == -1):
            o_r_max = -1
        else:
            o_r_max = o_r_max.argmax()
        # index of row with max objects
        o_c_max = np.sum(b == o, axis=0)
        if np.all(o_c_max == -1):
            o_c_max = -1
        else:
            o_c_max = o_c_max.argmax()
        # number of player blocks in the main diagonal
        o_md_max = np.sum(np.diagonal(b) == o)
        # number of player blocks in the other diagonal
        o_od_max = np.sum(np.diagonal(np.fliplr(b)) == o)

        if md_max == 0 and od_max == 0:
            d_max = -1
        else:
            d_max = np.argmax([md_max, od_max])
        
        if o_md_max == 0 and o_od_max == 0:
            o_d_max = -1
        else:
            o_d_max = np.argmax([o_md_max, o_od_max])

        if self.m_state == "red1":
            return np.array([r_max, c_max, md_max, od_max, o_r_max, o_c_max, o_md_max, o_od_max])
        elif self.m_state == "red2":
            return np.array([r_max, c_max, d_max, o_r_max, o_c_max, o_d_max])
        elif self.m_state == "red3":
            return np.array([r_max, c_max, md_max, od_max])
        else:
            raise Exception(f"Modality not valid: {self.m_state}")

    def make_move(self, game: 'MyGame') -> tuple[tuple[int, int], Move]:
        available_moves = game.available_moves(self.player)

        if self.mode == "train" and np.random.uniform() < self.epsilon:
            return choice(available_moves)
        else:
            if self.m_state == "red1" or self.m_state == "red2" or self.m_state == "red3":
                Q_values = [self.get_value(self.reduce_state(game._board), move) for move in available_moves]
            elif self.m_state == "sym":
                Q_values = [self.get_value_wsymmetry(game._board, move)[3] for move in available_moves]
            elif self.m_state == "bug":
                Q_values = [self.get_value(game._board, move) for move in available_moves]
            else:
                raise Exception(f"Modality not valid: {self.m_state}")
            max_Q = max(Q_values)
            if Q_values.count(max_Q) > 1:
                best_moves = [i for i in range(len(available_moves)) if Q_values[i] == max_Q]
                i = choice(best_moves)
            else:
                i = Q_values.index(max_Q)
            return available_moves[i]

    def update(self, state, action, reward, next_state, next_available_moves):
        if self.m_state == "red1" or self.m_state == "red2" or self.m_state == "red3":
            next_Q_values = [self.get_value(self.reduce_state(next_state), next_action) for next_action in next_available_moves]    
            max_next_Q = max(next_Q_values) if next_Q_values else 0.0
            Q_value = self.get_value(self.reduce_state(state), action)
            hashable_state = QPlayer.state_to_hash(self.reduce_state(state), self.m_state)
        elif self.m_state == "sym":
            next_Q_values = [self.get_value_wsymmetry(next_state, next_action) for next_action in next_available_moves]
            _, _, _, max_next_Q = max(next_Q_values, key=lambda i: i[3]) if next_Q_values else 0.0
            hashable_state, action, _, Q_value = self.get_value_wsymmetry(state, action)
        elif self.m_state == "bug":
            next_Q_values = [self.get_value(next_state, next_action) for next_action in next_available_moves]    
            max_next_Q = max(next_Q_values) if next_Q_values else 0.0
            Q_value = self.get_value(state, action)
            hashable_state = QPlayer.state_to_hash(state, self.m_state)
        else:
            raise Exception(f"Modality not valid: {self.m_state}")

        self.Q[(hashable_state, action)] = Q_value + self.alpha * (reward + self.gamma * max_next_Q - Q_value)

    def save_Q(self):
        if self.output_filename is None:
            print("policy not saved!")
        else:
            with open(self.output_filename, 'wb') as f:
                pickle.dump(self.Q, f)
            print("policy saved succesfully!")
        

    def train_test(self, second_player: Player = RandomPlayer()):
        self.mode = "train"
        e = np.linspace(self.init_epsilon, self.final_epsilon, self.iterations)
        
        for i in tqdm(range(self.iterations)):
            self.epsilon = e[i]

            g = MyGame()
            prev_state = None
            prev_from_pos = None
            prev_slide = None

            while g.available_moves(g.current_player_idx) and g.check_winner() == -1:
                if prev_state is not None and prev_from_pos is not None and prev_slide is not None:
                    self.update(prev_state, (prev_from_pos, prev_slide), COMP_RES[0][g.check_winner()], g.get_board(), g.available_moves(g.current_player_idx))
                from_pos, slide = self.make_move(g)

                prev_state = g.get_board()
                prev_from_pos = from_pos
                prev_slide = slide

                g._Game__move(from_pos, slide, g.current_player_idx)
                g.switch_player()

                if len(g.available_moves(g.current_player_idx)) == 0 or g.check_winner() != -1:
                    self.update(prev_state, (prev_from_pos, prev_slide), COMP_RES[0][g.check_winner()], g.get_board(), g.available_moves(g.current_player_idx))
                    break

                from_pos, slide = second_player.make_move(g)
                g._Game__move(from_pos, slide, g.current_player_idx)
                g.switch_player()
        self.save_Q()
        self.mode = "val"

    def train(self, second_player: Player = RandomPlayer()):
        self.mode = "train"
        e = np.linspace(self.init_epsilon, self.final_epsilon, self.iterations)
        second_player.player = (self.player+1)%2

        for i in tqdm(range(self.iterations)):
            self.epsilon = e[i]
            g = MyGame()

            if self.player == 1:
                ok = False
                while not ok:
                    from_pos, slide = second_player.make_move(g)
                    ok = g._Game__move(from_pos, slide, second_player.player)
                    if not ok:
                        g.print()
                        raise Exception(f"player {self.player} made a wrong decision | move: {(from_pos[1], from_pos[0]), slide}")
            
            while g.available_moves(g.current_player_idx) and g.check_winner() == -1:
                state = deepcopy(g._board)

                ok = False
                while not ok:
                    from_pos, slide = self.make_move(g)
                    ok = g._Game__move(from_pos, slide, self.player)
                    if not ok:
                        g.print()
                        raise Exception(f"player {self.player} made a wrong decision | move: {(from_pos[1], from_pos[0]), slide}")

                if len(g.available_moves(second_player.player)) == 0 or g.check_winner() != -1:
                    next_state = deepcopy(g._board)
                else:
                    ok = False
                    while not ok:
                        from_pos, slide = second_player.make_move(g)
                        ok = g._Game__move(from_pos, slide, second_player.player)
                        if not ok:
                            g.print()
                            raise Exception(f"player {second_player.player} made a wrong decision | move: {(from_pos[1], from_pos[0]), slide}")

                    next_state = deepcopy(g._board)

                reward = COMP_RES[self.player][g.check_winner()]
                self.update(state, (from_pos, slide), reward, next_state, g.available_moves(self.player))
        self.save_Q()
        self.mode = "val"

class EAPlayer(Player):
    def __init__(self,
                 input_filename = "./policies/EA_quixo",
                 output_filename = "./policies/EA_quixo",
                 mode: str = "val",
                 n_games_fitness: int = 1_000,
                 player: int = 0,
                 second_player: Player = RandomPlayer(),
                 iterations: int = 50,
                 params: np.ndarray = None,
                 pop_size: int = 20,
                 off_size: int = 10,
                 tou_size: int = 15,  # increase to increase selective pressure
                 mut_prob: float = .15,
                 mut_rep: float = .05,
                 sigma: float = .025
                ):
        super().__init__()
        
        # population size
        self.pop_size = pop_size
        # offpring size
        self.off_size = off_size
        # tournament size
        self.tou_size = min(tou_size, pop_size)

        ## NOT USED (NO RECOMBINATION FOR NOW)
        self.mut_prob = mut_prob
        self.mut_rep = mut_rep

        # gaussian mutation standard deviation
        self.sigma = sigma

        # number of times new offspring is created
        self.iterations = iterations
        # number of games that are performed to evaluate a solution
        self.n_games_fitness = n_games_fitness

        # player idx
        self.player = player
        # opponent player type
        self.second_player = second_player

        self.mode = mode
        self.output_filename = output_filename

        if params is not None:
            self.params = params
            self.fitness = self.win_rate()
        elif input_filename and os.path.isfile(input_filename):
            with open(input_filename, "rb") as f:
                restore = pickle.load(f)
                self.params = restore[0]
                self.fitness = restore[1]
        else:
            self.params, self.fitness = self.gaussian_mutation(init=True)

    def fitness_fun(self, params: np.ndarray) -> float:
        return EAPlayer(params=params,
                         second_player=self.second_player,
                         n_games_fitness = self.n_games_fitness,
                         player = self.player,
                         mode="train"
                         ).fitness

    def win_rate(self) -> float:
        wins = 0
        for _ in range(self.n_games_fitness):
            g = MyGame()
            win = COMP_RES[self.player][g.play(self, self.second_player)]
            if (win == 1):
                wins += 1
        return wins/self.n_games_fitness
    
    def gaussian_mutation(self, parent: np.ndarray = None, init: bool =False):
        if init:
            params = np.random.rand(MyGame.MOVES_NUM)
        else:
            params = np.random.normal(loc=parent, scale=self.sigma)
        params = np.abs(params)
        params /= np.sum(params)
        return params, self.fitness_fun(params=params)

    def tournament_selection(self, population: np.ndarray) -> tuple[np.ndarray, float]:
        idx = np.random.choice(range(len(population)), size=self.tou_size, replace=False)
        tournament = [population[i] for i in idx]
        champion = max(tournament, key=lambda i: i[1]) #keep the champion based on fitness max value
        return champion
    
    def generate_offspring(self, population: list = None, init: bool=False) -> list:
        # if rand < mut_prob:
        #   mut
        # else:
        #   rec

        offspring = []
        if init:
            # random initialization of population
            for _ in range(self.pop_size):
                rand_individual = self.gaussian_mutation(init=True)
                offspring.append((rand_individual))
        else:
            for _ in range(self.off_size):
                # champion = tuple[ndarray, float]
                champion = self.tournament_selection(population)
                new_individual = self.gaussian_mutation(champion[0])
                offspring.append(new_individual)
        return offspring
    
    def train(self):
        self.mode == "train"

        population = []
        begin = True
        print(f"training EAPlayer with sigma = {self.sigma}")
        for _ in tqdm(range(self.iterations+1)): #count the initialization stage
            # list tuple[np.ndarray, float]
            new_offspring = self.generate_offspring(population, init=begin)
            population += new_offspring
            population.sort(key=lambda i: i[1], reverse=True)
            # check if the order is correct - OK
            population = population[:self.pop_size]
            begin = False
            print(f"fitness: {population[0][1]}")

        self.params = population[0][0]
        self.fitness = population[0][1]
        self.save_params()
        self.mode == "val"

    def make_move(self, game: 'MyGame') -> tuple[tuple[int, int], Move]:
        # choices return a list, so take the [0]
        move = random.choices(game.possible_moves_l, weights=self.params)[0]
        available_moves = game.available_moves(self.player)
        while move not in available_moves:
            move = random.choices(game.possible_moves_l, weights=self.params)[0]
        from_pos, slide = move
        return from_pos, slide

    def save_params(self):
        with open(self.output_filename, 'wb') as f:
            params_fitness = (self.params, self.fitness)
            pickle.dump(params_fitness, f)

class MyGame(Game):
    MOVES_NUM = 44
    def __init__(self):
        super(MyGame, self).__init__()
        self.possible_moves_l = self.available_moves(0)

    def move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        ok = self._Game__move(from_pos, slide, player_id)
        if ok:
            self.switch_player()
        return ok

    # receive (column, row) moves
    def check_move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        if player_id > 2:
            return False
        acceptable = self.__check_take((from_pos[1], from_pos[0]), player_id) and self.__check_slide((from_pos[1], from_pos[0]), slide)
        return acceptable

    def __check_take(self, from_pos: tuple[int, int], player_id: int) -> bool:
        # acceptable only if in border
        acceptable: bool = (
            # check if it is in the first row
            (from_pos[0] == 0 and from_pos[1] < 5)
            # check if it is in the last row
            or (from_pos[0] == 4 and from_pos[1] < 5)
            # check if it is in the first column
            or (from_pos[1] == 0 and from_pos[0] < 5)
            # check if it is in the last column
            or (from_pos[1] == 4 and from_pos[0] < 5)
            # and check if the piece can be moved by the current player
        ) and (self._board[from_pos] < 0 or self._board[from_pos] == player_id)
        return acceptable

    def __check_slide(self, from_pos: tuple[int, int], slide: Move) -> bool:
        # define the corners
        SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
        # if the piece position is not in a corner
        if from_pos not in SIDES:
            # if it is at the TOP, it can be moved down, left or right
            acceptable_top: bool = from_pos[0] == 0 and (
                slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is at the BOTTOM, it can be moved up, left or right
            acceptable_bottom: bool = from_pos[0] == 4 and (
                slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is on the LEFT, it can be moved up, down or right
            acceptable_left: bool = from_pos[1] == 0 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
            )
            # if it is on the RIGHT, it can be moved up, down or left
            acceptable_right: bool = from_pos[1] == 4 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT
            )
        # if the piece position is in a corner
        else:
            # if it is in the upper left corner, it can be moved to the right and down
            acceptable_top: bool = from_pos == (0, 0) and (
                slide == Move.BOTTOM or slide == Move.RIGHT)
            # if it is in the lower left corner, it can be moved to the right and up
            acceptable_left: bool = from_pos == (4, 0) and (
                slide == Move.TOP or slide == Move.RIGHT)
            # if it is in the upper right corner, it can be moved to the left and down
            acceptable_right: bool = from_pos == (0, 4) and (
                slide == Move.BOTTOM or slide == Move.LEFT)
            # if it is in the lower right corner, it can be moved to the left and up
            acceptable_bottom: bool = from_pos == (4, 4) and (
                slide == Move.TOP or slide == Move.LEFT)
        # check if the move is acceptable
        acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
        return acceptable

    def available_moves(self, player_idx) -> list:
        a_m = []
        for y in range(5):
            from_pos = (y, 0)
            for slide in Move:
                if self.check_move(from_pos, slide, player_idx):
                    a_m.append((from_pos, slide))
            from_pos = (y, 4)
            for slide in Move:
                if self.check_move(from_pos, slide, player_idx):
                    a_m.append((from_pos, slide))
        for x in range(1,4): # ignore corners since already appended
            from_pos = (0, x)
            for slide in Move:
                if self.check_move(from_pos, slide, player_idx):
                    a_m.append((from_pos, slide))
            from_pos = (4, x)
            for slide in Move:
                if self.check_move(from_pos, slide, player_idx):
                    a_m.append((from_pos, slide))
        return a_m

    def switch_player(self):
        self.current_player_idx += 1
        self.current_player_idx %= 2

def player_test(pov: int = 0, player1: Player = RandomPlayer(), player2: Player = RandomPlayer(player=1), evaluation_step: int = 10_000) -> float:
    wins = 0
    for _ in tqdm(range(evaluation_step)):
        g = MyGame()
        winner = g.play(player1, player2)
        if COMP_RES[pov][winner] == 1:
            wins += 1
    print(f"wins: {wins}")
    print(f"win rate: {wins/evaluation_step:.2%}")

    return wins/evaluation_step

if __name__ == '__main__':
    # setting the working dir to the one containing the program
    working_dir = os.path.sep.join(os.path.abspath(sys.argv[0]).split(os.path.sep)[:-1])
    os.chdir(working_dir)

    if len(sys.argv) < 2 or sys.argv[1] == 1:
        #### EVALUATION SECTION #####
        # TEST FUNCTIONALITY WITH RANDOM
        """
        print("-------- Random (just testing functionality) --------")
        player_test(evaluation_step=50)
        """
        # TEST MINMAX

        print("----------------------- MinMax ----------------------")
        print(" - testing as player 0...")
        player_test(player1=MinMaxPlayer(), evaluation_step=50)

        print(" - testing as player 1...")
        player_test(pov=1, player1=RandomPlayer(player=0), player2=MinMaxPlayer(player=1), evaluation_step=50)
       
        # TRAIN AND TEST A QLEARNING AGENT
        """
        print("--------------------- QLearning ---------------------")
        print(" - training new model and testing it as player 0...")
        qplayer = QPlayer(input_filename=None, m_state="bug")
        qplayer.train()
        player_test(player1=qplayer)
        """
        # TEST QLEARNING

        print("--------------------- QLearning ---------------------")
        print(" - testing the buggy previous run as player 0...")
        qplayer = QPlayer(input_filename="./policies/Q_bug_rand_init/Q_bug_rand_07", output_filename=None, m_state="bug")
        player_test(player1=qplayer, evaluation_step=5_000)

        print(" - testing the buggy previous run as player 1...")
        qplayer.player = 1
        player_test(pov=1, player1=RandomPlayer(player=0), player2=qplayer, evaluation_step=5_000)

        print(" - testing the reduced-state-space previous run as player 0...")
        qplayer = QPlayer(input_filename="./policies/Q_red3_100K_30K_ft", m_state="red3")
        player_test(player1=qplayer, evaluation_step=5_000)

        print(" - testing the reduced-state-space previous run as player 1...")
        qplayer.player = 1
        player_test(pov=1, player1=RandomPlayer(player=0), player2=qplayer, evaluation_step=5_000)
       
        # TRAIN AND TEST EVOLUTIONARY ALGORITHM AGENT
        """ 
        print("------------------------- EA ------------------------")
        eaplayer = EAPlayer()
        eaplayer.train()
        player_test(player1=eaplayer)
        """
        # TEST EA
        
        print("------------------------- EA ------------------------")
        print(" - testing a previous run...")
        eaplayer = EAPlayer(input_filename="./policies/EA_100")
        player_test(player1=eaplayer, evaluation_step=5_000)
        eaplayer.player = 1
        player_test(pov=1, player1=RandomPlayer(player=0), player2=eaplayer, evaluation_step=5_000)

    else:
        #### EXPLORATIVE SECTION ####

        ############# EA ############
        
        # TRAINING FOR 100
        """ 
        eaplayer = EAPlayer(input_filename=None, output_filename="./policies/EA_100", iterations=100)
        eaplayer.train()
        player_test(player1=eaplayer)
        eaplayer.player = 1
        player_test(pov=1, player1=RandomPlayer(player=0), player2=eaplayer)
         """
        # TOP K MOVES ACCORDING TO EA PLAYER
        """ 
        k = 5
        eaplayer = EAPlayer(input_filename="./policies/EA_100")
        best_move_idx = np.argsort(eaplayer.params)[-k:]
        g = MyGame()
        for idx in best_move_idx[::-1]:
            print(eaplayer.params[idx])
            print(eaplayer.fitness)
            print(g.possible_moves_l[idx])
         """

        ######### QLEARNING #########

        # RANDOM POLICY INIT - BUGGY SOLUTION
        """ 
        folder = "./policies/Q_bug_rand_init"
        base = "/Q_bug_rand"
        top_rate = 0.0
        top_idx = -1
        for i in range(10):
            filename = folder+base+f"_{i:02}"
            print(f"train for {filename}")
            qplayer = QPlayer(alpha=.3, gamma=.99, init_epsilon=1.0, final_epsilon=1.0, input_filename=None, output_filename=filename,  m_state="bug")
            qplayer.train()
            print(f"test for {filename}")
            win_rate = player_test(player1=qplayer)
            if win_rate > top_rate:
                top_rate = win_rate
                top_idx = i
        
        best = folder+base+f"_{top_idx:02}"
        print(f"best random initialization saved in: {best}")
        """
        # TRAINING FOR 100K - FT FOR 30K - REDUCED3 SOLUTION
        """ 
        qplayer = QPlayer(alpha=.3, gamma=.99, init_epsilon=1.0, final_epsilon=1.0, input_filename=None, output_filename="./policies/Q_red3_100K", m_state="red3", iterations=100_000)
        qplayer.train()
        qplayer = QPlayer(alpha=.3, gamma=.99, init_epsilon=.8, final_epsilon=.4, input_filename="./policies/Q_red3_100K", output_filename="./policies/Q_red3_100K_30K_ft", m_state="red3", iterations=30_000)
        qplayer.train()
         """
        # TESTING DIFFERENT PARAMETERS VALUES
        # best: alpha = 0.3 | gamma = 0.99 | f_epsilon = 0.2
        """ alpha = [.3,.4,.5]
        gamma = [.9, .95, .99]
        final_epsilon = [.3, .2, .1]
        for a in alpha:
            for g in gamma:
                for e in final_epsilon:
                    p_a = f"{a}".lstrip('0.')
                    p_g = f"{g}".lstrip('0.')
                    if p_g == "9":
                        p_g = "90"
                    p_e = f"{e}".lstrip('0.')
                    print(f"alpha = {a} | gamma = {g} | f_epsilon = {e}")
                    qplayer = QPlayer(input_filename=None, output_filename="./Q_x_val/Q_a"+p_a+"_g"+p_g+"_fe"+p_e, alpha=a, gamma=g, final_epsilon=e)
                    qplayer.train()
                    player_test(player1=qplayer)
                    print(end="\n\n") """