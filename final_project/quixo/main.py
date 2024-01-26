from copy import deepcopy
import pickle
import random
from random import choice
from game import Game, Move, Player
import numpy as np
from tqdm import tqdm
import os

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
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'MyGame') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

class MinMaxPlayer(Player):
    def __init__(self, soft: bool = False, player: int = 0, pruning_level: int = 3) -> None:
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
        moves = self.minmax(g_copy)[0]
        move = choice(moves)
        from_pos = (move[0][0], move[0][1])
        slide = move[1]
        return from_pos, slide

    def minmax(self, game: 'MyGame', level: int = 1, alpha = -np.inf, beta = np.inf) -> tuple[tuple[int, int], Move]:
        player_id = game.current_player_idx
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

            # backup - player ??
            #prev_player = game.current_player_idx

            # make a move
            if game.move(from_pos, slide, player_id) == False:
                raise Exception("Invalid move chosen")
            score = self.minmax(game, level+1, alpha, beta)

            # restore - restore row/column
            if slide == Move.LEFT or slide == Move.RIGHT:
                game._board[from_pos[1]] = prev_values
            else:
                game._board[:, from_pos[0]] = prev_values

            # restore - player ??
            #game.current_player_idx = prev_player

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
    def __init__(self, alpha = .5, epsilon = .8, final_epsilon = .8, gamma = .8, input_filename = "Q_quixo", output_filename = "Q_quixo", mode: str = "val", player: int = 0, iterations: int = 1000) -> None:
        super().__init__()
        if input_filename and os.path.isfile(input_filename):
            with open(input_filename, "rb") as f:
                self.Q = pickle.load(f)
        else:
            self.Q = {}
        self.alpha = alpha
        self.init_epsilon = epsilon
        self.epsilon = epsilon
        self.final_epsilon = final_epsilon
        self.gamma = gamma
        self.mode = mode
        self.player = player
        self.iterations = iterations

        self.output_filename = output_filename

    # state: board, action: move = ((col, row), slide)
    def get_value(self, state, action):
        hashable_state = QPlayer.state_to_set(state)
        if (hashable_state, action) not in self.Q:
            self.Q[(hashable_state, action)] = 0.0
        return self.Q[(hashable_state, action)]

    @staticmethod
    def state_to_set(state: np.ndarray):
        return frozenset(set.union(*map(set, state)))

    def make_move(self, game: 'MyGame') -> tuple[tuple[int, int], Move]:
        available_moves = game.available_moves(self.player)

        if self.mode == "train" and np.random.uniform() < self.epsilon:
            return choice(available_moves)
        else:
            Q_values = [self.get_value(game._board, move) for move in available_moves]
            max_Q = max(Q_values)
            if Q_values.count(max_Q) > 1:
                best_moves = [i for i in range(len(available_moves)) if Q_values[i] == max_Q]
                i = choice(best_moves)
            else:
                i = Q_values.index(max_Q)
            return available_moves[i]

    def update(self, state, action, reward, next_state, next_available_moves):
        next_Q_values = [self.get_value(next_state, next_action) for next_action in next_available_moves]
        max_next_Q = max(next_Q_values) if next_Q_values else 0.0

        Q_value = self.get_value(state, action)
        hashable_state = QPlayer.state_to_set(state)
        self.Q[(hashable_state, action)] = Q_value + self.alpha * (reward + self.gamma * max_next_Q - Q_value)

    def save_Q(self):
        with open(self.output_filename, 'wb') as f:
            pickle.dump(self.Q, f)

    def train_test(self, second_player: Player = RandomPlayer()):
        e = np.linspace(self.init_epsilon, self.final_epsilon, self.iterations)
        self.mode = "train"
        for i in tqdm(range(self.iterations)):
            g = MyGame()
            player2 = second_player

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

                from_pos, slide = player2.make_move(g)
                g._Game__move(from_pos, slide, g.current_player_idx)
                g.switch_player()
            self.epsilon = e[i]
        self.save_Q()
        self.mode = "val"
        self.epsilon = self.init_epsilon

    def train(self, second_player: Player = RandomPlayer()):
        self.mode = "train" # not necessary anymore
        e = np.linspace(self.epsilon, self.final_epsilon, self.iterations)
        for i in tqdm(range(self.iterations)):
            g = MyGame()
            player2 = second_player

            while g.available_moves(g.current_player_idx) and g.check_winner() == -1:
                state = deepcopy(g._board)
                from_pos, slide = self.make_move(g)
                g._Game__move(from_pos, slide, g.current_player_idx)
                g.switch_player()

                if len(g.available_moves(g.current_player_idx)) == 0 or g.check_winner() != -1:
                    next_state = deepcopy(g._board)
                else:
                    from_pos_2, slide_2 = player2.make_move(g)

                    g._Game__move(from_pos_2, slide_2, g.current_player_idx)
                    g.switch_player()

                    next_state = deepcopy(g._board)

                reward = COMP_RES[0][g.check_winner()]
                self.update(state, (from_pos, slide), reward, next_state, g.available_moves(g.current_player_idx))
            self.epsilon = e[i]
        self.save_Q()
        self.mode = "val" # not necessary anymore
        self.epsilon = self.init_epsilon

class Individual:
    def __init__(self, params: np.ndarray = None, second_player: Player = RandomPlayer(), n_games_fitness: int = 100) -> None:
        if params is not None:
            self.params = params
        else:
            params = np.random.rand(MyGame.MOVES_NUM) # don't know if, with gaussian mutation, it makes sense to have probability
            params /= np.sum(params)
            self.params = params
        self.n_games_fitness = n_games_fitness
        self.fitness = self.fitness_fun(second_player=second_player)
    
    def gaussian_mutation(self, s: float = .1):
        params = np.random.normal(loc=self.params, scale=s)
        params /= np.sum(params)
        return Individual(params, second_player=RandomPlayer())

    def fitness_fun(self, second_player: Player = RandomPlayer()) -> float:
        wins = 0
        ga = GAPlayer(params=self.params)
        for _ in range(self.n_games_fitness):
            g = MyGame()
            win = COMP_RES[ga.player][g.play(ga, second_player)]
            if (win == 1):
                wins += 1
        return wins/self.n_games_fitness

    def __str__(self) -> str:
        return f"{self.params}"
    def __gt__(self, other):
        return self.fitness > other.fitness
    def __lt__(self, other):
        return self.fitness < other.fitness
    def __ge__(self, other):
        return self.fitness >= other.fitness
    def __le__(self, other):
        return self.fitness <= other.fitness

class GAPlayer(Player):
    def __init__(self,
                 input_filename = "GA_quixo",
                 output_filename = "GA_quixo",
                 mode: str = "val",
                 n_games_fitness: int = 500,
                 player: int = 0,
                 second_player: Player = RandomPlayer(),
                 iterations: int = 50,
                 params: np.ndarray = None,
                 pop_size: int = 20,
                 off_size: int = 10,
                 tou_size: int = 10,  # increase to increase selective pressure
                 mut_prob: float = .15,
                 mut_rep: float = .05,
                 sigma: float = .01
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
        # NOT USED
        self.mode = mode
        self.output_filename = output_filename

        if params is not None:
            self.params = params
            self.fitness = self.win_rate()

        elif input_filename and os.path.isfile(input_filename):
            with open(input_filename, "rb") as f:
                restore = pickle.load(f)
                self.params = restore[0]
                self.fitness = restore[1] #?
        else:
            params = np.random.rand(MyGame.MOVES_NUM) # don't know if, with gaussian mutation, it makes sense to have probability
            params /= np.sum(params)
            self.params = params
            self.fitness = self.win_rate()

    #### IN FORSE: QUELLE PRESE DA INDIVIDUAL
    def fitness_fun(self, params: np.ndarray) -> float:
        return GAPlayer(params=params,
                         second_player=self.second_player,
                         n_games_fitness = self.n_games_fitness,
                         player = self.player
                         ).fitness

    def win_rate(self) -> float:
        wins = 0
        for _ in range(self.n_games_fitness):
            g = MyGame()
            win = COMP_RES[self.player][g.play(self, self.second_player)]
            if (win == 1):
                wins += 1
        return wins/self.n_games_fitness
    
    def gaussian_mutation(self, parent):
        params = np.random.normal(loc=parent, scale=self.sigma)
        params = np.abs(params)
        params /= np.sum(params)
        return params, self.fitness_fun(params=params)
    #### IN FORSE: QUELLE PRESE DA INDIVIDUAL

    def generate_offspring(self, population: list = None, init: bool=False) -> list:
        """ # in teoria
        # if rand < mut_prob:
        #   mut
        # else:
        #   rec
        # per ora solo mutation """
        offspring = []
        if init:
            # random initialization of population
            for _ in range(self.pop_size):
                params = np.random.rand(MyGame.MOVES_NUM)
                params = np.abs(params)
                params /= np.sum(params)
                offspring.append((params, self.fitness_fun(params)))
        else:
            for _ in range(self.off_size):
                # just mutation for now
                champion = self.tournament_selection(population)
                new_individual = self.gaussian_mutation(champion[0])
                offspring.append(new_individual)
        return offspring
    
    def tournament_selection(self, population: np.ndarray) -> Individual:
        idx = np.random.choice(range(len(population)), size=self.tou_size, replace=False)
        tournament = [population[i] for i in idx]
        champion = max(tournament, key=lambda i: i[1])
        return champion

    def train(self):
        # population initialization
        population = []
        begin = True
        for _ in tqdm(range(self.iterations)):
            new_offspring = self.generate_offspring(population, init=begin)
            population += new_offspring
            population.sort(key=lambda i: i[1], reverse=True)
            population = population[::-1]
            population = population[:self.pop_size]
            begin = False

        self.params = population[0]
        self.save_params()

    def make_move(self, game: 'MyGame') -> tuple[tuple[int, int], Move]:
        move = random.choices(game.possible_moves_l, weights=self.params)[0]
        available_moves = game.available_moves(self.player)
        while move not in available_moves:
            move = random.choices(game.possible_moves_l, weights=self.params)[0]
        from_pos, slide = move
        return from_pos, slide

    def save_params(self):
        with open(self.output_filename, 'wb') as f:
            pickle.dump(self.params, f)

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
        # for x in range(5):
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
    
    def possible_moves(self):
        p_m = []
        for y in range(5):
            from_pos = (y, 0)
            for slide in Move:
                p_m.append((from_pos, slide))
            from_pos = (y, 4)
            for slide in Move:
                p_m.append((from_pos, slide))
        for x in range(1, 4):
            from_pos = (0, x)
            for slide in Move:
                p_m.append((from_pos, slide))
            from_pos = (4, x)
            for slide in Move:
                p_m.append((from_pos, slide))
        return p_m

    def switch_player(self):
        self.current_player_idx+=1
        self.current_player_idx%=2

def player_test(pov: int = 0, player1: Player = RandomPlayer(), player2: Player = RandomPlayer(), evaluation_step: int = 10_000):
    wins = 0
    for _ in tqdm(range(evaluation_step)):
        g = MyGame()
        winner = g.play(player1, player2)
        if COMP_RES[pov][winner]==1:
            wins += 1
    print(f"wins: {wins}")
    print(f"win rate: {wins/evaluation_step:.2%}")

if __name__ == '__main__':
    """
    print("-------- Random (just testing functionality) --------")
    player_test(evaluation_step=50)
    """
    """
    print("----------------------- MinMax ----------------------")
    player_test(player1=MinMaxPlayer(), evaluation_step=50)
    """
    """
    print("--------------------- QLearning ---------------------")
    print(" - training new model and testing it...")
    qplayer = QPlayer(input_filename=None)
    qplayer.train_test()
    #qplayer.train()
    player_test(player1=qplayer)
    """
    """
    print("--------------------- QLearning ---------------------")
    print(" - testing a previous lucky run...")
    qplayer = QPlayer(input_filename="Q_model")
    player_test(player1=qplayer)
    """
    """
    print("------------------------- GA ------------------------")
    gaplayer = GAPlayer()
    ga.train()
    player_test(player1=gaplayer)
    """

    # policy creation
    """ base = "Q_iter"
    folder = "./Q_test/"
    for i in range(10):
        filename = folder+base+"_tt"+f"_{i:02}"
        q = QPlayer(output_filename=filename)
        q.train_test()
        print(f"testing: {filename}")
        player_test(player1=q, evaluation_step=1000)

        filename = folder+base+"_t"+f"_{i:02}"
        q = QPlayer(output_filename=filename)
        q.train()
        print(f"testing: {filename}")
        player_test(player1=q, evaluation_step=1000) """

    # best policies test
    """ policy_name = "./Q_test/Q_iter_tt_01"
    print(f"{policy_name}")
    player_test(player1=QPlayer(input_filename=policy_name), evaluation_step=100_000)

    policy_name = "./Q_test/Q_iter_tt_05"
    print(f"{policy_name}")
    player_test(player1=QPlayer(input_filename=policy_name), evaluation_step=100_000)

    policy_name = "./Q_test/Q_iter_t_06"
    print(f"{policy_name}")
    player_test(player1=QPlayer(input_filename=policy_name), evaluation_step=100_000)

    policy_name = "./Q_test/Q_iter_tt_08"
    print(f"{policy_name}")
    player_test(player1=QPlayer(input_filename=policy_name), evaluation_step=100_000)

    policy_name = "./Q_test/Q_iter_t_09"
    print(f"{policy_name}")
    player_test(player1=QPlayer(input_filename=policy_name), evaluation_step=100_000) """
