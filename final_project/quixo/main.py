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

PRUNING_LEVEL = 3

class HumanPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'MyGame') -> tuple[tuple[int, int], Move]:
        col = int(input("insert a number from 1 to 5 to select the column -> "))-1
        row = int(input("insert a number from 1 to 5 to select the row -> "))-1
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
    def __init__(self, soft: bool = False, player: int = 0) -> None:
        super().__init__()
        self.soft = soft
        player = player % 2
        self.player = player

    def make_move(self, game: 'MyGame') -> tuple[tuple[int, int], Move]:
        g_copy = deepcopy(game)
        # try to implement something to get equally rewarded moves to be chosen randomly, and not just taken the first
        move = self.minmax(g_copy)
        from_pos = (move[0], move[1])
        slide = move[2]
        return from_pos, slide

    def minmax(self, game: 'MyGame', level: int = 1, alpha = -np.inf, beta = np.inf) -> tuple[tuple[int, int], Move]:
        player_id = game.current_player_idx
        available_moves = game.available_moves(player_id)

        if player_id == self.player: # my player plays always as MAX
            best = [-1, -1, -1, -np.inf]
        else:
            best = [-1, -1, -1, +np.inf]

        if len(available_moves) == 0 or game.check_winner() != -1 or level > PRUNING_LEVEL:
            return [-1, -1, -1, COMP_RES[self.player][game.check_winner()]]

        for move in available_moves: # same level nodes
            from_pos = move[0]
            slide = move[1]

            # backup - save row/column
            if slide == Move.LEFT or slide == Move.RIGHT:
                prev_values = deepcopy(game._board[from_pos[1], :])
            else:
                prev_values = deepcopy(game._board[:, from_pos[0]])

            game.move(from_pos, slide, player_id)
            score = self.minmax(game, level+1, alpha, beta)

            # restore - restore row/column
            if slide == Move.LEFT or slide == Move.RIGHT:
                game._board[from_pos[1]] = prev_values
            else:
                game._board[:, from_pos[0]] = prev_values

            score[0] = from_pos[0]
            score[1] = from_pos[1]
            score[2] = slide
            
            if player_id == self.player: # my player plays always as MAX
                if score[3] > best[3]:
                    best = score  # max value
                    alpha = score[3]
            else:
                if score[3] < best[3]:
                    best = score  # min value
                    beta = score[3]

            if self.soft:
                if alpha >= beta: # <- PRUNE EVEN IF EQUAL
                    break
            else:
                if alpha > beta:
                    break
        return best

class QPlayer(Player):
    def __init__(self, alpha = .5, epsilon = 1.0, final_epsilon = 0.0, gamma = .8, input_filename = "Q_quixo", output_filename = "Q_quixo", mode: str = "val", player: int = 0, iterations: int = 1000) -> None:
        super().__init__()
        if os.path.isfile(input_filename):
            with open(input_filename, "rb") as f:
                self.Q = pickle.load(f)
        else:
            self.Q = {}
        self.alpha = alpha
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
        print("training QLearning agent...")
        e = np.linspace(self.epsilon, self.final_epsilon, self.iterations)
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
            self.epsilon *= e[i]
        self.save_Q()
        self.mode = "val"

    def train(self, second_player: Player = RandomPlayer()):
        print("training QLearning agent...")
        self.mode = "train"
        for _ in tqdm(range(self.iterations)):
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
            
            self.epsilon *= self.rand_dec_rate
        self.save_Q()
        self.mode = "val"

class ESPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'MyGame') -> tuple[tuple[int, int], Move]:
        pass
    
    def fitness():
        ''' do x games and evaluate the strategy '''
        ''' necessary to have a state: what is a state in this case? '''
        pass
    
    def generate_offspring():
        ''' necessary to have an individual: what is the individual? '''
        pass

class MyGame(Game):
    def __init__(self):
        super(MyGame, self).__init__()

    # to call the move externally on a game deepcopy
    def move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        ok = self._Game__move((from_pos[1], from_pos[0]), slide, player_id)
        if ok:
            self.current_player_idx = (self.current_player_idx+1)%2
        return ok

    # receive (column, row) moves
    def check_move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        '''Perform a move'''
        if player_id > 2:
            return False
        acceptable = self.__check_take((from_pos[1], from_pos[0]), player_id) and self.__check_slide((from_pos[1], from_pos[0]), slide)
        return acceptable
    
    def __check_take(self, from_pos: tuple[int, int], player_id: int) -> bool:
        '''Take piece'''
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
        '''Slide the other pieces'''
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
        for y in range(4):
            from_pos = (y, 0)
            for slide in Move:
                # check move invert order in (row, col) -> ok
                if self.check_move(from_pos, slide, player_idx):
                    # so appended from_pos is in the form (col, row) -> ok
                    a_m.append((from_pos, slide))
            from_pos = (y, 4)
            for slide in Move:
                if self.check_move(from_pos, slide, player_idx):
                    a_m.append((from_pos, slide))
        for x in range(4):
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
        self.current_player_idx+=1
        self.current_player_idx%=2

def player_test(pov: int = 0, player1: Player = RandomPlayer(), player2: Player = RandomPlayer(), evaluation_step: int = 1_000):
    wins = 0
    ties = 0
    for _ in tqdm(range(evaluation_step)):
        g = MyGame()
        winner = g.play(player1, player2)
        if COMP_RES[pov][winner]==1:
            wins += 1
        elif COMP_RES[pov][winner]==0:
            ties += 1
    print(f"wins: {wins} | ties: {ties} | losses: {evaluation_step-wins-ties}")
    print(f"win rate: {wins/evaluation_step:.2%}")

if __name__ == '__main__':
    # recursion limit setting for MinMax
    import sys
    if sys.getrecursionlimit() < (PRUNING_LEVEL+9):
        sys.setrecursionlimit(PRUNING_LEVEL+9)
    
    # evaluation games number setting
    if len(sys.argv) == 1:
        total = 1_000
    else:
        total = int(sys.argv[1])

    print("-------- MinMax --------")
    
    # player_test(player1=MinMaxPlayer(), evaluation_step=100)

    print("------- QLearning ------")
    qplayer = QPlayer(input_filename="")
    qplayer.train_test()
    #qplayer.train()

    player_test(player1=qplayer)