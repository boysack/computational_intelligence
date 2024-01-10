from copy import deepcopy
import random
from random import choice
from game import Game, Move, Player
import numpy as np
import os

# need to compare results: lose < draw < win
MINMAX_RES = {
    -1:  0, # draw
     0:  1, # win
     1: -1  # lose
}

PRUNING_LEVEL = 3

class HumanPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        row = int(input("insert a number from 1 to 5 to select the row -> "))-1
        col = int(input("insert a number from 1 to 5 to select the column -> "))-1
        from_pos = (row, col)
        move = input("insert a string with the direction you want to move the piece (TOP, BOTTOM, LEFT, RIGTH) -> ").upper()
        move = next(value for name, value in vars(Move).items() if name == move)
        return from_pos, move

class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        from_pos = (random.randint(0, 4), random.randint(0, 4))
        move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
        return from_pos, move

class MinMaxPlayer(Player):
    def __init__(self, soft: bool = False) -> None:
        super().__init__()
        self.soft = soft

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        g_copy = deepcopy(game)
        if self.soft:
            move = self.soft_minmax(g_copy)
        else:
            move = self.minmax(g_copy)
        from_pos = (move[0], move[1])
        slide = move[2]
        # print(f"move: {move}")
        return from_pos, slide
        
    def minmax(self, game: Game, level: int = 1, alpha = -np.inf, beta = np.inf) -> tuple[tuple[int, int], Move]:
        # print(f"level: {level}")
        player_id = game.current_player_idx
        available_moves = game.available_moves(player_id)
        #print(f"am: {len(available_moves)} gm: {game.check_winner()} ")

        if player_id == 0: # MAX gioca per primo, adatta anche ad altro
            best = [-1, -1, -1, -np.inf] # MAX player
        else:
            best = [-1, -1, -1, +np.inf] # MIN player

        # return condition
        # prune if not a good path
        # compare alpha and beta value to update
        if len(available_moves) == 0 or game.check_winner() != -1 or level > PRUNING_LEVEL:
            #print(f" - game_over - {MINMAX_RES[game.check_winner()]}")
            """ if level > PRUNING_LEVEL:
                print("prune") """
            return [-1, -1, -1, MINMAX_RES[game.check_winner()]]

        for move in available_moves: # same level nodes
            if alpha > beta:
                # print("ab-prune")
                break

            from_pos = move[0]
            slide = move[1]

            # backup
            if slide == Move.LEFT or slide == Move.RIGHT:
                prev_values = deepcopy(game._board[from_pos[0], :])
            else:
                prev_values = deepcopy(game._board[:, from_pos[1]])

            game.move(from_pos, slide, player_id)
            score = self.minmax(game, level+1, alpha, beta)
            
            # restore
            if slide == Move.LEFT or slide == Move.RIGHT:
                game._board[from_pos[0]] = prev_values
            else:
                game._board[:, from_pos[1]] = prev_values

            score[0] = from_pos[0]
            score[1] = from_pos[1]
            score[2] = slide
            
            if player_id == 0: # MAX - da adattare
                if score[3] > best[3]:
                    best = score  # max value
                    alpha = score[3]
            else:
                if score[3] < best[3]:
                    best = score  # min value
                    beta = score[3]
        return best
    
    def soft_minmax(self, game: Game, level: int = 1, alpha = -np.inf, beta = np.inf) -> tuple[tuple[int, int], Move]:
        # print(f"level: {level}")
        player_id = game.current_player_idx
        available_moves = game.available_moves(player_id)
        #print(f"am: {len(available_moves)} gm: {game.check_winner()} ")

        if player_id == 0: # MAX gioca per primo, adatta anche ad altro
            best = [-1, -1, -1, -np.inf] # MAX player
        else:
            best = [-1, -1, -1, +np.inf] # MIN player

        # return condition
        # prune if not a good path
        # compare alpha and beta value to update
        if len(available_moves) == 0 or game.check_winner() != -1 or level > PRUNING_LEVEL:
            #print(f" - game_over - {MINMAX_RES[game.check_winner()]}")
            """ if level > PRUNING_LEVEL:
                print("prune") """
            return [-1, -1, -1, MINMAX_RES[game.check_winner()]]

        for move in available_moves: # same level nodes
            if alpha >= beta: # <- PRUNE EVEN IF EQUAL
                # print("ab-prune")
                break

            from_pos = move[0]
            slide = move[1]

            # backup
            if slide == Move.LEFT or slide == Move.RIGHT:
                prev_values = deepcopy(game._board[from_pos[0], :])
            else:
                prev_values = deepcopy(game._board[:, from_pos[1]])

            game.move(from_pos, slide, player_id)
            score = self.minmax(game, level+1, alpha, beta)
            
            # restore
            if slide == Move.LEFT or slide == Move.RIGHT:
                game._board[from_pos[0]] = prev_values
            else:
                game._board[:, from_pos[1]] = prev_values

            score[0] = from_pos[0]
            score[1] = from_pos[1]
            score[2] = slide
            
            if player_id == 0: # MAX - da adattare
                if score[3] > best[3]:
                    best = score  # max value
                    alpha = score[3]
            else:
                if score[3] < best[3]:
                    best = score  # min value
                    beta = score[3]
        return best

class QPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        pass

class ESPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: Game) -> tuple[tuple[int, int], Move]:
        pass
    
    def fitness():
        ''' do x games and evaluate the strategy '''
        ''' necessary to have a state: what is a state in this case? '''
        pass
    
    def generate_offspring():
        ''' necessary to have an individual: what is the individual? '''
        pass

if __name__ == '__main__':
    import sys
    if sys.getrecursionlimit() < (PRUNING_LEVEL+9):
        sys.setrecursionlimit(PRUNING_LEVEL+9)
    g = Game()
    g.print()
    player1 = MinMaxPlayer(soft=False)
    player2 = RandomPlayer()
    winner = g.play(player1, player2)
    g.print()
    print(f"Winner: Player {winner}")
