# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
# from tkinter import Frame, Label, CENTER
# import random
# import logic
# import constants as c
# import time

# def build_model(input_shape, num_actions):
#     model = Sequential([
#         Flatten(input_shape=input_shape),
#         Dense(128, activation='relu'),
#         Dense(128, activation='relu'),
#         Dense(num_actions, activation='linear')
#     ])
#     model.compile(optimizer='adam', loss='mse')
#     return model

# def matrix_to_input(matrix):
#     return np.expand_dims(matrix, axis=(0, -1))

# def evaluate_board(matrix):
#     # Heuristic evaluation based on the number of empty cells and monotonicity
#     empty_cells = len([(i, j) for i in range(4) for j in range(4) if matrix[i][j] == 0])

#     monotonicity_score = 0
#     for i in range(4):
#         for j in range(3):
#             if matrix[i][j] >= matrix[i][j+1]:
#                 monotonicity_score += 1
#             if matrix[j][i] >= matrix[j+1][i]:
#                 monotonicity_score += 1

#     return empty_cells + monotonicity_score

# class GameGrid(Frame):
#     def __init__(self, model):
#         Frame.__init__(self)
#         self.model = model
#         self.grid()
#         self.master.title('2048')
#         self.commands = {c.KEY_UP: logic.up, c.KEY_DOWN: logic.down, c.KEY_LEFT: logic.left, c.KEY_RIGHT: logic.right}
#         self.grid_cells = []
#         self.init_grid()
#         self.matrix = logic.new_game(c.GRID_LEN)
#         self.history_matrices = []
#         self.update_grid_cells()
#         self.perform_ai_move()
        

#     def init_grid(self):
#         background = Frame(self, bg=c.BACKGROUND_COLOR_GAME, width=c.SIZE, height=c.SIZE)
#         background.grid()
#         for i in range(c.GRID_LEN):
#             grid_row = []
#             for j in range(c.GRID_LEN):
#                 cell = Frame(background, bg=c.BACKGROUND_COLOR_CELL_EMPTY, width=c.SIZE / c.GRID_LEN, height=c.SIZE / c.GRID_LEN)
#                 cell.grid(row=i, column=j, padx=c.GRID_PADDING, pady=c.GRID_PADDING)
#                 t = Label(master=cell, text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=c.FONT, width=5, height=2)
#                 t.grid()
#                 grid_row.append(t)
#             self.grid_cells.append(grid_row)

#     def update_grid_cells(self):
#         for i in range(c.GRID_LEN):
#             for j in range(c.GRID_LEN):
#                 new_number = self.matrix[i][j]
#                 self.grid_cells[i][j].configure(text="" if new_number == 0 else str(new_number),
#                                                 bg=c.BACKGROUND_COLOR_DICT.get(new_number, c.BACKGROUND_COLOR_CELL_EMPTY))
#         self.update()

#     # def perform_ai_move(self):
#     #     move = self.monte_carlo_move()
#     #     if move in self.commands:
#     #         self.matrix, changed = self.commands[move](self.matrix)
#     #         if changed:
#     #             self.matrix = logic.add_two(self.matrix)
#     #             self.history_matrices.append([row[:] for row in self.matrix])
#     #             self.update_grid_cells()
#     #             if logic.game_state(self.matrix) == 'win':
#     #                 self.grid_cells[1][1].configure(text="You Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
#     #                 self.grid_cells[1][2].configure(text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
#     #             elif logic.game_state(self.matrix) == 'lose':
#     #                 self.grid_cells[1][1].configure(text="Game Over", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
#     #                 self.grid_cells[1][2].configure(text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
#     #             else:
#     #                 self.after(100, self.perform_ai_move)  # Continue performing AI moves
#     #         else:
#     #             print("Move was not effective, trying another move.")
#     #             self.after(100, self.perform_ai_move)  # Continue performing AI moves if move was not effective
#     #     else:
#     #         print("Invalid move selected.")
#     def perform_ai_move(self):
#         while True:
#             move = self.monte_carlo_move()
#             if move in self.commands:
#                 self.matrix, changed = self.commands[move](self.matrix)
#                 if changed:
#                     self.matrix = logic.add_two(self.matrix)
#                     self.history_matrices.append([row[:] for row in self.matrix])
#                     self.update_grid_cells()
#                     game_state = logic.game_state(self.matrix)
#                     if game_state == 'win':
#                         self.grid_cells[1][1].configure(text="You Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
#                         self.grid_cells[1][2].configure(text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
#                         break  # Break the loop if game is won
#                     elif game_state == 'lose':
#                         self.grid_cells[1][1].configure(text="Game Over", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
#                         self.grid_cells[1][2].configure(text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
#                         break  # Break the loop if game is lost
#                 else:
#                     print("Move was not effective, trying another move.")
#             else:
#                 print("Invalid move selected.")


#     def monte_carlo_move(self):
#         num_simulations = 100
#         move_scores = {move: 0 for move in self.commands.keys()}
        
#         print("Starting Monte Carlo simulations...")
#         for move in self.commands.keys():
#             print(f"Testing move: {move}")
#             for _ in range(num_simulations):
#                 simulated_matrix, done = self.commands[move](self.matrix)
#                 if done and simulated_matrix is not None:
#                     score = self.simulate_playout(simulated_matrix)
#                     move_scores[move] += score
#                     print(f"Move {move}, Score after simulation: {score}")
#                 else:
#                     print(f"Move {move} did not change the matrix.")

#         best_move = max(move_scores, key=move_scores.get)
#         print(f"Best move selected: {best_move} with score: {move_scores[best_move]}")
#         return best_move

#     def simulate_playout(self, matrix, max_depth=100):
#         current_matrix = np.copy(matrix)
#         depth = 0
#         while not logic.is_game_over(current_matrix) and depth < max_depth:
#             move_values = {}
#             for move, command in self.commands.items():
#                 new_matrix, done = command(current_matrix)
#                 if done:
#                     heuristic_value = evaluate_board(new_matrix)
#                     move_values[move] = heuristic_value
#                     print(f"Simulating {move}: heuristic value {heuristic_value}")
#                     current_matrix = new_matrix
#                 else:
#                     print(f"Simulating {move}: NOT effective")
#             if not move_values:
#                 break
#             depth += 1
#         return max(move_values.values())

# if __name__ == '__main__':
#     model = build_model((c.GRID_LEN, c.GRID_LEN, 1), 4)
#     game_grid = GameGrid(model)

import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
from tkinter import Frame, Label, CENTER
import random
import logic
import constants as c
import time

# def build_model(input_shape, num_actions):
#     model = Sequential([
#         Flatten(input_shape=input_shape),
#         Dense(128, activation='relu'),
#         Dense(128, activation='relu'),
#         Dense(num_actions, activation='linear')
#     ])
#     model.compile(optimizer='adam', loss='mse')
#     return model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

def build_q_network(input_shape, num_actions):
    inputs = Input(shape=input_shape)
    layer1 = Dense(128, activation='relu')(inputs)
    layer2 = Dense(128, activation='relu')(layer1)
    actions = Dense(num_actions, activation='linear')(layer2)

    return Model(inputs=inputs, outputs=actions)


def matrix_to_input(matrix):
    return np.expand_dims(matrix, axis=(0, -1))

# def evaluate_board(matrix):
#     # Heuristic evaluation based on the number of empty cells and monotonicity
#     empty_cells = len([(i, j) for i in range(4) for j in range(4) if matrix[i][j] == 0])

#     monotonicity_score = 0
#     for i in range(4):
#         for j in range(3):
#             if matrix[i][j] >= matrix[i][j+1]:
#                 monotonicity_score += 1
#             if matrix[j][i] >= matrix[j+1][i]:
#                 monotonicity_score += 1

#     return empty_cells + monotonicity_score

def evaluate_board(matrix):
    empty_cells = len([(i, j) for i in range(4) for j in range(4) if matrix[i][j] == 0])
    monotonicity_score = 0
    smoothness = 0
    max_tile_weight = 0
    # Weight matrix to prioritize the top-left corner
    weights = [[15, 14, 13, 12], [9, 10, 11, 8], [5, 6, 7, 4], [0, 1, 2, 3]]

    max_tile = 0
    max_tile_position = (0, 0)

    for i in range(4):
        for j in range(4):
            if matrix[i][j] > max_tile:
                max_tile = matrix[i][j]
                max_tile_position = (i, j)

            if j+1 < 4 and matrix[i][j] > 0 and matrix[i][j+1] > 0:
                monotonicity_score += 1 if matrix[i][j] >= matrix[i][j+1] else 0
                smoothness -= abs(matrix[i][j] - matrix[i][j+1])
            if i+1 < 4 and matrix[i][j] > 0 and matrix[i+1][j] > 0:
                monotonicity_score += 1 if matrix[i][j] >= matrix[i+1][j] else 0
                smoothness -= abs(matrix[i][j] - matrix[i+1][j])

            max_tile_weight += matrix[i][j] * weights[i][j]

    # Bonus points if the max tile is in the top-left corner
    corner_bonus = 0
    if max_tile_position == (0, 0):
        corner_bonus = max_tile

    return empty_cells + monotonicity_score + max_tile_weight + smoothness + corner_bonus



class GameGrid(Frame):
    def __init__(self, model):
        Frame.__init__(self)
        self.model = model
        self.grid()
        self.master.title('2048')
        self.commands = {c.KEY_UP: logic.up, c.KEY_DOWN: logic.down, c.KEY_LEFT: logic.left, c.KEY_RIGHT: logic.right}
        self.grid_cells = []
        self.init_grid()
        self.matrix = logic.new_game(c.GRID_LEN)
        self.history_matrices = []
        self.update_grid_cells()
        self.start_time = time.time()  # Record start time
        self.perform_ai_move()

    def init_grid(self):
        background = Frame(self, bg=c.BACKGROUND_COLOR_GAME, width=c.SIZE, height=c.SIZE)
        background.grid()
        for i in range(c.GRID_LEN):
            grid_row = []
            for j in range(c.GRID_LEN):
                cell = Frame(background, bg=c.BACKGROUND_COLOR_CELL_EMPTY, width=c.SIZE / c.GRID_LEN, height=c.SIZE / c.GRID_LEN)
                cell.grid(row=i, column=j, padx=c.GRID_PADDING, pady=c.GRID_PADDING)
                t = Label(master=cell, text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY, justify=CENTER, font=c.FONT, width=5, height=2)
                t.grid()
                grid_row.append(t)
            self.grid_cells.append(grid_row)

    def update_grid_cells(self):
        for i in range(c.GRID_LEN):
            for j in range(c.GRID_LEN):
                new_number = self.matrix[i][j]
                self.grid_cells[i][j].configure(text="" if new_number == 0 else str(new_number),
                                                bg=c.BACKGROUND_COLOR_DICT.get(new_number, c.BACKGROUND_COLOR_CELL_EMPTY))
        self.update()

    def perform_ai_move(self):
        while True:
            move = self.monte_carlo_move()
            if move in self.commands:
                self.matrix, changed = self.commands[move](self.matrix)
                if changed:
                    self.matrix = logic.add_two(self.matrix)
                    self.history_matrices.append([row[:] for row in self.matrix])
                    self.update_grid_cells()
                    game_state = logic.game_state(self.matrix)
                    if game_state == 'win':
                        self.grid_cells[1][1].configure(text="You Win!", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                        self.grid_cells[1][2].configure(text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                        self.end_game()
                        break  # Break the loop if game is won
                    elif game_state == 'lose':
                        self.grid_cells[1][1].configure(text="Game Over", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                        self.grid_cells[1][2].configure(text="", bg=c.BACKGROUND_COLOR_CELL_EMPTY)
                        self.end_game()
                        break  # Break the loop if game is lost
                else:
                    print("Move was not effective, trying another move.")
            else:
                print("Invalid move selected.")

    def end_game(self):
        end_time = time.time()  # Record end time
        elapsed_time = end_time - self.start_time
        print(f"Game ended. Elapsed time: {elapsed_time} seconds")
        highest_tile = np.max(self.matrix)
        print(f"Highest tile on the board: {highest_tile}")

    def monte_carlo_move(self):
        num_simulations = 100
        move_scores = {move: 0 for move in self.commands.keys()}
        
        print("Starting Monte Carlo simulations...")
        for move in self.commands.keys():
            print(f"Testing move: {move}")
            for _ in range(num_simulations):
                simulated_matrix, done = self.commands[move](self.matrix)
                if done and simulated_matrix is not None:
                    score = self.simulate_playout(simulated_matrix)
                    move_scores[move] += score
                    print(f"Move {move}, Score after simulation: {score}")
                else:
                    print(f"Move {move} did not change the matrix.")

        best_move = max(move_scores, key=move_scores.get)
        print(f"Best move selected: {best_move} with score: {move_scores[best_move]}")
        return best_move

    def simulate_playout(self, matrix, max_depth=100):
        current_matrix = np.copy(matrix)
        depth = 0
        while not logic.is_game_over(current_matrix) and depth < max_depth:
            move_values = {}
            for move, command in self.commands.items():
                new_matrix, done = command(current_matrix)
                if done:
                    heuristic_value = evaluate_board(new_matrix)
                    move_values[move] = heuristic_value
                    print(f"Simulating {move}: heuristic value {heuristic_value}")
                    current_matrix = new_matrix
                else:
                    print(f"Simulating {move}: NOT effective")
            if not move_values:
                break
            depth += 1
        return max(move_values.values())

if __name__ == '__main__':
    model = build_q_network((c.GRID_LEN, c.GRID_LEN, 1), 4)
    game_grid = GameGrid(model)
