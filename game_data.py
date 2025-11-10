#!/usr/bin/env python3

import copy
import random
import numpy as np
from tabulate import tabulate
from colorama import Fore, Style


class Game:
    """
        Game class for the Snake game
        In this class, we will implement the game logic and the game board.
    """

    # -----------------------------------------------------------------------
    # Class-Level Constants

    EMPTY = 0
    WALL = 1
    HEAD = 2
    BODY = 3
    RED_APPLE = 4
    GREEN_APPLE = 5

    CHARS = [
        'EMPTY',
        'WALL',
        'HEAD',
        'BODY',
        'RED_APPLE',
        'GREEN_APPLE',
    ]

    CHARS_CODES = [
        '0',
        'W',
        'H',
        'S',
        'R',
        'G',
    ]

    directions = {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }

    COLORS = {
        EMPTY: Fore.WHITE + Style.DIM + '0' + Style.RESET_ALL,
        WALL: Fore.BLACK + Style.BRIGHT + 'W' + Style.RESET_ALL,
        HEAD: Fore.BLUE + Style.BRIGHT + 'H' + Style.RESET_ALL,
        BODY: Fore.CYAN + Style.BRIGHT + 'S' + Style.RESET_ALL,
        RED_APPLE: Fore.RED + Style.BRIGHT + 'R' + Style.RESET_ALL,
        GREEN_APPLE: Fore.GREEN + Style.BRIGHT + 'G' + Style.RESET_ALL,
    }

    COLORS_INDEX = {
        '0': 0,
        'W': 1,
        'H': 2,
        'S': 3,
        'R': 4,
        'G': 5,
    }

    # -----------------------------------------------------------------------
    # Initialisation

    def ft_reset(self):
        """
        Logic:
        - Reset game state and rebuild the board.

        Return:
        - None
        """
        self.ft_initialize_variables()
        self.ft_initialize_map()
        self.ft_get_vision_snake()

    def ft_initialize_variables(self):
        """
        Logic:
        - Initialise core structures backing the grid and entities.

        Return:
        - None
        """
        self.map = np.zeros((self.width, self.height), dtype=np.int16)
        self.map_colored = np.zeros((self.width, self.height), dtype=np.int16)
        self.snake = np.empty((0, 2), np.int64)
        self.snake_after_death = np.empty((0, 2), np.int64)
        self.green_apples = np.empty((0, 2), np.int64)
        self.red_apples = np.empty((0, 2), np.int64)
        self.vision = {}
        self.vision_colored = {}
        self.vision_extended = []
        self.is_over = False
        self.last_move = ""

    def __init__(self, width: int, height: int):
        """
        Logic:
        - Prepare the board with an outer wall and seeded entities.

        Return:
        - None
        """
        self.height = height + 2
        self.width = width + 2

        self.ft_initialize_variables()
        self.ft_initialize_map()
        self.ft_get_vision_snake()

    # -----------------------------------------------------------------------
    # Board Construction

    def ft_put_walls(self):
        """
        Logic:
        - Surround the playable grid with blocking tiles.

        Return:
        - None
        """
        self.map[0, :] = self.WALL
        self.map[-1, :] = self.WALL
        self.map[:, 0] = self.WALL
        self.map[:, -1] = self.WALL

    def ft_initialize_map(self):
        """
        Logic:
        - Populate the board with walls and starter entities.

        Return:
        - None
        """
        self.ft_put_walls()
        self.ft_place_initial_elements_into_map()

    def ft_put_body_into_the_map(self, col: int, row: int):
        """
        Logic:
        - Extend the snake body from a starting coordinate.

        Return:
        - tuple[int, int]
        """
        while (True):
            col_delta = random.choice([-1, 0, 1])
            col_body = col + col_delta
            row_body = row

            if col_delta == 0:
                row_body = row + random.choice([-1, 1])

            if self.map[row_body, col_body] == self.EMPTY:
                self.map[row_body, col_body] = self.BODY
                self.snake = np.vstack([self.snake, [row_body, col_body]])
                break

        return col_body, row_body

    def ft_put_any_element_into_the_map(
            self, code: int, nbr: int, coord: np.ndarray):
        """
        Logic:
        - Randomly drop `nbr` elements of type `code` onto empty tiles.

        Return:
        - np.ndarray
        """
        for i in range(nbr):
            while (True):
                col = random.randrange(1, self.width - 1)
                row = random.randrange(1, self.height - 1)

                if self.map[row, col] == self.EMPTY:
                    self.map[row, col] = code
                    coord = np.vstack([coord, [row, col]])
                    break

        return coord

    def ft_put_snake_into_map(self):
        """
        Logic:
        - Seed the snake head and initial body segments.

        Return:
        - None
        """
        col_head = random.randrange(1, self.width - 1)
        row_head = random.randrange(1, self.height - 1)
        self.map[row_head, col_head] = self.HEAD

        self.snake = np.vstack([self.snake, [row_head, col_head]])

        col_body1, row_body1 = self.ft_put_body_into_the_map(
            col_head, row_head)
        self.ft_put_body_into_the_map(col_body1, row_body1)

    # -----------------------------------------------------------------------
    # Entity Placement

    def ft_place_initial_elements_into_map(self):
        """
        Logic:
        - Place the snake along with initial green and red apples.

        Return:
        - None
        """
        self.ft_put_snake_into_map()
        self.green_apples = self.ft_put_any_element_into_the_map(
            self.GREEN_APPLE, 2, self.green_apples)
        self.red_apples = self.ft_put_any_element_into_the_map(
            self.RED_APPLE, 1, self.red_apples)

    # -----------------------------------------------------------------------
    # Vision Computation

    def ft_get_vision_with_direction(self):
        """
        Logic:
        - Collect the immediate surroundings of the snake head.

        Return:
        - None
        """
        if len(self.snake) == 0:
            start = tuple(self.snake_after_death[0])
        else:
            start = tuple(self.snake[0])

        for key, value in self.directions.items():
            position = tuple(np.add(start, value))
            code = self.map[position]
            self.vision += self.CHARS_CODES[code]

    def ft_get_vision_snake(self):
        """
        Logic:
        - Build the four-neighbour encoding around the head.

        Return:
        - str
        """
        self.vision = ""
        self.ft_get_vision_with_direction()
        return self.vision

    def ft_get_vision_with_direction_extended(self):
        """
        Logic:
        - Capture directional rays until a wall is hit.

        Return:
        - None
        """
        if len(self.snake) == 0:
            start = tuple(self.snake_after_death[0])
        else:
            start = tuple(self.snake[0])

        vision_direction = ""

        for key, value in self.directions.items():
            vision_direction = ""
            position = start
            for i in range(0, self.width):
                position = tuple(np.add(position, value))
                position_code = self.map[position]
                vision_direction += self.CHARS_CODES[position_code]
                if position_code == self.WALL:
                    break

            self.vision_extended.append(vision_direction)

    def ft_get_vision_snake_extended(self):
        """
        Logic:
        - Expose the extended directional vision payload.

        Return:
        - list[str]
        """
        self.vision_extended = []
        self.ft_get_vision_with_direction_extended()
        return self.vision_extended

    # -----------------------------------------------------------------------
    # Terminal Visualisation

    def ft_show_terminal_movements(self, code, direction: str, terminal: str):
        """
        Logic:
        - Print a verbose snapshot of the last move when enabled.

        Return:
        - None
        """
        if terminal == "on":
            self.ft_get_vision_snake_extended()
            print("******* Move on *******")
            print(f"Action taken: {direction}")
            print(f"Game over ?: {self.is_over}")
            print(f"Why?: {self.CHARS[code]}")
            if len(self.snake) > 0:
                print(f"Last head position: \n{tuple(self.snake[0])}")
            else:
                print(
                    f"Last head position: \n"
                    f"{tuple(self.snake_after_death[0])}")
            rows = self.ft_compute_vision_colored_extended()
            print(
                f"Direction: \n"
                f"{tabulate(rows, headers=['direction', 'vision'])}")
            print("***********************")

    # -----------------------------------------------------------------------
    # Movement Resolution

    def ft_empty_case(self, map_pos: tuple, snake: np.ndarray):
        """
        Logic:
        - Advance the snake into an empty cell and move the tail forward.

        Return:
        - None
        """
        for i in range(len(self.snake)):
            self.map[tuple(self.snake[i])] = self.BODY

        self.map[map_pos] = self.HEAD

        self.snake[0] = map_pos
        for i in range(1, len(self.snake)):
            self.snake[i] = snake[i - 1]

        old_tail = tuple(snake[-1])
        self.map[old_tail] = self.EMPTY

        self.is_over = False

    def ft_green_apple_case(self, map_pos: tuple, snake: np.ndarray):
        """
        Logic:
        - Grow the snake when a green apple is collected.

        Return:
        - None
        """
        for i in range(len(self.snake)):
            self.map[tuple(self.snake[i])] = self.BODY
        self.map[map_pos] = self.HEAD

        self.snake[0] = map_pos
        for i in range(1, len(self.snake)):
            self.snake[i] = snake[i - 1]

        self.snake = np.vstack([self.snake, snake[-1]])

        self.green_apples = self.ft_put_any_element_into_the_map(
            self.GREEN_APPLE, 1, self.green_apples)

        self.is_over = False

    def ft_red_apple_case(self, map_pos: tuple, snake: np.ndarray):
        """
        Logic:
        - Handle red apple encounters, shortening the snake.

        Return:
        - None
        """
        L = len(self.snake)

        for i in range(L):
            self.map[tuple(self.snake[i])] = self.BODY
        self.map[map_pos] = self.HEAD

        self.snake[0] = map_pos
        for i in range(1, L):
            self.snake[i] = snake[i - 1]

        old_tail = tuple(snake[-1])
        self.map[old_tail] = self.EMPTY

        if len(self.snake) == 0:
            self.is_over = True
            return
        extra_tail = tuple(self.snake[-1])
        self.map[extra_tail] = self.EMPTY
        self.snake_after_death = self.snake.copy()
        self.snake = np.delete(self.snake, -1, axis=0)

        if len(self.snake) == 0:
            self.is_over = True
            return

        self.red_apples = self.ft_put_any_element_into_the_map(
            self.RED_APPLE, 1, self.red_apples)
        self.is_over = False

    def ft_move_snake_detail(self, direction: str):
        """
        Logic:
        - Resolve a single snake action and update the board state.

        Return:
        - int
        """
        start = tuple(self.snake[0])
        dir_rows, dir_cols = self.directions[direction]
        map_pos = (start[0] + dir_rows, start[1] + dir_cols)
        tile = self.map[map_pos]

        if map_pos == tuple(self.snake[-1]):
            self.is_over = True
            return tile

        tail = tuple(self.snake[-1])
        tail_pass = (tile == self.BODY and map_pos == tail)

        if tile == self.WALL or tile == self.HEAD or (
                tile == self.BODY and not tail_pass):
            self.is_over = True
            return tile

        snake_prev = self.snake.copy()
        if tile == self.EMPTY or tail_pass:
            self.ft_empty_case(map_pos, snake_prev)
        elif tile == self.GREEN_APPLE:
            self.ft_green_apple_case(map_pos, snake_prev)
        elif tile == self.RED_APPLE:
            self.ft_red_apple_case(map_pos, snake_prev)
        else:
            self.is_over = False

        if len(self.snake) == 1:
            self.is_over = True
            return tile

        return tile

    def ft_move_snake(self, direction: str, terminal: str):
        """
        Logic:
        - Execute a move command and surface the resulting transition.

        Return:
        - tuple[int, bool]
        """
        code = self.ft_move_snake_detail(direction)
        self.last_move = direction
        self.ft_get_vision_snake()
        self.ft_show_terminal_movements(code, direction, terminal)
        return code, self.is_over

    # -----------------------------------------------------------------------
    # Presentation Helpers

    def ft_compute_vision_colored(self):
        """
        Logic:
        - Build the coloured representation of the immediate vision.

        Return:
        - list[tuple[str, str]]
        """
        colored_line = [
            self.COLORS.get(
                self.COLORS_INDEX[v],
                v) for v in self.vision]
        rows = [("Snake Vision", " ".join(colored_line))]
        return rows

    def ft_compute_vision_colored_extended(self):
        """
        Logic:
        - Return formatted extended vision rows for terminal printing.

        Return:
        - list[tuple[str, str]]
        """
        directions = ['up', 'down', 'left', 'right']
        rows = []
        for i, vision_str in enumerate(self.vision_extended):
            colored_line = [
                self.COLORS.get(
                    self.COLORS_INDEX[v],
                    v) for v in vision_str]
            rows.append((directions[i], " ".join(colored_line)))
        return rows

    def __str__(self):
        """
        Logic:
        - Represent the board as a coloured table for debugging.

        Return:
        - str
        """
        self.map_colored = np.vectorize(lambda v: self.COLORS[v])(self.map)
        rows = self.ft_compute_vision_colored()

        print(f"Visions: \n{self.vision}")
        print(
            f"Snake coordinates: \n"
            f"{self.snake} \nGreen apples: \n{self.green_apples} \n"
            f"Red apples: \n{self.red_apples}")

        headers = ['values', 'direction (up, down, left, right)']
        string = "Current state map\n" \
            f"{tabulate(self.map_colored)} \n" \
            f"Snake Vision\n" \
            f"{tabulate(rows, headers=headers)}"
        return string

    # -----------------------------------------------------------------------
    # Type Hints

    map: np.ndarray
    map_colored: np.ndarray
    width: int
    height: int
    snake: np.ndarray
    snake_after_death: np.ndarray
    green_apples: np.ndarray
    red_apples: np.ndarray
    vision: dict
    vision_extended: list
    last_vision: np.ndarray
    last_move: str
    vision_colored: dict
    is_over: bool


def main():
    size = 10
    game = Game(size, size)

    print(f"initial length: {len(game.snake)}")
    print(game)
    is_over = False

    print("############## GREEN APPLE TEST #################")
    while (True):
        mouvement = random.choice(['up', 'down', 'left', 'right'])
        len_snake = len(game.snake.copy())
        code, is_over = game.ft_move_snake(mouvement, 'off')
        len_snake2 = len(game.snake)

        if is_over:
            game.ft_reset()

        if code == game.GREEN_APPLE:
            print(f"Length before: {len_snake}")
            print(f"Length after: {len_snake2}")
            break

    print(game)
    print(game.snake)
    print(game.green_apples)

    before_game = None

    print("################ RED APPLE TEST #################")
    while (True):
        mouvement = random.choice(['up', 'down', 'left', 'right'])
        len_snake = len(game.snake.copy())
        before_game = copy.deepcopy(game)
        code, is_over = game.ft_move_snake(mouvement, 'off')
        len_snake2 = len(game.snake)

        if is_over:
            game.ft_reset()

        if code == game.RED_APPLE:
            print(f"Length before: {len_snake}")
            print(f"Length after: {len_snake2}")

            break

    print("### Before ###")
    print(before_game)
    print("### After ###")
    print(game)

    print("################ RED APPLE LOSE #################")
    while (True):
        mouvement = random.choice(['up', 'down', 'left', 'right'])
        len_snake = len(game.snake.copy())
        before_game = copy.deepcopy(game)
        code, is_over = game.ft_move_snake(mouvement, 'off')
        len_snake2 = len(game.snake)

        if is_over and code == game.RED_APPLE:
            print(f"Length before: {len_snake}")
            print(f"Length after: {len_snake2}")
            break
        elif is_over:
            game.ft_reset()
        else:
            pass

    print("### Before ###")
    print(before_game)
    print("### After ###")
    print(game)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
