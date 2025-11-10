
import numpy as np
from tabulate import tabulate
from colorama import Fore, Style
import copy

import random

class Game:
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

    directions = {
        "up"    :   (-1, 0),
        "down"  :   (1, 0),
        "left"  :   (0, -1),
        "right" :   (0, 1),
    }

    COLORS = {
        EMPTY: Fore.WHITE  + Style.DIM +  '0' + Style.RESET_ALL,
        WALL: Fore.BLACK + Style.BRIGHT + 'W' + Style.RESET_ALL,
        HEAD: Fore.BLUE + Style.BRIGHT + 'H' + Style.RESET_ALL,
        BODY: Fore.CYAN + Style.BRIGHT + 'S' + Style.RESET_ALL,
        GREEN_APPLE: Fore.GREEN + Style.BRIGHT + 'G' + Style.RESET_ALL,
        RED_APPLE: Fore.RED + Style.BRIGHT + 'R' + Style.RESET_ALL,
    }

    map : np.ndarray
    map_colored : np.ndarray
    width : int
    height : int
    snake : np.ndarray
    green_apples: np.ndarray
    red_apples: np.ndarray
    vision: dict
    vision_colored : dict
    is_over : bool
    
    def ft_reset(self):
        self.ft_initialize_variables()
        self.ft_initialize_map()
        self.ft_get_vision_snake()

    def ft_initialize_variables(self):
        self.map = np.zeros((self.width, self.height), dtype=np.int16)
        self.map_colored = np.zeros((self.width, self.height), dtype=np.int16)
        self.snake = np.empty((0, 2), np.int64)
        self.green_apples = np.empty((0, 2), np.int64)
        self.red_apples = np.empty((0, 2), np.int64)
        self.vision = {}
        self.vision_colored = {}
        self.is_over = False

    def __init__(self, width : int, height : int):
        
        self.height = height + 2
        self.width = width + 2
        
        self.ft_initialize_variables()
        self.ft_initialize_map()
        self.ft_get_vision_snake()

    def ft_put_walls(self):
        self.map[0, :] = self.WALL
        self.map[-1, :] = self.WALL
        self.map[:, 0] = self.WALL
        self.map[:, -1] = self.WALL    

    def ft_initialize_map(self):
        self.ft_put_walls()        
        self.ft_place_initial_elements_into_map()

    def ft_put_body_into_the_map(self, col : int, row: int):
        while(True):
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

    def ft_put_any_element_into_the_map(self, code : int, nbr: int, coord : np.ndarray):
        for i in range(nbr):
            while(True):
                col = random.randrange(1, self.width - 1)
                row = random.randrange(1, self.height - 1)

                if self.map[row, col] == self.EMPTY:
                    self.map[row, col] = code
                    coord = np.vstack([coord, [row, col]])
                    break
                
        return coord  
        
    def ft_put_snake_into_map(self):        
        col_head = random.randrange(1, self.width - 1)
        row_head = random.randrange(1, self.height - 1)
        self.map[row_head,col_head] = self.HEAD
    
        self.snake = np.vstack([self.snake, [row_head, col_head]])

        col_body1, row_body1 = self.ft_put_body_into_the_map(col_head, row_head)
        self.ft_put_body_into_the_map(col_body1, row_body1)

    def ft_place_initial_elements_into_map(self):
        self.ft_put_snake_into_map()
        self.green_apples = self.ft_put_any_element_into_the_map(self.GREEN_APPLE, 2, self.green_apples)
        self.red_apples = self.ft_put_any_element_into_the_map(self.RED_APPLE, 1, self.red_apples)


    def ft_get_vision_with_direction(self, direction : str):
        start = tuple(self.snake[0])
        while(True):

            start = tuple(np.add(start, self.directions[direction]))
            code = self.map[start]
            if code == 1:
                self.vision[direction].append(code)
                break
            else:
                self.vision[direction].append(code)

    def ft_get_vision_snake(self):
        self.vision = {"up": [], "down" : [], "left" : [], "right" : []}
        for key in self.vision.keys():
            self.ft_get_vision_with_direction(key)
        
        return self.vision


    def ft_show_terminal_movements(self, code,  direction : str, terminal : str):
        if terminal == "on":
            print("******* Move on *******")
            print(f"Action taken: {direction}")
            print(f"Game over ?: {self.is_over}")
            print(f"Why?: {self.CHARS[code]}")
            print(f"Last position saved: \n{self.snake}")
            rows = self.ft_compute_vision_colored()
            print(f"Direction: \n{tabulate(rows, headers=['values', 'direction'])}")
            print("***********************")

    def ft_move_snake(self, direction : str, terminal : str):

        code = self.EMPTY
        try :
            code = self.ft_move_snake_detail(direction)
            self.ft_get_vision_snake()

        except Exception as e:
            if len(self.snake) == 0:
                self.ft_show_terminal_movements(code, direction, terminal)
                self.is_over = True

                return self.RED_APPLE, self.is_over
            else:
                raise RuntimeError("Unexpected Error in ft_move_snake") from e

        self.ft_show_terminal_movements(code, direction, terminal)
        
        return code, self.is_over
        
    def ft_empty_case(self, map_pos : tuple, snake : np.ndarray):
        self.map[map_pos] = self.HEAD
        for i, value in enumerate(self.snake):
            self.map[tuple(self.snake[i])] = self.BODY
        self.map[tuple(self.snake[-1])] = self.EMPTY
        self.snake[0] = map_pos
        for i in range(1, len(self.snake)):
            self.snake[i] = snake[i - 1]
    
        self.is_over = False
    
    def ft_green_apple_case(self, map_pos : tuple, snake : np.ndarray):
        for i in range(len(self.snake)):
            self.map[tuple(self.snake[i])] = self.BODY
        self.map[map_pos] = self.HEAD
        
        self.snake[0] = map_pos
        for i in range(1, len(self.snake)):
            self.snake[i] = snake[i - 1]
        self.snake = np.vstack([self.snake, snake[-1]]) 

        index = np.where((self.green_apples == map_pos).all(axis=1))[0]
        self.green_apples = np.delete(self.green_apples, index[0], axis=0)
        self.green_apples = self.ft_put_any_element_into_the_map(self.GREEN_APPLE, \
                                                                    1, self.green_apples)
        self.is_over = False
            
    def ft_red_apple_case(self, map_pos : tuple, snake : np.ndarray):
        for i in range(len(self.snake)):
            self.map[tuple(self.snake[i])] = self.BODY
        self.map[map_pos] = self.HEAD

        # Remove tail after mouvement
        removed_tail = tuple(self.snake[-1])
        self.map[removed_tail] = self.EMPTY

        self.snake[0] = map_pos
        for i in range(1, len(self.snake)):
            self.snake[i] = snake[i - 1]
        
        # Remove tail after eating a red apple            
        if len(self.snake) == 1:
            self.map[map_pos] = self.EMPTY
            self.snake = np.empty((0, 2), dtype=int)
            self.is_over = True
        else:
            removed_tail = tuple(self.snake[-1])
            self.map[removed_tail] = self.EMPTY
            self.snake = np.delete(self.snake, -1, axis=0)
        
        index = np.where((self.red_apples == map_pos).all(axis=1))[0]
        self.red_apples = np.delete(self.red_apples, index[0], axis=0)
        self.red_apples = self.ft_put_any_element_into_the_map(self.RED_APPLE, 1, self.red_apples)
        

    def ft_move_snake_detail(self, direction :  str):
        start = tuple(self.snake[0])
        mouvement = self.directions[direction]
        map_pos = tuple(np.add(start, mouvement))
        code = self.map[map_pos]        

        if code == self.HEAD \
            or code == self.BODY \
            or code == self.WALL:
            self.is_over = True
            return code

        snake = self.snake.copy()
        
        if code == self.EMPTY:
            self.ft_empty_case(map_pos, snake)            
            return code
        elif code == self.GREEN_APPLE:
            self.ft_green_apple_case(map_pos, snake)
            return code
        elif code == self.RED_APPLE:
            self.ft_red_apple_case(map_pos, snake)
            return code
        
        self.is_over = False
        return code   

    def ft_compute_vision_colored(self):
        for direction, values in self.vision.items():
            colored_line = [self.COLORS[v] for v in values]
            self.vision_colored[direction] = colored_line

        rows = [(keys, " ".join(values)) for keys, values in self.vision_colored.items()]
        return rows

    def __str__(self):
        self.map_colored = np.vectorize(lambda v: self.COLORS[v])(self.map)
        rows = self.ft_compute_vision_colored()

        print(f"Visions: \n{self.vision}")
        print(f"Snake coordinates: \n{self.snake} \nGreen apples: \n{self.green_apples} \nRed apples: \n{self.red_apples}")
        return "Current state map\n" + tabulate(self.map_colored) + '\n' + "Snake Vision\n" + tabulate(rows, headers=['values', 'direction'])



def main():
    size = 10
    game = Game(size, size)

    print(f"initial length: {len(game.snake)}")
    print(game)
    is_over = False

    # Testing mouvement green apple + 1 in snake
    print("############## GREEN APPLE TEST #################")
    while(True):
        mouvement = random.choice(['up', 'down', 'left', 'right'])
        len_snake = len(game.snake.copy())
        code, is_over = game.ft_move_snake(mouvement, 'off')
        len_snake2 = len(game.snake)
        
        if is_over == True:
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
    while(True):
        mouvement = random.choice(['up', 'down', 'left', 'right'])
        len_snake = len(game.snake.copy())
        before_game = copy.deepcopy(game)
        code, is_over = game.ft_move_snake(mouvement, 'off')
        len_snake2 = len(game.snake)

        if is_over == True:
            game.ft_reset()
        
        if code == game.RED_APPLE:
            print(f"Length before: {len_snake}")
            print(f"Length after: {len_snake2}")
            
            break

    print(f"### Before ###")
    print(before_game)
    print(f"### After ###")
    print(game)

    print("################ RED APPLE LOSE #################")
    while(True):
        mouvement = random.choice(['up', 'down', 'left', 'right'])
        len_snake = len(game.snake.copy())
        before_game = copy.deepcopy(game)
        code, is_over = game.ft_move_snake(mouvement, 'off')
        len_snake2 = len(game.snake)

        if is_over == True and code == game.RED_APPLE:
            print(f"Length before: {len_snake}")
            print(f"Length after: {len_snake2}")
            break
        elif is_over == True:
            game.ft_reset()
        else:
            pass

    print(f"### Before ###")
    print(before_game)
    print(f"### After ###")
    print(game)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass