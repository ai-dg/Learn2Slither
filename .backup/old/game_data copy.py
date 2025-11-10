
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
        self._enforce_apple_invariants()
        self._ensure_single_head()
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
        
    def ft_empty_case(self, map_pos: tuple, snake: np.ndarray):
        # 1) tout l'ancien snake en BODY
        for i in range(len(self.snake)):
            self.map[tuple(self.snake[i])] = self.BODY

        # 2) nouvelle tête
        self.map[map_pos] = self.HEAD

        # 3) décalage du corps
        self.snake[0] = map_pos
        for i in range(1, len(self.snake)):
            self.snake[i] = snake[i - 1]

        # 4) vider l'ANCIENNE queue (dans 'snake', pas dans self.snake modifié)
        old_tail = tuple(snake[-1])
        self.map[old_tail] = self.EMPTY

        # ⚠️ pas de np.delete ici : longueur conserve
        self.is_over = False

    
    def ft_green_apple_case(self, map_pos: tuple, snake: np.ndarray):
        # 1) marquer ancien snake en BODY, poser HEAD
        for i in range(len(self.snake)):
            self.map[tuple(self.snake[i])] = self.BODY
        self.map[map_pos] = self.HEAD

        # 2) décaler le corps (pas de pop de queue)
        self.snake[0] = map_pos
        for i in range(1, len(self.snake)):
            self.snake[i] = snake[i - 1]

        # 3) grandir : duplique l'ancienne queue
        self.snake = np.vstack([self.snake, snake[-1]])

        # 4) respawn 1 verte (celle mangée est écrasée par HEAD)
        self.green_apples = self.ft_put_any_element_into_the_map(self.GREEN_APPLE, 1, self.green_apples)

        self.is_over = False


            
    def ft_red_apple_case(self, map_pos: tuple, snake: np.ndarray):
        L = len(self.snake)

        # 1) ancien snake en BODY, poser HEAD
        for i in range(L):
            self.map[tuple(self.snake[i])] = self.BODY
        self.map[map_pos] = self.HEAD

        # 2) décalage corps (comme empty)
        self.snake[0] = map_pos
        for i in range(1, L):
            self.snake[i] = snake[i - 1]

        # 3) vider l'ANCIENNE queue (comme empty)
        old_tail = tuple(snake[-1])
        self.map[old_tail] = self.EMPTY

        # 4) effet rouge : retirer 1 segment supplémentaire
        #    => vider la "nouvelle" queue (après décalage) ET raccourcir le tableau de 1
        if len(self.snake) == 0:
            self.is_over = True
            return
        extra_tail = tuple(self.snake[-1])
        self.map[extra_tail] = self.EMPTY
        self.snake = np.delete(self.snake, -1, axis=0)  # longueur L-1

        if len(self.snake) == 0:
            self.is_over = True
            return

        # 5) respawn rouge (le compte exact sera garanti par _enforce_apple_invariants)
        self.red_apples = self.ft_put_any_element_into_the_map(self.RED_APPLE, 1, self.red_apples)
        self.is_over = False


    
    def _sync_apples_from_map(self):
        """Recalcule les positions des pommes depuis self.map (au cas où la map ait été modifiée directement)."""
        self.green_apples = np.argwhere(self.map == self.GREEN_APPLE)
        self.red_apples   = np.argwhere(self.map == self.RED_APPLE)

    def _enforce_apple_invariants(self):
        """Garantit: exactement 2 vertes et 1 rouge sur des cases vides."""
        # Resync d’abord
        self._sync_apples_from_map()

        # Trop de vertes -> on en retire jusqu'à 2
        while len(self.green_apples) > 2:
            r, c = tuple(self.green_apples[-1])  # retire la dernière trouvée
            self.map[r, c] = self.EMPTY
            self._sync_apples_from_map()

        # Pas assez de vertes -> on en ajoute
        while len(self.green_apples) < 2:
            self.green_apples = self.ft_put_any_element_into_the_map(self.GREEN_APPLE, 1, self.green_apples)
            self._sync_apples_from_map()

        # Rouge: max 1
        while len(self.red_apples) > 1:
            r, c = tuple(self.red_apples[-1])
            self.map[r, c] = self.EMPTY
            self._sync_apples_from_map()

        while len(self.red_apples) < 1:
            self.red_apples = self.ft_put_any_element_into_the_map(self.RED_APPLE, 1, self.red_apples)
            self._sync_apples_from_map()

    def _ensure_single_head(self):
        """Force exactement 1 HEAD : celle de self.snake[0]."""
        if len(self.snake) == 0:
            return
        wanted = tuple(self.snake[0])

        # 1) Enlever toutes les têtes existantes (s'il y en a 0/1/n peu importe)
        heads = np.argwhere(self.map == self.HEAD)
        for r, c in heads:
            self.map[r, c] = self.BODY

        # 2) S'assurer que la case voulue est bien la tête
        self.map[wanted] = self.HEAD


    def ft_move_snake_detail(self, direction: str):
        start = tuple(self.snake[0])
        dr, dc = self.directions[direction]
        map_pos = (start[0] + dr, start[1] + dc)
        tile = self.map[map_pos]

        # --- TAIL PASS ---
        # Si on entre sur l'ANCIENNE queue et que le move est "à vide" (pas de verte),
        # on doit autoriser (car la queue sera pop).
        tail = tuple(self.snake[-1])
        tail_pass = (tile == self.BODY and map_pos == tail)

        # Collisions réelles (HEAD/BODY sauf tail-pass, ou WALL)
        if tile == self.WALL or tile == self.HEAD or (tile == self.BODY and not tail_pass):
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
            # fallback
            self.is_over = False

        # Resynchronise et réimpose les invariants (important si la map a été modifiée "à la main")
        # ... après l'appel à ft_empty_case / ft_green_apple_case / ft_red_apple_case
        self._enforce_apple_invariants()
        if not self.is_over:
            self._ensure_single_head()
        return tile

  

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