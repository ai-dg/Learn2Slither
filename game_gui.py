#!/usr/bin/env python3

import sys
from typing import Any, Dict, Tuple

import pygame
from pygame.color import THECOLORS

from game_data import Game


class GameGUI:
    """
        GameGUI class for the graphique interface and interactive mode.
    """

    # -----------------------------------------------------------------------
    # Class-Level Colours

    GOLDEN = THECOLORS['gold']
    BLACK = THECOLORS['black']
    RED = THECOLORS['red']
    GREEN = THECOLORS['green']
    BLUE = THECOLORS['blue']
    YELLOW = THECOLORS['yellow']
    PURPLE = THECOLORS['purple']
    ORANGE = THECOLORS['orange']
    PINK = THECOLORS['pink']
    BROWN = THECOLORS['brown']
    GRAY = THECOLORS['gray']
    WHITE = THECOLORS['white']
    BEIGE = THECOLORS['beige']

    # -----------------------------------------------------------------------
    # High-Level Setup

    def ft_initialize_game(self):
        """
        Logic:
        - Initialise graphical assets and preload scene primitives.

        Return:
        - None
        """
        self.ft_initilize_config_game()
        self.ft_initialize_text_game()
        self.ft_initialize_elements_game()

    def ft_initialize_values_for_gui(self):
        """
        Logic:
        - Seed default values displayed in the GUI panel.

        Return:
        - None
        """
        self.speed_gui = 50

        self.gui_data = {
            'next_step': False,
            'speed': self.speed_gui,
            'learn': False,
            'step_by_step': False,
            'session_id': 0,
            'max_length': 0,
            'epsilon': 0.75,
            'q_value_taken': 0.34,
            'green_apples_taken': 3,
            'red_apples_taken': 3,
            'game_over_reason': 'W',
            'game_over': False,
            'model_loaded': './models/10sess.pkl',
            'duration': 275,
            'reward_session': -3450,
            'vision': ['SW', '00000000W', '000000000W', 'W'],
        }

    def ft_initialize_variables_in_zero(self):
        """
        Logic:
        - Create placeholders for surfaces, fonts, and runtime tracking.

        Return:
        - None
        """
        self.windows = None
        self.background = None
        self.background_image = None
        self.font = None
        self.borders = 0.0
        self.background_snake_size_1 = 0.0
        self.background_snake_size_2 = 0.0
        self.background_snake_pos = (0.0, 0.0)
        self.background_snake = None
        self.side_1 = 0.0
        self.side_2 = 0.0
        self.pos_instructions = (0.0, 0.0)
        self.background_instructions = None
        self.text = None
        self.text_pos = None
        self.font_title = None
        self.font_instructions = None
        self.font_stats = None
        self.font_vision = None
        self.font_key = None
        self.font_button = None
        self.title_left = None
        self.title_right = None
        self.tiles_dimension = (0.0, 0.0)
        self.empty_sprite = None
        self.wall_sprite = None
        self.head_up_sprite = None
        self.head_down_sprite = None
        self.head_left_sprite = None
        self.head_right_sprite = None
        self.tail_up_sprite = None
        self.tail_down_sprite = None
        self.tail_left_sprite = None
        self.tail_right_sprite = None
        self.body_horizontal_sprite = None
        self.body_vertical_sprite = None
        self.body_up_right_sprite = None
        self.body_up_left_sprite = None
        self.body_down_right_sprite = None
        self.body_down_left_sprite = None
        self.green_apple_sprite = None
        self.red_apple_sprite = None
        self.clock = None
        self.event = None
        self.continue_game = False
        self.main_size = 0
        self.title_size = 0
        self.small_size = 0
        self.button_size = 0

    def __init__(self, game: Game, windows_size: int):
        """
        Logic:
        - Bind the game model and prepare lazy-loaded resources.

        Return:
        - None
        """
        self.game = game
        self.windows_size = windows_size

        self.ft_initialize_variables_in_zero()
        self.ft_initialize_values_for_gui()

    # -----------------------------------------------------------------------
    # Window Configuration

    def ft_initilize_config_game(self):
        """
        Logic:
        - Set up the Pygame window, backgrounds, and layout metrics.

        Return:
        - None
        """
        pygame.init()
        self.windows = pygame.display.set_mode(
            (self.windows_size, self.windows_size))
        pygame.display.set_caption("Snake Game")

        self.background = pygame.Surface(self.windows.get_size())

        self.background_image = pygame.image.load(
            './assets/background.png').convert_alpha()
        self.background_image = pygame.transform.scale(
            self.background_image, (self.windows_size, self.windows_size))
        self.background.blit(self.background_image, (0, 0))

        self.borders = self.windows_size * 0.05
        self.background_snake_size_1 = self.windows_size - \
            self.borders * 2 - (self.windows_size * 0.25)
        self.background_snake_size_2 = self.windows_size - self.borders * 2
        self.background_snake_pos = (self.borders, self.borders)
        self.background_snake = pygame.Surface(
            (self.background_snake_size_2, self.background_snake_size_1))
        self.background_snake.fill(self.GRAY)

        self.side_1 = self.windows_size - \
            self.borders * 2 - (self.windows_size * 0.7)
        self.side_2 = self.windows_size - self.borders * 2
        self.pos_instructions = (
            self.borders,
            self.windows_size -
            self.borders -
            self.side_1)
        self.background_instructions = pygame.Surface(
            (self.side_2, self.side_1))
        self.background_instructions.fill(self.GOLDEN)

    # -----------------------------------------------------------------------
    # Static Text Setup

    def ft_initialize_text_game(self):
        """
        Logic:
        - Load fonts and pre-render immutable labels.

        Return:
        - None
        """
        self.main_size = max(12, int(self.windows_size * 0.025))
        self.title_size = max(12, int(self.windows_size * 0.02))
        self.small_size = max(10, int(self.windows_size * 0.0133))
        self.button_size = max(10, int(self.windows_size * 0.014))

        self.font = pygame.font.Font(
            './fonts/JetbrainsMonoBold-51Xez.ttf', self.main_size)
        self.text = self.font.render(
            "Welcome to the AI Snake Game with Q-Learning", True, self.BLACK)
        self.text_pos = self.text.get_rect()
        self.text_pos.centerx = self.background.get_rect().centerx
        self.text_pos.centery = self.borders * 0.5

        self.font_title = pygame.font.Font(
            './fonts/JetbrainsMonoBold-51Xez.ttf', self.title_size)
        self.font_instructions = pygame.font.Font(
            './fonts/JetbrainsMonoRegular-RpvmM.ttf', self.small_size)
        self.font_stats = pygame.font.Font(
            './fonts/JetbrainsMonoRegular-RpvmM.ttf', self.small_size)
        self.font_vision = pygame.font.Font(
            './fonts/JetbrainsMonoBoldItalic-6YyW6.ttf', self.small_size)
        self.font_key = pygame.font.Font(
            './fonts/JetbrainsMonoRegular-RpvmM.ttf', self.small_size)
        self.font_button = pygame.font.Font(
            './fonts/JetbrainsMonoBold-51Xez.ttf', self.button_size)

        self.title_left = self.font_title.render("INSTRUCTIONS", True, self.BLACK)
        self.title_right = self.font_title.render("STATS AGENT", True, self.BLACK)

    # -----------------------------------------------------------------------
    # Sprite Preparation

    def ft_initialize_elements_game(self):
        """
        Logic:
        - Load, scale, and cache all textures used during rendering.

        Return:
        - None
        """
        self.tiles_dimension = (
            self.background_snake_size_2 /
            self.game.width,
            self.background_snake_size_1 /
            self.game.height)

        empty_image = pygame.image.load('./assets/empty.png').convert_alpha()
        wall_image = pygame.image.load('./assets/wall.png').convert_alpha()

        head_up_image = pygame.image.load(
            './assets/head_up.png').convert_alpha()
        head_down_image = pygame.image.load(
            './assets/head_down.png').convert_alpha()
        head_left_image = pygame.image.load(
            './assets/head_left.png').convert_alpha()
        head_right_image = pygame.image.load(
            './assets/head_right.png').convert_alpha()

        tail_up_image = pygame.image.load(
            './assets/tail_up.png').convert_alpha()
        tail_down_image = pygame.image.load(
            './assets/tail_down.png').convert_alpha()
        tail_left_image = pygame.image.load(
            './assets/tail_left.png').convert_alpha()
        tail_right_image = pygame.image.load(
            './assets/tail_right.png').convert_alpha()

        body_horizontal_image = pygame.image.load(
            './assets/body_horizontal.png').convert_alpha()
        body_vertical_image = pygame.image.load(
            './assets/body_vertical.png').convert_alpha()
        body_up_right_image = pygame.image.load(
            './assets/body_up_right.png').convert_alpha()
        body_up_left_image = pygame.image.load(
            './assets/body_up_left.png').convert_alpha()
        body_down_right_image = pygame.image.load(
            './assets/body_down_right.png').convert_alpha()
        body_down_left_image = pygame.image.load(
            './assets/body_down_left.png').convert_alpha()

        green_apple_image = pygame.image.load(
            './assets/green_apple.png').convert_alpha()
        red_apple_image = pygame.image.load(
            './assets/red_apple.png').convert_alpha()

        scaled_size = (int(self.tiles_dimension[0]), int(
            self.tiles_dimension[1]))

        self.empty_sprite = pygame.transform.scale(empty_image, scaled_size)
        self.wall_sprite = pygame.transform.scale(wall_image, scaled_size)

        self.head_up_sprite = pygame.transform.scale(
            head_up_image, scaled_size)
        self.head_down_sprite = pygame.transform.scale(
            head_down_image, scaled_size)
        self.head_left_sprite = pygame.transform.scale(
            head_left_image, scaled_size)
        self.head_right_sprite = pygame.transform.scale(
            head_right_image, scaled_size)

        self.tail_up_sprite = pygame.transform.scale(
            tail_up_image, scaled_size)
        self.tail_down_sprite = pygame.transform.scale(
            tail_down_image, scaled_size)
        self.tail_left_sprite = pygame.transform.scale(
            tail_left_image, scaled_size)
        self.tail_right_sprite = pygame.transform.scale(
            tail_right_image, scaled_size)

        self.body_horizontal_sprite = pygame.transform.scale(
            body_horizontal_image, scaled_size)
        self.body_vertical_sprite = pygame.transform.scale(
            body_vertical_image, scaled_size)
        self.body_up_right_sprite = pygame.transform.scale(
            body_up_right_image, scaled_size)
        self.body_up_left_sprite = pygame.transform.scale(
            body_up_left_image, scaled_size)
        self.body_down_right_sprite = pygame.transform.scale(
            body_down_right_image, scaled_size)
        self.body_down_left_sprite = pygame.transform.scale(
            body_down_left_image, scaled_size)

        self.green_apple_sprite = pygame.transform.scale(
            green_apple_image, scaled_size)
        self.red_apple_sprite = pygame.transform.scale(
            red_apple_image, scaled_size)

    # -----------------------------------------------------------------------
    # Sprite Selection Helpers

    def ft_setting_head_snake_sprite(self, row: int, col: int):
        """
        Logic:
        - Resolve the appropriate head sprite based on orientation.

        Return:
        - pygame.Surface
        """
        head_sprite = self.head_up_sprite

        up = (row - 1, col)
        down = (row + 1, col)
        left = (row, col - 1)
        right = (row, col + 1)

        if len(self.game.snake) > 1:
            body_pos = tuple(self.game.snake[1])

            if body_pos == up:
                head_sprite = self.head_down_sprite
            elif body_pos == down:
                head_sprite = self.head_up_sprite
            elif body_pos == left:
                head_sprite = self.head_right_sprite
            elif body_pos == right:
                head_sprite = self.head_left_sprite

        if self.game.last_move == "up":
            head_sprite = self.head_up_sprite
        elif self.game.last_move == "down":
            head_sprite = self.head_down_sprite
        elif self.game.last_move == "left":
            head_sprite = self.head_left_sprite
        elif self.game.last_move == "right":
            head_sprite = self.head_right_sprite

        return head_sprite

    def ft_setting_body_snake_sprite(self, row: int, col: int):
        """
        Logic:
        - Pick the body segment sprite matching its neighbours.

        Return:
        - pygame.Surface
        """
        body_sprite = self.body_horizontal_sprite

        if len(self.game.snake) < 3:
            return body_sprite

        current_pos = (row, col)
        body_index = -1
        for i, pos in enumerate(self.game.snake):
            if tuple(pos) == current_pos:
                body_index = i
                break

        if body_index == - \
                1 or body_index == 0 or body_index == len(self.game.snake) - 1:
            return body_sprite

        up = (row - 1, col)
        down = (row + 1, col)
        left = (row, col - 1)
        right = (row, col + 1)

        prev_pos = tuple(self.game.snake[body_index - 1])
        next_pos = tuple(self.game.snake[body_index + 1])

        has_up = (prev_pos == up or next_pos == up)
        has_down = (prev_pos == down or next_pos == down)
        has_left = (prev_pos == left or next_pos == left)
        has_right = (prev_pos == right or next_pos == right)

        if has_left and has_right:
            body_sprite = self.body_horizontal_sprite
        elif has_up and has_down:
            body_sprite = self.body_vertical_sprite
        elif has_up and has_right:
            body_sprite = self.body_up_right_sprite
        elif has_up and has_left:
            body_sprite = self.body_up_left_sprite
        elif has_down and has_right:
            body_sprite = self.body_down_right_sprite
        elif has_down and has_left:
            body_sprite = self.body_down_left_sprite

        return body_sprite

    def ft_setting_tail_snake_sprite(
            self,
            row: int,
            col: int,
            sprite: pygame.Surface):
        """
        Logic:
        - Swap the sprite if the segment corresponds to the tail tip.

        Return:
        - pygame.Surface
        """
        if len(self.game.snake) <= 1:
            return sprite

        tail_pos = tuple(self.game.snake[-1])
        if (row, col) != tail_pos:
            return sprite

        up = (row - 1, col)
        down = (row + 1, col)
        left = (row, col - 1)
        right = (row, col + 1)

        tail_sprite = self.tail_up_sprite
        if len(self.game.snake) > 1:
            body_before_tail = tuple(self.game.snake[-2])

            if body_before_tail == up:
                tail_sprite = self.tail_up_sprite
            elif body_before_tail == down:
                tail_sprite = self.tail_down_sprite
            elif body_before_tail == left:
                tail_sprite = self.tail_left_sprite
            elif body_before_tail == right:
                tail_sprite = self.tail_right_sprite

        return tail_sprite

    def ft_draw_map(self):
        """
        Logic:
        - Render the current game grid onto the snake surface.

        Return:
        - None
        """
        self.background_snake.fill(self.GRAY)

        for row in range(self.game.height):
            for col in range(self.game.width):
                x_pos = col * int(self.tiles_dimension[0])
                y_pos = row * int(self.tiles_dimension[1])

                cell_value = self.game.map[row, col]

                if cell_value == Game.EMPTY:
                    sprite = self.empty_sprite
                elif cell_value == Game.WALL:
                    sprite = self.wall_sprite
                elif cell_value == Game.HEAD:
                    sprite = self.ft_setting_head_snake_sprite(row, col)
                elif cell_value == Game.BODY:
                    sprite = self.ft_setting_body_snake_sprite(row, col)
                elif cell_value == Game.GREEN_APPLE:
                    sprite = self.green_apple_sprite
                elif cell_value == Game.RED_APPLE:
                    sprite = self.red_apple_sprite
                else:
                    sprite = self.empty_sprite

                sprite = self.ft_setting_tail_snake_sprite(row, col, sprite)
                self.background_snake.blit(self.empty_sprite, (x_pos, y_pos))
                self.background_snake.blit(sprite, (x_pos, y_pos))

    # -----------------------------------------------------------------------
    # Instruction Panel Rendering

    def ft_draw_borders_in_background_instructions(self):
        """
        Logic:
        - Render ornamental borders framing the instruction panel.

        Return:
        - None
        """
        beige_color = self.BEIGE
        self.background_instructions.fill(beige_color)

        border_thickness = 5
        pygame.draw.rect(self.background_instructions, self.BLACK,
                         (0, 0, self.side_2, self.side_1), border_thickness)

        inner_border_thickness = 3
        inner_offset = border_thickness + inner_border_thickness
        inner_width = self.side_2 - inner_offset * 2
        inner_height = self.side_1 - inner_offset * 2
        pygame.draw.rect(
            self.background_instructions,
            self.GRAY,
            (border_thickness,
             border_thickness,
             self.side_2 - border_thickness * 2,
             self.side_1 - border_thickness * 2),
            inner_border_thickness)

        pygame.draw.rect(self.background_instructions, self.BLACK,
                         (inner_offset,
                          inner_offset,
                          inner_width,
                          inner_height),
                         2)

    def ft_draw_boutons_in_background_instructions(
            self,
            buttons: dict,
            start_x: int,
            start_y: int,
            line_spacing: int,
            color_key: tuple,
            color_button: tuple,
            font_key: pygame.font.Font,
            font_button: pygame.font.Font):
        """
        Logic:
        - Draw instruction and stats content onto the instruction panel.

        Return:
        - None
        """
        for i, (key, button) in enumerate(buttons.items()):
            text_surface_key = font_key.render(f"{key}:", True, color_key)
            text_surface_button = font_button.render(
                f"{button}", True, color_button)
            y_pos = start_y + i * line_spacing
            button_x = start_x + text_surface_key.get_width() + 10
            self.background_instructions.blit(
                text_surface_key, (start_x, y_pos))
            self.background_instructions.blit(
                text_surface_button, (button_x, y_pos))

    def ft_draw_instructions_text(self):
        """
        Logic:
        - Lay out instructions, stats, and vision columns.

        Return:
        - None
        """
        height_position = max(14, int(self.windows_size * 0.025))
        separator_x = int(self.side_2 * 0.30)
        pygame.draw.line(self.background_instructions, self.BLACK,
                         (separator_x, 20), (separator_x, self.side_1 - 20), 2)

        left_title_x = (separator_x - self.title_left.get_width()) // 2
        self.background_instructions.blit(
            self.title_left, (left_title_x, height_position))

        right_title_x = separator_x + \
            (self.side_2 - separator_x - self.title_right.get_width()) // 2
        self.background_instructions.blit(
            self.title_right, (right_title_x, height_position))

        start_X = max(10, int(self.side_2 * 0.03)) 
        start_y = max(10, int(self.side_1 * 0.12)) + (height_position * 1.50)
        line_spacing = max(12, int(self.side_1 * 0.12))

        buttons = {
            "Next step": "RIGHT arrow key",
            "Speed +1/-1": "UP/DOWN arrow key",
            "Learn on/off": "L button",
            "Toggle Step by step": "S button",
            "Quit": "Q button"
        }
        self.ft_draw_boutons_in_background_instructions(
            buttons,
            start_X,
            start_y,
            line_spacing,
            self.BLACK,
            self.BLUE,
            self.font_key,
            self.font_instructions)
        
        start_x = separator_x + max(8, int(self.side_2 * 0.02))
        stats_start_y = start_y
        line_spacing_stats = max(12, int(self.side_1 * 0.1))
        right_col_padding = max(50, int(self.side_2 * 0.12))
        right_col_width = self.side_2 - separator_x - right_col_padding
        col_width = right_col_width // 3

        stats_col1 = {
            "Session": f"{self.gui_data['session_id']}",
            "Max Length": f"{self.gui_data['max_length']}",
            "Epsilon": f"{self.gui_data['epsilon']:.2f}",
            "Q-Value": f"{self.gui_data['q_value_taken']:.2f}",
            "Duration": f"{self.gui_data['duration']} steps",
            "Reward": f"{self.gui_data['reward_session']}"
        }

        self.ft_draw_boutons_in_background_instructions(
            stats_col1,
            start_x,
            stats_start_y,
            line_spacing_stats,
            self.BLACK,
            self.RED,
            self.font_key,
            self.font_button)

        start_x = separator_x + col_width + 10
        stats_col2 = {
            "Green Apples": f"{self.gui_data['green_apples_taken']}",
            "Red Apples": f"{self.gui_data['red_apples_taken']}",
            "Game Over": f"{self.gui_data['game_over']}",
            "Reason": f"{self.gui_data['game_over_reason']}",
            "Model": f"{self.gui_data['model_loaded'].split('/')[-1]}",
            "Mode Learning": f"{self.gui_data['learn']}"
        }
        self.ft_draw_boutons_in_background_instructions(
            stats_col2,
            start_x,
            stats_start_y,
            line_spacing_stats,
            self.BLACK,
            self.RED,
            self.font_key,
            self.font_button)

        start_x = separator_x + 2 * col_width + 5
        stats_col3 = {
            "Speed": f"{self.gui_data['speed']} FPS",
            "Step-by-step": f"{self.gui_data['step_by_step']}"
        }

        self.ft_draw_boutons_in_background_instructions(
            stats_col3,
            start_x,
            stats_start_y,
            line_spacing_stats,
            self.BLACK,
            self.RED,
            self.font_key,
            self.font_button)

        visions_start_y = stats_start_y + len(stats_col3) * line_spacing_stats + max(5, int(self.side_1 * 0.02))
        visions = {
            "Up": f"{self.gui_data['vision'][0]}",
            "Down": f"{self.gui_data['vision'][1]}",
            "Left": f"{self.gui_data['vision'][2]}",
            "Right": f"{self.gui_data['vision'][3]}"
        }
        visions_line_spacing = max(15, int(self.side_1 * 0.05))
        self.ft_draw_boutons_in_background_instructions(
            visions,
            start_x,
            visions_start_y,
            visions_line_spacing,
            self.BLACK,
            self.BLUE,
            self.font_key,
            self.font_instructions)

    # -----------------------------------------------------------------------
    # Event Loop

    def ft_manage_events(self):
        """
        Logic:
        - Blit the latest scene and flip the display buffer.

        Return:
        - None
        """
        self.windows.blit(self.background, (0, 0))

        self.ft_draw_map()
        self.ft_draw_borders_in_background_instructions()
        self.ft_draw_instructions_text()

        self.windows.blit(self.background_snake, self.background_snake_pos)
        self.windows.blit(self.background_instructions, self.pos_instructions)
        self.windows.blit(self.text, self.text_pos)

        pygame.display.flip()

    def ft_run_game(self):
        """
        Logic:
        - Drive a manual play session controlled by the keyboard.

        Return:
        - None
        """
        self.continue_game = True
        clock = pygame.time.Clock()
        self.event = None

        while self.continue_game:
            for event in pygame.event.get():
                self.event = event
                if event.type == pygame.QUIT:
                    self.continue_game = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        code, is_over = self.game.ft_move_snake('up', 'off')
                    elif event.key == pygame.K_DOWN:
                        code, is_over = self.game.ft_move_snake('down', 'off')
                    elif event.key == pygame.K_LEFT:
                        code, is_over = self.game.ft_move_snake('left', 'off')
                    elif event.key == pygame.K_RIGHT:
                        code, is_over = self.game.ft_move_snake('right', 'off')
                    elif event.key == pygame.K_q:
                        self.ft_quit_game()
                    else:
                        continue
                    if is_over:
                        self.game.ft_reset()
            self.ft_manage_events()
            clock.tick(10)

    def ft_run_game_with_agent(self, gui_data: dict):
        """
        Logic:
        - Refresh the GUI while delegating decisions to the agent.

        Return:
        - dict
        """
        self.clock = pygame.time.Clock()
        self.gui_data = gui_data

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.ft_quit_game()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.ft_quit_game()
                elif event.key == pygame.K_UP:
                    if gui_data['speed'] == 1:
                        gui_data['speed'] = 5
                    else:
                        gui_data['speed'] += 5
                elif event.key == pygame.K_DOWN:
                    gui_data['speed'] = max(1, gui_data['speed'] - 5)
                elif event.key == pygame.K_RIGHT:
                    if gui_data['step_by_step']:
                        gui_data['next_step'] = True
                elif event.key == pygame.K_l:
                    gui_data['learn'] = not gui_data['learn']
                elif event.key == pygame.K_s:
                    gui_data['step_by_step'] = not gui_data['step_by_step']
                    if not gui_data['step_by_step']:
                        gui_data['next_step'] = False

        self.ft_manage_events()
        self.clock.tick(gui_data['speed'])

        return gui_data

    def ft_quit_game(self):
        """
        Logic:
        - Close the Pygame window and exit cleanly.

        Return:
        - None
        """
        pygame.quit()
        sys.exit()

    # -----------------------------------------------------------------------
    # Type Hints

    game: Game
    windows_size: int
    speed_gui: int
    gui_data: Dict[str, Any]
    windows: pygame.Surface
    background: pygame.Surface
    background_image: pygame.Surface
    font: pygame.font.Font
    borders: float
    background_snake_size_1: float
    background_snake_size_2: float
    background_snake_pos: Tuple[float, float]
    background_snake: pygame.Surface
    side_1: float
    side_2: float
    pos_instructions: Tuple[float, float]
    background_instructions: pygame.Surface
    text: pygame.Surface
    text_pos: pygame.Rect
    font_title: pygame.font.Font
    font_instructions: pygame.font.Font
    font_stats: pygame.font.Font
    font_vision: pygame.font.Font
    font_key: pygame.font.Font
    font_button: pygame.font.Font
    title_left: pygame.Surface
    title_right: pygame.Surface
    tiles_dimension: Tuple[float, float]
    empty_sprite: pygame.Surface
    wall_sprite: pygame.Surface
    head_up_sprite: pygame.Surface
    head_down_sprite: pygame.Surface
    head_left_sprite: pygame.Surface
    head_right_sprite: pygame.Surface
    tail_up_sprite: pygame.Surface
    tail_down_sprite: pygame.Surface
    tail_left_sprite: pygame.Surface
    tail_right_sprite: pygame.Surface
    body_horizontal_sprite: pygame.Surface
    body_vertical_sprite: pygame.Surface
    body_up_right_sprite: pygame.Surface
    body_up_left_sprite: pygame.Surface
    body_down_right_sprite: pygame.Surface
    body_down_left_sprite: pygame.Surface
    green_apple_sprite: pygame.Surface
    red_apple_sprite: pygame.Surface
    clock: pygame.time.Clock
    event: pygame.event.Event
    continue_game: bool


def main():
    try:
        size = 10
        windows_size = 1400
        game = Game(size, size)
        game_gui = GameGUI(game, windows_size)
        game_gui.ft_initialize_game()
        game_gui.ft_run_game()
        game_gui.ft_quit_game()

    except Exception as e:
        print(f"Error: {e}")
        del game
        del game_gui
        sys.exit(1)


if __name__ == "__main__":
    main()
