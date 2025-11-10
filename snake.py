#!/usr/bin/env python3

import sys
import argparse
from tabulate import tabulate
from agent import QLearning
from game_data import Game
from game_gui import GameGUI


class Snake:
    """
        Snake class for the Snake game
        In this class, general implementation to handle QLearning
        and to hangle the Game and GameGUI data.
    """

    def __init__(self, args: argparse.Namespace):
        """
        Logic:
        - Store CLI arguments and prepare lazy-loaded components.

        Return:
        - None
        """
        self.args = args
        self.game = None
        self.gui = None
        self.agent = None

    def __str__(self):
        """
        Logic:
        - Return a tabular summary of the parsed arguments.

        Return:
        - str
        """
        rows = [
            ["Sessions", self.args.sessions],
            ["Visual", self.args.visual],
            ["Save", self.args.save],
            ["Load", self.args.load],
            ["Dontlearn", self.args.dontlearn],
            ["Step-by-step", self.args.step_by_step],
            ["Speed", self.args.speed],
            ["Size", self.args.size],
            ["Stats", self.args.stats],
            ["Plot", self.args.plot]
        ]
        return tabulate(
            rows,
            headers=[
                "Parameter",
                "Value"],
            tablefmt="simple")

    def ft_initialize_snake(self):
        """
        Logic:
        - Instantiate the game, GUI, and learning agent.

        Return:
        - None
        """
        size = self.args.size
        windows_size = 1500
        self.game = Game(size, size)
        self.gui = GameGUI(self.game, windows_size)
        self.agent = QLearning(self.game)
        self.agent.ft_set_terminal_mode(self.args.terminal)
        self.agent.ft_set_speed_gui(self.args.speed)

    def ft_set_options_snake(self):
        """
        Logic:
        - Configure optional behaviours based on CLI flags.

        Return:
        - None
        """
        if self.args.visual == 'on':
            self.agent.ft_set_visual_mode(True, self.gui)
        if self.args.load:
            self.agent.ft_load_model(self.args.load)
        if self.args.dontlearn:
            self.agent.ft_set_learning_mode(False)
        if self.args.step_by_step:
            self.agent.ft_set_step_by_step_mode(True)
        if self.args.stats:
            self.agent.ft_show_stats(True)
        if self.args.plot:
            self.agent.ft_set_plot(True)

    def ft_launch_snake(self):
        """
        Logic:
        - Run the full training workflow and cleanup afterwards.

        Return:
        - None
        """
        self.ft_initialize_snake()
        self.ft_set_options_snake()
        self.agent.ft_train_agent(self.args.sessions)
        if self.args.save:
            self.agent.ft_save_model(self.args.save)
        del self.game
        del self.gui
        del self.agent


def ft_parsing_arguments() -> argparse.Namespace:
    """
    Logic:
    - Parse command-line arguments exposed by the CLI.

    Return:
    - argparse.Namespace
    """

    # Mandatory part
    parser = argparse.ArgumentParser(
        description="Learn2Slither - IA Q function with Neural Networks")
    parser.add_argument('-sessions', type=int, default=10,
                        help='how many training sessions?')
    parser.add_argument('-visual', choices=['on', 'off'], default='off',
                        help="visual on or off?")
    parser.add_argument('-terminal', choices=['on', 'off'], default='on',
                        help="visual on or off?")
    parser.add_argument('-save', type=str,
                        help='path to save the model .pkl')
    parser.add_argument('-load', type=str,
                        help='path to load the model .pkl')
    parser.add_argument('-dontlearn', action='store_true',
                        help='disable learning (only during evaluation)')
    parser.add_argument('-step-by-step', action='store_true',
                        help='pause in each step in visual mode')

    # Bonus
    parser.add_argument('-speed', type=int, default=50,
                        help='visual speed frame per second fps (default 10)')
    parser.add_argument('-size', type=int, default=10,
                        help='board size of one side in px (default: 10px)')
    parser.add_argument('-stats', action='store_true',
                        help='show stats at the end of the game')
    parser.add_argument('-plot', action='store_true',
                        help='plot the stats')
    return parser.parse_args()


def ft_check_parser_values(args: argparse.Namespace) -> None:
    """
    Logic:
    - Validate parsed arguments before launching the program.

    Return:
    - None
    """
    if args.size > 40:
        print("Size is too big. It must be less than 40.")
        sys.exit(1)


def main():
    args = ft_parsing_arguments()
    ft_check_parser_values(args)
    game = Snake(args)
    game.ft_launch_snake()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, Exception) as e:
        if isinstance(e, KeyboardInterrupt):
            print("KeyboardInterrupt detected. Exiting...")
            sys.exit(0)
        else:
            print(f"Error: {e}")
        sys.exit(1)
