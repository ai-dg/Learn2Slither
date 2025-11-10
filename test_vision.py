#!/usr/bin/env python3

"""
Logic:
- Compare the two vision functions in a standalone script.

Return:
- None
"""

from game_data import Game


def test_vision_comparison():
    """
    Logic:
    - Print the output lengths for standard and extended vision.

    Return:
    - None
    """
    print("🔍 COMPARISON OF THE TWO VISION FUNCTIONS")
    print("=" * 50)

    game = Game(10, 10)

    for test_num in range(5):
        game.ft_reset()
        print(f"Game: \n{game}")

        vision_ft = game.ft_get_vision_snake()
        vision_ft_extended = game.ft_get_vision_snake_extended()

        print(f"\nTest {test_num + 1}:")
        print(
            f"  ft_get_vision_snake(): "
            f"'{vision_ft}' (length: {len(vision_ft)})")
        print(
            f"  ft_get_vision_snake_extended(): "
            f"'{vision_ft_extended}' (length: {len(vision_ft_extended)})")
        print(
            f"  Snake head: "
            f"{game.snake[0] if len(game.snake) > 0 else 'None'}")
        print(f"  Game over: {game.is_over}")


if __name__ == "main__":
    test_vision_comparison()
