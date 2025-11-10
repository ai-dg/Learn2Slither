# test_game_data.py
# pytest -v test_game_data.py

import random
import numpy as np
import pytest

from game_data import Game


# ---------------------------
# Helpers déterministes
# ---------------------------

def seed(n=0):
    random.seed(n)
    np.random.seed(n)

def make_game(w=10, h=10, seedv=0):
    seed(seedv)
    return Game(w, h)

def empty_cells(g: Game):
    return np.argwhere(g.map == g.EMPTY)

def wall_cells(g: Game):
    return np.argwhere(g.map == g.WALL)

def head(g: Game):
    return tuple(g.snake[0])

def tail(g: Game):
    return tuple(g.snake[-1])

def set_cell(g: Game, rc, code):
    g.map[tuple(rc)] = code

def counts(g: Game):
    return {
        "greens": int(np.sum(g.map == g.GREEN_APPLE)),
        "reds": int(np.sum(g.map == g.RED_APPLE)),
        "heads": int(np.sum(g.map == g.HEAD)),
        "bodies": int(np.sum(g.map == g.BODY)),
        "empties": int(np.sum(g.map == g.EMPTY)),
        "walls": int(np.sum(g.map == g.WALL)),
    }

def dir_to(a, b):
    # a,b: (r,c) adjacents
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    if dr == -1 and dc == 0: return "up"
    if dr == 1 and dc == 0: return "down"
    if dr == 0 and dc == -1: return "left"
    if dr == 0 and dc == 1: return "right"
    raise AssertionError(f"not adjacent: a={a}, b={b}")

def put_green_ahead(g: Game):
    """Place une pomme verte juste devant la tête (case libre)."""
    r, c = head(g)
    for d in ["up", "down", "left", "right"]:
        dr, dc = g.directions[d]
        tgt = (r + dr, c + dc)
        if g.map[tgt] != g.WALL and g.map[tgt] in (g.EMPTY, g.GREEN_APPLE, g.RED_APPLE, g.BODY):
            # on force la case à EMPTY puis on met GREEN
            set_cell(g, tgt, g.EMPTY)
            set_cell(g, tgt, g.GREEN_APPLE)
            return tgt, d
    raise AssertionError("No adjacent place to put green apple!")

def put_red_ahead(g: Game):
    r, c = head(g)
    for d in ["up", "down", "left", "right"]:
        dr, dc = g.directions[d]
        tgt = (r + dr, c + dc)
        if g.map[tgt] != g.WALL and g.map[tgt] in (g.EMPTY, g.GREEN_APPLE, g.RED_APPLE, g.BODY):
            set_cell(g, tgt, g.EMPTY)
            set_cell(g, tgt, g.RED_APPLE)
            return tgt, d
    raise AssertionError("No adjacent place to put red apple!")

def shape_ok(g: Game, w=10, h=10):
    # Game ajoute une bordure de murs → (h+2, w+2)
    assert g.map.shape == (h + 2, w + 2)

# ---------------------------
# Tests de base / invariants
# ---------------------------

def test_reset_invariants():
    g = make_game(10, 10, seedv=1)
    shape_ok(g)
    cnt = counts(g)
    assert len(g.snake) == 3
    assert cnt["greens"] == 2
    assert cnt["reds"] == 1
    # head unique, bodies = 2
    assert cnt["heads"] == 1
    assert cnt["bodies"] == 2
    # murs présents sur le pourtour
    assert wall_cells(g).shape[0] > 0

def test_reset_randomness_head_positions():
    g = make_game(10, 10, seedv=2)
    heads = set()
    for s in range(30):
        g.ft_reset()
        heads.add(head(g))
    # on veut plusieurs positions distinctes
    assert len(heads) >= 5

def test_apples_never_on_wall_or_snake():
    g = make_game(10, 10, seedv=3)
    # vérifie à l'init
    for rc in np.argwhere((g.map == g.GREEN_APPLE) | (g.map == g.RED_APPLE)):
        rc = tuple(rc)
        assert g.map[rc] not in (g.WALL, g.HEAD, g.BODY)

    # puis après plusieurs steps+resets
    for _ in range(150):
        a = random.choice(["up", "down", "left", "right"])
        code, over = g.ft_move_snake(a, "off")
        for rc in np.argwhere((g.map == g.GREEN_APPLE) | (g.map == g.RED_APPLE)):
            rc = tuple(rc)
            assert g.map[rc] not in (g.WALL, g.HEAD, g.BODY)
        if over:
            g.ft_reset()

def test_invariants_apples_count_always_2_green_1_red():
    g = make_game(10, 10, seedv=4)
    for _ in range(200):
        a = random.choice(["up", "down", "left", "right"])
        code, over = g.ft_move_snake(a, "off")
        cnt = counts(g)
        assert cnt["greens"] == 2
        assert cnt["reds"] == 1
        if over:
            g.ft_reset()

# ---------------------------
# Mouvement / collisions
# ---------------------------

def test_move_into_wall_is_gameover():
    g = make_game(10, 10, seedv=5)
    # place la tête près d'un mur en forçant des moves aléatoires
    # puis bouge vers le mur
    found = False
    for _ in range(200):
        r, c = head(g)
        # si un mur est adjacent on le percute
        for d in ["up", "down", "left", "right"]:
            dr, dc = g.directions[d]
            tgt = (r + dr, c + dc)
            if g.map[tgt] == g.WALL:
                code, over = g.ft_move_snake(d, "off")
                assert code == g.WALL and over
                found = True
                break
        if found:
            break
        # sinon on se déplace au hasard
        d = random.choice(["up", "down", "left", "right"])
        code, over = g.ft_move_snake(d, "off")
        if over:
            g.ft_reset()
    assert found, "n'a pas trouvé de mur adjacent dans 200 steps"

def test_tail_pass_on_empty_move_no_collision():
    g = make_game(10, 10, seedv=6)
    g.ft_reset()

    # Nettoie serpent actuel
    g.map[g.map == g.HEAD] = g.EMPTY
    g.map[g.map == g.BODY] = g.EMPTY

    # U-shape pour que HEAD soit adjacent à TAIL :
    #   (r,c) = B
    #   (r,c+1) = H
    #   (r+1,c) = T   -> la tête va bouger "left" vers T (ancienne queue)
    r, c = 5, 5
    H = (r, c+1)
    B = (r, c)
    T = (r+1, c)

    # s'assurer que ces cases sont libres
    for rc in (H, B, T):
        assert g.map[rc] == g.EMPTY

    g.snake = np.array([H, B, T])
    set_cell(g, H, g.HEAD)
    set_cell(g, B, g.BODY)
    set_cell(g, T, g.BODY)

    # Move "left": map_pos == B (corps) -> collision; on veut passer par T !
    # Donc on se décale d'abord "down" pour mettre la tête à (r+1,c+1),
    # puis "left" pour entrer sur (r+1,c) == T (ancienne queue).
    code, over = g.ft_move_snake("down", "off")
    assert not over
    # maintenant HEAD = (r+1, c+1) et TAIL = T ; on va à gauche sur T (tail-pass)
    code, over = g.ft_move_snake("left", "off")
    assert not over, "tail-pass devrait être permis quand on entre sur l'ANCIENNE queue"

def test_collision_with_body_not_tail_is_gameover():
    """
    Percuter le corps (autre que l'ancienne queue) doit tuer.
    """
    g = make_game(10, 10, seedv=7)
    g.ft_reset()
    # Nettoie serpent actuel
    g.map[g.map == g.HEAD] = g.EMPTY
    g.map[g.map == g.BODY] = g.EMPTY

    # Ligne horizontale : H(5,5) - B1(5,4) - T(5,3)
    H = (5, 5); B1 = (5, 4); T = (5, 3)
    for rc in (H, B1, T):
        assert g.map[rc] == g.EMPTY
    g.snake = np.array([H, B1, T])
    set_cell(g, H, g.HEAD)
    set_cell(g, B1, g.BODY)
    set_cell(g, T, g.BODY)

    # Aller "left" : on percute B1 (pas la queue) ⇒ game over avec code BODY
    code, over = g.ft_move_snake("left", "off")
    assert over and code == g.BODY


# ---------------------------
# Croissance / réduction
# ---------------------------

def test_green_increases_length_by_one_and_respawns():
    g = make_game(10, 10, seedv=9)
    L0 = len(g.snake)
    pos, d = put_green_ahead(g)
    code, over = g.ft_move_snake(d, "off")
    assert code == g.GREEN_APPLE and not over
    assert len(g.snake) == L0 + 1
    cnt = counts(g)
    assert cnt["greens"] == 2 and cnt["reds"] == 1  # invariant maintenu

def test_red_decreases_length_by_one_or_gameover_and_respawns():
    g = make_game(10, 10, seedv=10)
    L0 = len(g.snake)
    pos, d = put_red_ahead(g)
    code, over = g.ft_move_snake(d, "off")
    assert code == g.RED_APPLE
    if not over:
        assert len(g.snake) == L0 - 1
        cnt = counts(g)
        assert cnt["reds"] == 1  # respawn ok

def test_many_steps_keep_invariants_and_no_index_errors():
    g = make_game(10, 10, seedv=11)
    for _ in range(500):
        d = random.choice(["up", "down", "left", "right"])
        code, over = g.ft_move_snake(d, "off")
        # invariants pommes
        cnt = counts(g)
        assert cnt["greens"] == 2
        assert cnt["reds"] == 1
        # synchro: 1 seule tête
        assert cnt["heads"] == 1 if not over else True
        if over:
            g.ft_reset()

# ---------------------------
# Vision (test léger)
# ---------------------------

def test_vision_has_walls_at_end_each_direction():
    g = make_game(10, 10, seedv=12)
    vis = g.ft_get_vision_snake()
    # chaque direction doit se terminer sur un mur (1)
    for k, arr in vis.items():
        assert len(arr) >= 1
        assert arr[-1] == g.WALL

# ---------------------------
# Lancement direct sans pytest CLI
# ---------------------------

if __name__ == "__main__":
    import pytest as _pytest
    raise SystemExit(_pytest.main(["-v", __file__]))
