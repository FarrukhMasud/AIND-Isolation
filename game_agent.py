"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    p1 = player
    p2 = game.get_opponent(p1)
    p1_move_list = game.get_legal_moves(p1)
    p2_move_list = game.get_legal_moves(p2)
    open_move_list = game.get_blank_spaces()
    open_moves = float(len(open_move_list))
    p1_moves = float(len(p1_move_list))
    p2_moves = float(len(p2_move_list))

    if game.is_loser(p1) or p1_moves == 0:
        return float('-inf')
    if game.is_winner(p1) or p2_moves == 0:
        return float('inf')

    for p in p1_move_list:
        for q in p2_move_list:
            x = abs(p[0] - q[0])
            y = abs(p[1] - q[1])
            if (x == 0 and y == 0) or (x == 2 and y == 1) or (y == 2 and x == 1):
                p1_moves -= 1
                break

    return p1_moves - p2_moves


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=8, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        if len(legal_moves) == 0:
            invalid = (-1, -1)
            return invalid
        if game.move_count <= 1:
            return legal_moves[int(len(legal_moves) / 2)]
        mv = legal_moves[0]
        depth = 1
        if not self.iterative:
            depth = self.search_depth
        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            while depth <= self.search_depth or self.search_depth < 0:
                if self.method == 'minimax':
                    _, mv = self.minimax(game, depth, True)
                else:
                    _, mv = self.alphabeta(game, depth, True)

                depth += 1
        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return mv

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        # if game.is_winner(game.active_player):
        #     return float('inf') if maximizing_player else float('-inf'), game.get_player_location(game.active_player)
        #
        # if game.is_loser(game.active_player):
        #     return float('-inf') if maximizing_player else float('inf'), game.get_player_location(game.active_player)

        legal_moves = game.get_legal_moves(game.active_player)
        if len(legal_moves) == 0:
            return float('-inf') if maximizing_player else float('inf'), game.get_player_location(game.active_player)

        if depth == 1:
            if maximizing_player:
                scr, mv = max([(self.score(game.forecast_move(m), self), m) for m in legal_moves])
            else:
                scr, mv = min([(self.score(game.forecast_move(m), self), m) for m in legal_moves])
        else:
            if maximizing_player:
                scr, mv = max(
                    [(self.minimax(game.forecast_move(m), depth - 1, not maximizing_player)[0], m) for m in
                     legal_moves])
            else:
                scr, mv = min(
                    [(self.minimax(game.forecast_move(m), depth - 1, not maximizing_player)[0], m) for m in
                     legal_moves])

        return scr, mv

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        #
        # if game.is_winner(game.active_player):
        #     return float('inf') if maximizing_player else float('-inf'), game.get_player_location(game.active_player)
        #
        # if game.is_loser(game.active_player):
        #     return float('-inf') if maximizing_player else float('inf'), game.get_player_location(game.active_player)

        mv = None
        legal_moves = game.get_legal_moves(game.active_player)
        if len(legal_moves) == 0:
            return float('-inf') if maximizing_player else float('inf'), game.get_player_location(game.active_player)
        if depth == 0:
            mv = game.get_player_location(game.active_player)
            scr = self.score(game, game.active_player)
        else:
            if maximizing_player:
                scr = float('-inf')
                for m in legal_moves:
                    scr_local, _ = self.alphabeta(game.forecast_move(m), depth - 1, alpha, beta, False)
                    if scr_local >= scr:
                        scr = scr_local
                        mv = m
                        if scr >= beta:
                            break
                        alpha = max(alpha, scr)

            else:
                scr = float('inf')
                for m in legal_moves:
                    scr_local, _ = self.alphabeta(game.forecast_move(m), depth - 1, alpha, beta, True)
                    if scr >= scr_local:
                        scr = scr_local
                        mv = m

                        if scr <= alpha:
                            break
                        beta = min(beta, scr)

        if mv is None:
            print('We have a problem')
        return scr, mv
