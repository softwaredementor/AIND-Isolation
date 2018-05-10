"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    moves_legal = game.get_legal_moves(player)
    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    d = float((h - y)**2 + (w - x)**2)
    min = float("inf")
    for m in moves_legal:
        x1, y1 = m
        d1 = float((h - y1)**2 + (w - x1)**2)
        val = (d1 - d) / d
        if val <= min:
            min = val
    return 1/(1+min)

def custom_score_2(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if own_moves == 0:
        return float("-inf")

    if opp_moves == 0:
        return float("inf")

    return float(own_moves/opp_moves - opp_moves/own_moves)


def custom_score_3(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    y1, x1 = game.get_player_location(game.get_opponent(player))
    d1 = float((h - y)**2 + (w - x)**2)
    d2 = float((h - y1)**2 + (w - x1)**2)
    return float(d2/d1)

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        self.time_left = time_left
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def time_remaining(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def _min_value(self, game, depth):
        self.time_remaining()
        if self._is_terminal(game, depth):
            return self.score(game, self)
        min_val = float("inf")
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            forecast = game.forecast_move(move)
            min_val = min(min_val, self._max_value(forecast, depth - 1))
        return min_val

    def _max_value(self, game, depth):
        self.time_remaining()
        if self._is_terminal(game, depth):
            return self.score(game, self)
        max_val = float("-inf")
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            forecast = game.forecast_move(move)
            max_val = max(max_val, self._min_value(forecast, depth - 1))
        return max_val

    def _is_terminal(self, game, depth):
        """Helper method to check if we've reached the end of the game tree or
        if the maximum depth has been reached.
        """
        self.time_remaining()
        if len(game.get_legal_moves()) != 0 and depth !=0:
            return False
        return True

    def minimax(self, game, depth):
        self.time_remaining()
        best_score = float("-inf")
        best_move = None

        moves_legal = game.get_legal_moves()

        if not moves_legal:
            return (-1, -1)
        for m in moves_legal:
            v = self._min_value(game.forecast_move(m), depth - 1)
            if v > best_score:
                best_score = v
                best_move = m
        return best_move


class AlphaBetaPlayer(IsolationPlayer):
    """
    Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        self.time_left = time_left
        legal_moves = game.get_legal_moves()
        if len(legal_moves) > 0:
            best_move = legal_moves[0]
        else:
            best_move = (-1, -1)
        try:
            depth = 1
            while True:
                current_move = self.alphabeta(game, depth)
                if current_move == (-1, -1):
                    return best_move
                else:
                    best_move = current_move
                depth += 1
        except SearchTimeout:
            return best_move
        return best_move

    def time_remaining(self):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

    def _min_value(self, game, depth, alpha, beta):
        self.time_remaining()
        if self._is_terminal(game, depth):
            return self.score(game, self)
        min_val = float("inf")
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            forecast = game.forecast_move(move)
            min_val = min(min_val, self._max_value(forecast, depth - 1, alpha, beta))
            if min_val <= alpha:
                return min_val
            beta = min(beta, min_val)
        return min_val

    def _max_value(self, game, depth, alpha, beta):
        self.time_remaining()
        if self._is_terminal(game, depth):
            return self.score(game, self)
        max_val = float("-inf")
        legal_moves = game.get_legal_moves()
        for move in legal_moves:
            forecast = game.forecast_move(move)
            max_val = max(max_val, self._min_value(forecast, depth - 1, alpha, beta))
            if max_val >= beta:
                return max_val
            alpha = max(alpha, max_val)
        return max_val

    def _is_terminal(self, game, depth):
        self.time_remaining()
        if len(game.get_legal_moves()) != 0 and depth > 0:
            return False
        return True

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        self.time_remaining()
        best_score = float("-inf")
        best_move = (-1, -1)

        moves_legal = game.get_legal_moves()
        if not moves_legal:
            return best_move
        for m in moves_legal:
            v = self._min_value(game.forecast_move(m), depth - 1, alpha, beta)
            if v > best_score:
                best_score = v
                best_move = m
            # if v >= beta:
            #     return v
            alpha = max(alpha, v)
        return best_move
