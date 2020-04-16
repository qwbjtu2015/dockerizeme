# Author: Kyle Kastner
# License: BSD 3-Clause

# Vectorized, alternating update CFR+ for Kuhn poker
# For specific algorithmic details see
# http://jeskola.net/cfrp_mt.pdf
# https://pdfs.semanticscholar.org/ed52/8594465e96f51fd9a3cf00401bfba20597fa.pdf

# An alternate overview using scalar, simultaneous updates
# https://justinsermeno.com/posts/cfr/ 

# Reference code
# http://www.jeskola.net/cfr/demo/solve.js
# http://www.jeskola.net/cfr/demo/algorithms/cfr.js

# Compare the resulting table with theoretically optimal values 
# https://github.com/Limegrass/Counterfactual-Regret-Minimization/blob/notes/Learning_to_Play_Poker_using_CounterFactual_Regret_Minimization.pdf

# Local references for various algorithms
# cfr: http://poker.cs.ualberta.ca/publications/NIPS07-cfr.pdf 
# cfrbr: http://www.johanson.ca/publications/poker/2012-aaai-cfr-br/2012-aaai-cfr-br.pdf
# cfrplus: https://arxiv.org/abs/1407.5042
# cfrplus: http://poker.cs.ualberta.ca/publications/2015-ijcai-cfrplus.pdf
# cfrplusavg: https://www.tngtech.com/fileadmin/Public/Images/BigTechday/BTD10/Folien/Folien_MichaelSchubert.pdf 
# cfrplusavg: https://arxiv.org/abs/1811.00164, section 2.2 (also called linear cfr)
# cfrplusavg2: variant with squared iterations, https://arxiv.org/abs/1809.04040
# lcfr: https://arxiv.org/abs/1809.04040
# dcfr: https://arxiv.org/abs/1809.04040
# cfu: http://www.poker-ai.org/phpbb/viewtopic.php?f=25&t=2852
# dcfu: https://jeskola.net/cfr/dcfu/
# dcfu: https://jeskola.net/cfr/avgstr/
# xfp: http://www0.cs.ucl.ac.uk/staff/d.silver/web/Publications_files/fsp.pdf
# p_prune: comments of https://media.nips.cc/nipsbooks/nipspapers/paper_files/nips28/reviews/1198.html

# TODO:
# lazycfr: https://arxiv.org/abs/1810.04433
# lazycfr+: https://arxiv.org/abs/1810.04433

# Additional references:
# Video overview from Noam Brown
# https://www.microsoft.com/en-us/research/video/ai-for-imperfect-information-games-beating-top-humans-in-no-limit-poker/

# Great interview with Tuomas Sandholm with Lex Fridman
# https://www.youtube.com/watch?v=b7bStIQovcY

# Detailed discussion of Libratus from Noam Brown
# https://www.youtube.com/watch?v=2dX0lwaQRX0

# Lossless abstraction slides
# https://cs.uwaterloo.ca/~klarson/teaching/F08-886/talks/Fabien.pdf

# Lossless abstraction paper 
# http://www.cs.cmu.edu/~gilpin/papers/extensive.JACM.pdf

# C++ code from Marc Lanctot , bluff11.zip here
# http://mlanctot.info/

# Pruning talk from Noam Brown
# https://vimeo.com/238229930

# A nice overview of poker rules and concepts
# http://gmurru.altervista.org/public/documents/report-poker.pdf

# Interview on the '17 NeurIPS paper on nested subgame solving
# https://medium.com/this-week-in-machine-learning-ai/safe-and-nested-subgame-solving-for-imperfect-information-games-with-tuomas-sandholm-nips-best-c9c664197e88

# Tutorial post
# https://int8.io/counterfactual-regret-minimization-for-poker-ai/

# Multiplayer games
# http://webdocs.cs.ualberta.ca/~duane/publications/pdf/2010aamas.pdf

# Overview and other applications
# http://webdocs.cs.ualberta.ca/~hayward/670gga/jem/burch.pdf

# Detailed tutorial
# http://cs.gettysburg.edu/~tneller/modelai/2013/cfr/cfr.pdf

# Course writeup with detailed notes and some code
# https://github.com/Limegrass/Counterfactual-Regret-Minimization/blob/notes/Learning_to_Play_Poker_using_CounterFactual_Regret_Minimization.pdf

# PhD theses
# http://poker.cs.ualberta.ca/publications/Burch_Neil_E_201712_PhD.pdf
# http://mlanctot.info/files/papers/PhD_Thesis_MarcLanctot.pdf
# http://richardggibson.appspot.com/static/work/thesis-phd/thesis-phd-paper.pdf

# MSc theses
# http://poker.cs.ualberta.ca/publications/Davis_Trevor_R_201506_MSc.pdf

# Explanation of CFR 
# https://www.quora.com/What-is-an-intuitive-explanation-of-counterfactual-regret-minimization

# Demo discussion of Liberatus
# https://www.cs.cmu.edu/~noamb/papers/17-IJCAI-Libratus.pdf

# Demo discussion of older bot
# https://www.cs.cmu.edu/~sandholm/BabyTartanian8.ijcai16demo.pdf

# Discussion of some work on poker bot forums
# http://www.poker-ai.org/phpbb/viewtopic.php?f=25&t=2852

# Recent integrations with deep learning
# https://arxiv.org/pdf/1809.03057.pdf
# https://arxiv.org/abs/1811.00164
# https://openreview.net/forum?id=Bkeuz20cYm

# Source code for MATLAB including iqph solver...
# http://www.cs.cmu.edu/~ggordon/poker/
# http://www.cs.cmu.edu/~ggordon/poker/source/

import numpy as np

N_CARDS = 3 # possible cards
global NODES_TOUCHED
NODES_TOUCHED = 0
global ITERATIONS
ITERATIONS = 0

RANDOM_STATE = np.random.RandomState(1122)

def get_nodes_touched():
   return NODES_TOUCHED

def increment_nodes_touched():
    global NODES_TOUCHED
    NODES_TOUCHED += 1

def get_iteration_count():
    return ITERATIONS

def increment_iterations():
    global ITERATIONS
    ITERATIONS += 1

class InformationSet(object):
    def __init__(self, key, action_names):
        self.key = key
        self.action_names = action_names
        self.n_actions = len(action_names)

        self.regret_sum = [np.zeros(self.n_actions) for i in range(N_CARDS)]
        self.strategy_sum = [np.zeros(self.n_actions) + 1. / float(self.n_actions) for i in range(N_CARDS)]
        self.best_response_actions = [None for i in range(N_CARDS)]

    def get_normalized_strategy(self, hand):
        return normalize(self.strategy_sum[hand])

    def get_current_strategy(self, hand):
        total = 0.
        for a in range(self.n_actions):
            if self.regret_sum[hand][a] > 0:
                total += self.regret_sum[hand][a]

        if total > 0:
            s = 0. * self.regret_sum[hand]
            for a in range(self.n_actions):
                s[a] = 0 if self.regret_sum[hand][a] < 0. else self.regret_sum[hand][a] / float(total)
        else:
            s = np.zeros(self.n_actions) + 1. / float(self.n_actions)
        return s

    def get_average_strategy(self, hand):
        total = 0.
        total = sum(self.strategy_sum[hand])
        if total > 0:
            s = self.strategy_sum[hand] / float(total)
        else:
            s = np.zeros(self.n_actions) + 1. / float(self.n_actions)
        return s

    def __repr__(self):
        # show a summarized prob table as the repr
        name = self.key.ljust(4)
        s = [self.get_average_strategy(i) for i in range(N_CARDS)]
        full = ""
        for card in range(N_CARDS):
            joined = ""
            for a in range(self.n_actions):
                joined += "{}:{:03.2f},".format(self.action_names[a], s[card][a])
            joined = joined[:-1]
            full += " {} {} ".format(card_str(card), joined)
        return name + " |" + str(full)

def deal_hands(n_cards=N_CARDS, random_state=RANDOM_STATE):
    h = random_state.choice(n_cards, size=2, replace=False)
    return [[h[0]], [h[1]]]

def normalize(s):
    p_sum = 0.
    for i in range(len(s)):
        p_sum += s[i]

    r = np.zeros(len(s))
    if p_sum > 0:
        r = s / float(p_sum)
    else:
        r += 1. / float(len(s))
    return r

def copy_or_add(copy_flag, a, b):
    assert len(a) == len(b)
    if copy_flag == 1:
        for i in range(len(a)):
            a[i] = b[i]
    else:
        for i in range(len(a)):
            a[i] += b[i]

def is_terminal(history):
    # returns True if history is an end state
    keys = []
    keys += [tuple(["c", "c"])]
    keys += [tuple(["c", "b", "c"])]
    keys += [tuple(["c", "b", "b"])]
    keys += [tuple(["b", "c"])]
    keys += [tuple(["b", "b"])]
    possibilities = {k: True for k in keys}
    return tuple(history) in possibilities

def terminal_util(i_map, player, history, o_p, update_player):
    ev = 0. * o_p
    if (history == ["c", "b", "c"]
        or history == ["b", "c"]):
        # fold - payoff is 1 if other player folded, else we lost 1
        payoff = 1. if player == update_player else -1.
        return (sum(o_p) - o_p) * payoff
    elif history == ["c", "c"]:
        # showdown, no bets
        payoff = 1.
    else:
        assert(history == ["c", "b", "b"]
               or history == ["b", "b"])
        payoff = 2.
    # Exploit structure to make it O(n) to calculate all pairs
    # instead of n^2
    # This cleverness comes from here
    # http://www.jeskola.net/cfr/demo/solve.js
    sum_o_p = 0.
    for i in range(len(o_p)):
        ev[i] = sum_o_p * float(payoff)
        sum_o_p += o_p[i]
    sum_o_p = 0.
    for i in reversed(range(len(o_p))):
       ev[i] -= sum_o_p * float(payoff)
       sum_o_p += o_p[i]
    return ev

def payoff_util(i_map, player, history, hands, update_player):
    if (history == ["c", "b", "c"]
        or history == ["b", "c"]):
        # fold - payoff is 1 if other player folded, else we lost 1
        payoff = 1.
        return payoff if player == update_player else -payoff
    elif history == ["c", "c"]:
        # showdown, no bets
        payoff = 1.
    else:
        assert(history == ["c", "b", "b"]
               or history == ["b", "b"])
        payoff = 2.
    # payoffs are relative to the player we are currently updating
    return payoff if hands[update_player][0] > hands[update_player ^ 1][0] else -payoff

def card_str(card):
    # print the name of the card
    if card == 0:
        return "J"
    elif card == 1:
        return "Q"
    return "K"

def get_info_set(i_map, player, history, o_p, update_player):
    key = "_".join(history)
    info_set = None

    if key not in i_map:
        info_set = InformationSet(key, action_names=["c", "b"])
        i_map[key] = info_set
        return info_set

    return i_map[key]

def set_best_response_actions(i_map, player, history, o_p, update_player):
    if is_terminal(history):
        return terminal_util(i_map, player, history, o_p, update_player)

    info_set = get_info_set(i_map, player, history, o_p, update_player)
    if player == update_player:
        next_history = history + [info_set.action_names[0]]
        ev = set_best_response_actions(i_map, player ^ 1, next_history, o_p, update_player)
        for i in range(len(o_p)):
           info_set.best_response_actions[i] = 0
        for a in range(info_set.n_actions)[1:]:
            next_history = history + [info_set.action_names[a]]
            u = set_best_response_actions(i_map, player ^ 1, next_history, o_p, update_player)
            for i in range(len(o_p)):
                if u[i] > ev[i]:
                   ev[i] = u[i]
                   info_set.best_response_actions[i] = a
    else:
        s = [info_set.get_current_strategy(i) for i in range(len(o_p))]
        ev = 0. * o_p
        for a in range(info_set.n_actions):
            new_o_p = 0. * o_p
            for i in range(len(o_p)):
                new_o_p[i] = s[i][a] * o_p[i]
            next_history = history + [info_set.action_names[a]]
            ev_br = set_best_response_actions(i_map, player ^ 1, next_history, new_o_p, update_player)
            copy_or_add(a == 0, ev, ev_br)
    return ev

def best_response(i_map, player, history, o_p, update_player):
    if is_terminal(history):
        return terminal_util(i_map, player, history, o_p, update_player)

    info_set = get_info_set(i_map, player, history, o_p, update_player)
    if player == update_player:
        next_history = history + [info_set.action_names[0]]
        ev = best_response(i_map, player ^ 1, next_history, o_p, update_player)
        for a in range(info_set.n_actions)[1:]:
            next_history = history + [info_set.action_names[a]]
            ev_a = best_response(i_map, player ^ 1, next_history, o_p, update_player)
            ev = np.maximum(ev, ev_a)
    else:
        new_o_p = 0. * o_p
        for a in range(info_set.n_actions):
            for i in range(len(o_p)):
                s = info_set.get_normalized_strategy(i)
                new_o_p[i] = s[a] * o_p[i]
            next_history = history + [info_set.action_names[a]]
            br = best_response(i_map, player ^ 1, next_history, new_o_p, update_player)
            if a == 0:
                ev = br
            else:
                ev += br
    return ev

def best_response_value(i_map, player, update_player):
    o_p = np.zeros(N_CARDS) + 1. / (N_CARDS - 1)
    history = []
    ev = best_response(i_map, player, history, o_p, update_player)
    return sum(ev) / float(N_CARDS)

def get_exploitability(i_map):
    br0 = best_response_value(i_map, 0, 0)
    br1 = best_response_value(i_map, 0, 1)
    return (br0 + br1) / float(2) * 1000

def cfr(i_map, player, history, o_p, update_player, p_prune=True):
    if is_terminal(history):
        return terminal_util(i_map, player, history, o_p, update_player)

    increment_nodes_touched()

    info_set = get_info_set(i_map, player, history, o_p, update_player)
    s = [info_set.get_current_strategy(i) for i in range(len(o_p))]
    ev = 0. * o_p

    if player == update_player:
        u = []
        for a in range(info_set.n_actions):
            next_history = history + [info_set.action_names[a]]
            ua = cfr(i_map, player ^ 1, next_history, o_p, update_player)
            u.append(ua)

        for i in range(len(o_p)):
            for a in range(info_set.n_actions):
                ev[i] += s[i][a] * u[a][i]

        for i in range(len(o_p)):
            for a in range(info_set.n_actions):
                info_set.regret_sum[i][a] += u[a][i] - ev[i]
    else:
        # for average strategy
        for i in range(len(o_p)):
            for a in range(info_set.n_actions):
                info_set.strategy_sum[i][a] += o_p[i] * s[i][a]

        for a in range(info_set.n_actions):
            next_history = history + [info_set.action_names[a]]
            new_o_p = 0. * o_p
            for i in range(len(o_p)):
                new_o_p[i] = s[i][a] * o_p[i]

            if p_prune and sum(new_o_p) <= 0.:
                ev_a = 0. * new_o_p
            else:
                ev_a = cfr(i_map, player ^ 1, next_history, new_o_p, update_player)
            copy_or_add(a == 0, ev, ev_a)
    return ev

def xfp(i_map, player, history, p, br_p, update_player):
    if is_terminal(history):
        return
    increment_nodes_touched()

    info_set = get_info_set(i_map, player, history, p, update_player)

    if player == update_player:
        iteration_count = get_iteration_count()
        s = [info_set.get_current_strategy(i) for i in range(len(p))]
        for a in range(info_set.n_actions):
            for i in range(len(p)):
                if br_p[i] != 0.:
                    abrs = 1. if info_set.best_response_actions[i] == a else 0.
                    info_set.regret_sum[i][a] += (abrs - info_set.regret_sum[i][a]) / (1. + p[i] * (iteration_count + 1))
                    info_set.strategy_sum[i][a] = info_set.regret_sum[i][a]
            new_p = 0. * p
            new_br_p = 0. * br_p
            for i in range(len(p)):
                abrs = 1. if info_set.best_response_actions[i] == a else 0.
                new_p[i] = s[i][a] * p[i]
                new_br_p[i] = abrs * br_p[i]
            next_history = history + [info_set.action_names[a]]
            xfp(i_map, player ^ 1, next_history, new_p, new_br_p, update_player)
    else:
        for a in range(info_set.n_actions):
            next_history = history + [info_set.action_names[a]]
            xfp(i_map, player ^ 1, next_history, p, br_p, update_player)

def cfrbr(i_map, player, history, o_p, update_player, p_prune=True):
    if is_terminal(history):
        return terminal_util(i_map, player, history, o_p, update_player)

    increment_nodes_touched()

    info_set = get_info_set(i_map, player, history, o_p, update_player)
    s = [info_set.get_current_strategy(i) for i in range(len(o_p))]
    ev = 0. * o_p

    if player == update_player:
        u = []
        for a in range(info_set.n_actions):
            next_history = history + [info_set.action_names[a]]
            ua = cfrbr(i_map, player ^ 1, next_history, o_p, update_player, p_prune=p_prune)
            u.append(ua)

        for i in range(len(o_p)):
            for a in range(info_set.n_actions):
                ev[i] += s[i][a] * u[a][i]

        for i in range(len(o_p)):
            for a in range(info_set.n_actions):
                info_set.regret_sum[i][a] += u[a][i] - ev[i]
    else:
        # for average strategy?
        for i in range(len(o_p)):
            for a in range(info_set.n_actions):
                info_set.strategy_sum[i][a] += o_p[i] * s[i][a]

        for a in range(info_set.n_actions):
            next_history = history + [info_set.action_names[a]]
            new_o_p = 0. * o_p
            for i in range(len(o_p)):
                new_o_p[i] = o_p[i] if a == info_set.best_response_actions[i] else 0.
            if p_prune and sum(new_o_p) <= 0.:
                ev_a = 0. * new_o_p
            else:
                ev_a = cfrbr(i_map, player ^ 1, next_history, new_o_p, update_player, p_prune=p_prune)
            copy_or_add(a == 0, ev, ev_a)
    return ev

def cfrplus(i_map, player, history, o_p, update_player, p_prune=True):
    if is_terminal(history):
        return terminal_util(i_map, player, history, o_p, update_player)

    increment_nodes_touched()

    info_set = get_info_set(i_map, player, history, o_p, update_player)
    s = [info_set.get_current_strategy(i) for i in range(len(o_p))]
    ev = 0. * o_p

    if player == update_player:
        u = []
        for a in range(info_set.n_actions):
            next_history = history + [info_set.action_names[a]]
            ua = cfrplus(i_map, player ^ 1, next_history, o_p, update_player)
            u.append(ua)

        for i in range(len(o_p)):
            for a in range(info_set.n_actions):
                ev[i] += s[i][a] * u[a][i]

        for i in range(len(o_p)):
            for a in range(info_set.n_actions):
                info_set.regret_sum[i][a] += u[a][i] - ev[i]
                info_set.regret_sum[i][a] = np.maximum(info_set.regret_sum[i][a], 0.)
    else:
        # for average strategy
        for i in range(len(o_p)):
            for a in range(info_set.n_actions):
                info_set.strategy_sum[i][a] += o_p[i] * s[i][a]

        for a in range(info_set.n_actions):
            next_history = history + [info_set.action_names[a]]
            new_o_p = 0. * o_p
            for i in range(len(o_p)):
                new_o_p[i] = s[i][a] * o_p[i]
            if p_prune and sum(new_o_p) <= 0.:
                ev_a = 0. * new_o_p
            else:
                ev_a = cfrplus(i_map, player ^ 1, next_history, new_o_p, update_player)
            copy_or_add(a == 0, ev, ev_a)
    return ev


def abgcfr(i_map, player, history, o_p, update_player, alpha=1., beta=1., gamma=1., p_prune=True):
    if is_terminal(history):
        return terminal_util(i_map, player, history, o_p, update_player)

    increment_nodes_touched()

    iteration_count = get_iteration_count()
    i_a = iteration_count ** alpha
    i_b = iteration_count ** beta
    pos_weight = i_a / float(i_a + 1.)
    neg_weight = i_b / float(i_b + 1.)
    gamma_weight = (iteration_count / float(iteration_count + 1.)) ** gamma

    info_set = get_info_set(i_map, player, history, o_p, update_player)
    s = [info_set.get_current_strategy(i) for i in range(len(o_p))]
    ev = 0. * o_p

    if player == update_player:
        u = []
        for a in range(info_set.n_actions):
            next_history = history + [info_set.action_names[a]]
            ua = abgcfr(i_map, player ^ 1, next_history, o_p, update_player, alpha=alpha, beta=beta, gamma=gamma)
            u.append(ua)

        for i in range(len(o_p)):
            for a in range(info_set.n_actions):
                ev[i] += s[i][a] * u[a][i]

        for i in range(len(o_p)):
            for a in range(info_set.n_actions):
                delta = u[a][i] - ev[i]
                if info_set.regret_sum[i][a] >= 0:
                    info_set.regret_sum[i][a] += pos_weight * delta
                else:
                    info_set.regret_sum[i][a] += neg_weight * delta
    else:
        # for average strategy
        for i in range(len(o_p)):
            for a in range(info_set.n_actions):
                info_set.strategy_sum[i][a] += gamma_weight * o_p[i] * s[i][a]

        for a in range(info_set.n_actions):
            next_history = history + [info_set.action_names[a]]
            new_o_p = 0. * o_p
            for i in range(len(o_p)):
                new_o_p[i] = s[i][a] * o_p[i]
            if p_prune and sum(new_o_p) <= 0.:
                ev_a = 0. * new_o_p
            else:
                ev_a = abgcfr(i_map, player ^ 1, next_history, new_o_p, update_player, alpha=alpha, beta=beta, gamma=gamma)
            copy_or_add(a == 0, ev, ev_a)
    return ev

def cfrplusavg(i_map, player, history, o_p, update_player, pow_v=1., p_prune=True):
    if is_terminal(history):
        return terminal_util(i_map, player, history, o_p, update_player)

    increment_nodes_touched()

    info_set = get_info_set(i_map, player, history, o_p, update_player)
    s = [info_set.get_current_strategy(i) for i in range(len(o_p))]
    ev = 0. * o_p

    if player == update_player:
        u = []
        for a in range(info_set.n_actions):
            next_history = history + [info_set.action_names[a]]
            ua = cfrplusavg(i_map, player ^ 1, next_history, o_p, update_player, pow_v=pow_v)
            u.append(ua)

        for i in range(len(o_p)):
            for a in range(info_set.n_actions):
                ev[i] += s[i][a] * u[a][i]

        for i in range(len(o_p)):
            for a in range(info_set.n_actions):
                info_set.regret_sum[i][a] += u[a][i] - ev[i]
                info_set.regret_sum[i][a] = np.maximum(info_set.regret_sum[i][a], 0.)
    else:
        iteration_count = get_iteration_count()
        weight = pow(iteration_count + 1, pow_v)
        # for average strategy
        for i in range(len(o_p)):
            for a in range(info_set.n_actions):
                info_set.strategy_sum[i][a] += weight * o_p[i] * s[i][a]

        for a in range(info_set.n_actions):
            next_history = history + [info_set.action_names[a]]
            new_o_p = 0. * o_p
            for i in range(len(o_p)):
                new_o_p[i] = s[i][a] * o_p[i]
            if p_prune and sum(new_o_p) == 0.:
                ev_a = 0. * new_o_p
            else:
                ev_a = cfrplusavg(i_map, player ^ 1, next_history, new_o_p, update_player, pow_v=pow_v)
            copy_or_add(a == 0, ev, ev_a)
    return ev

def cfu(i_map, player, history, o_p, update_player, p_prune=True):
    if is_terminal(history):
        return terminal_util(i_map, player, history, o_p, update_player)

    increment_nodes_touched()

    info_set = get_info_set(i_map, player, history, o_p, update_player)
    s = [np.argmax(info_set.regret_sum[i]) for i in range(len(o_p))]
    ev = 0. * o_p

    if player == update_player:
        u = []
        for a in range(info_set.n_actions):
            next_history = history + [info_set.action_names[a]]
            ua = cfu(i_map, player ^ 1, next_history, o_p, update_player)
            u.append(ua)

        for i in range(len(o_p)):
            for a in range(info_set.n_actions):
                ev[i] = u[s[i]][i]

        for i in range(len(o_p)):
            for a in range(info_set.n_actions):
                info_set.regret_sum[i][a] += u[a][i]
    else:
        # for average strategy
        for i in range(len(o_p)):
            if o_p[i] != 0:
                info_set.strategy_sum[i][s[i]] += 1

        for a in range(info_set.n_actions):
            next_history = history + [info_set.action_names[a]]
            new_o_p = 0. * o_p
            for i in range(len(o_p)):
                new_o_p[i] = o_p[i] if s[i] == a else 0.
            # pruning is pretty effective in this algorithm
            if p_prune and sum(new_o_p) == 0.:
                ev_a = 0. * new_o_p
            else:
                ev_a = cfu(i_map, player ^ 1, next_history, new_o_p, update_player)
            copy_or_add(a == 0, ev, ev_a)
    return ev

def escfr(i_map, player, history, hands, update_player, random_state=RANDOM_STATE):
    if is_terminal(history):
        return payoff_util(i_map, player, history, hands, update_player)

    increment_nodes_touched()

    info_set = get_info_set(i_map, player, history, hands, update_player)
    s = info_set.get_current_strategy(hands[player][0])

    if player == update_player:
        ev = 0.
        u = []
        for a in range(info_set.n_actions):
            next_history = history + [info_set.action_names[a]]
            ua = escfr(i_map, player ^ 1, next_history, hands, update_player)
            u.append(ua)
            ev += s[a] * u[a]

        for a in range(info_set.n_actions):
            # update_player and player are the same here
            info_set.regret_sum[hands[player][0]][a] += (u[a] - ev)
        return ev
    else:
        # for average strategy
        for a in range(info_set.n_actions):
            # update_player and player are NOT the same here
            info_set.strategy_sum[hands[player][0]][a] += s[a]

        which = random_state.choice(info_set.n_actions, p=s)
        next_history = history + [info_set.action_names[which]]
        return escfr(i_map, player ^ 1, next_history, hands, update_player)

def cfrplusavg2(i_map, player, history, o_p, update_player, p_prune=True):
    return cfrplusavg(i_map, player, history, o_p, update_player, pow_v=2., p_prune=p_prune)

def lcfr(i_map, player, history, o_p, update_player, p_prune=True):
    return abgcfr(i_map, player, history, o_p, update_player, alpha=1., beta=1., gamma=1., p_prune=p_prune)

def dcfr(i_map, player, history, o_p, update_player, alpha=1., beta=1., gamma=1., p_prune=True):
    return abgcfr(i_map, player, history, o_p, update_player, alpha=3./2., beta=0., gamma=2., p_prune=p_prune)

def main():
    N_ITERATIONS = 1000 # number of iterations for CFR
    i_map = {} # information sets aka decision nodes
    mc_names = ["escfr"]

    #alg = cfr
    #alg = cfrbr
    #alg = cfrplus
    #alg = lcfr
    #alg = dcfr
    #alg = cfrplusavg
    #alg = cfrplusavg2
    #alg = cfu
    #alg = escfr
    alg = xfp

    for it in range(N_ITERATIONS):
        history = []
        if alg.func_name == "cfrbr":
            player = 0
            update_player = 0
            o_p = np.ones(N_CARDS)
            set_best_response_actions(i_map, player, history, o_p, update_player)
            update_player = 1
            o_p = np.ones(N_CARDS)
            set_best_response_actions(i_map, player, history, o_p, update_player)
        player = 0
        update_player = 0
        if alg.func_name in mc_names:
            hands = deal_hands()
            alg(i_map, player, history, hands, update_player)
        elif alg.func_name == "xfp":
            p = np.ones(N_CARDS)
            brp = np.ones(N_CARDS)
            set_best_response_actions(i_map, player, history, p, update_player)
            alg(i_map, player, history, p, brp, update_player)
        else:
            o_p = np.ones(N_CARDS)
            alg(i_map, player, history, o_p, update_player)

        history = []
        player = 0
        update_player = 1
        if alg.func_name in mc_names:
            hands = deal_hands()
            alg(i_map, player, history, hands, update_player)
        elif alg.func_name == "xfp":
            p = np.ones(N_CARDS)
            brp = np.ones(N_CARDS)
            set_best_response_actions(i_map, player, history, p, update_player)
            alg(i_map, player, history, p, brp, update_player)
        else:
            o_p = np.ones(N_CARDS)
            alg(i_map, player, history, o_p, update_player)

        e = get_exploitability(i_map)
        increment_iterations()
        if (it % 100 == 0) or it == (N_ITERATIONS - 1):
            node_count = get_nodes_touched()
            #print("{} exploitability (milli-bb) @ iteration {} : {}".format(alg.func_name, it, e))
            print("{} exploitability (milli-bb) @ nodes {} : {}".format(alg.func_name, node_count, e))

    # format
    # previous_actions | card action1:prob1,action2:prob2

    # For example, the ideal setup from the reference
    # https://github.com/Limegrass/Counterfactual-Regret-Minimization/blob/notes/Learning_to_Play_Poker_using_CounterFactual_Regret_Minimization.pdf
    # 
    # root | J c:1-alpha,b:alpha
    # root | Q c:1,b:0
    # root | K c:1-3*alpha,b:3*alpha 
    #
    print(" ")
    print("Final strategy table:")
    for k in i_map.keys():
        print(i_map[k])
    from IPython import embed; embed(); raise ValueError()

if __name__ == "__main__":
    main()
