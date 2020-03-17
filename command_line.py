#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import time

N_STATES = 6
MAX_EPISODE = 40
ACTIONS = ['left', 'right']
ALPHA = 0.1    #learning rate
GAMMA = 0.9
EPSILON = 0.9    #Greedy policy
FRESH_TIME = 0.01


def create_q_table(n_states,actions):
    """create a q table

    :S: states
    :A: actions
    :returns: a states-actions table

    """
    q_table = pd.DataFrame(np.zeros((n_states, len(actions))),columns = actions)
    print(q_table)
    return q_table



def choose_action(state, q_table):
    """Use Epsilon Greedy to choose action

    :S: current state
    :q_table: TODO
    :returns: action

    """
    state_actions = q_table.iloc[state,:]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        action = np.random.choice(ACTIONS)
    else:
        action = state_actions.idxmax()
    return action


def get_env_feedback(state, action):
    """TODO: Docstring for get_env_feedback.

    :state: TODO
    :action: TODO
    :returns: next state and reward

    """
    if action == 'right':
        if state == N_STATES - 2:
            S_next = 'end'
            R = 1
        else:
            R = 0
            S_next = state + 1
    else:
        R = 0
        if state == 0:
            S_next = state
        else:
            S_next = state - 1
    return S_next, R

def update_env(S, episode, step_counter):
    # This is how environment be updated
    env_list = ['-']*(N_STATES-1) + ['X']   # '---------T' our environment
    if S == 'end':
        interaction = 'Episode %s: total_steps = %s' % (episode+1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(1)
        print('\r                                ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def RL_BRAIN():
    """Q_Learning Algorithm
    :returns: q_table

    """
    q_table = create_q_table(N_STATES,ACTIONS)
    for episode in range(MAX_EPISODE):
        step_counter = 0
        S = 0    # state
        is_end = False
        update_env(S, episode, step_counter)
        while not is_end:
            A = choose_action(S,q_table)
            S_next, R = get_env_feedback(S,A)
            q_predict = q_table.loc[S, A]
            if S_next != 'end':
                q_target = R + GAMMA * q_table.loc[S_next, :].max()
            else:
                q_target = R
                is_end = True
            q_table.loc[S,A] += ALPHA * (q_target - q_predict)
            S = S_next
            step_counter += 1
            update_env(S, episode,step_counter)
    return q_table



if __name__ == "__main__":
    q_table = RL_BRAIN()
    print('\r\nQ-table:\n')
    print(q_table)
