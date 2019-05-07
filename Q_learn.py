# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import collections

from SwingyMonkey import SwingyMonkey

SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400

# Hyperparameters - Change these values for experimentation
EPSILON = 0.001  # Start e-greedy factor; baseline = 0.001
ETA = 0.3  # Our learning rate; baseline = 0.3
RANDOM_PROB = 0.4  # When we do random choice: what probability do we not jump; baseline = 0.4
GAMMA = 1  # Discount factor; baseline = 1
WIDTH_BIN_SIZE = 100  # Pixels per bin; baseline = 100
HEIGHT_BIN_SIZE = 80  # baseline = 80

class Learner(object):
    def __init__(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.epoch = 1
        self.Q = collections.Counter()

        # number of actions a that has been taken from state s
        self.A_at_S = collections.Counter()

        self.high_score = 0
        self.iter = 0
        self.gravity = None
        self.gravity_sure = False

    def reset(self):
        self.last_state = None
        self.last_action = None
        self.last_reward = None
        self.gravity = None
        self.gravity_sure = False
        self.epoch += 1
        self.iter = 0

    # State Feature Selection: Extract features
    def tuple_generate(self, state, gravity):

        # Feature 1 - Horizontal Distance from Monkey to Tree
        tree_dist = state['tree']['dist']
        if tree_dist >= 500:
            tree_dist = 499
        val_dist = tree_dist // WIDTH_BIN_SIZE
        if val_dist < 0:
            val_dist = 0

        # Uncomment lines 57-60 and comment lines 49-54 to do Trial 3 for State Feature Selection
        # val_dist = 0
        # tree_dist = abs(state['tree']['dist'])
        # if tree_dist < WIDTH_BIN_SIZE:
        #     val_dist = 1

        # Feature 2 - Vertical Distance between Monkey Top and Tree Top
        val_top_diff = (state["tree"]["top"] - state["monkey"]["top"]) // HEIGHT_BIN_SIZE

        # Feature 3 - Danger Indicator
        val_danger = 1
        if state['monkey']['top'] > 350:
            val_danger = 0
        elif state['monkey']['bot'] < 50:
            val_danger = 2

        # Uncomment line 73 and comment lines 66-70 to do Trial 4 for State Feature Selection
        # val_danger = (state['monkey']['top'] + state['monkey']['bot'])//2 // HEIGHT_BIN_SIZE

        # Uncomment lines 76-80 and comment lines 66-70 to do Trial 5 for State Feature Selection
        # val_danger = 1
        # if state['monkey']['top'] > 375:
        #     val_danger = 0
        # elif state['monkey']['bot'] < 25:
        #     val_danger = 2

        # Feature 4 - Discretized Monkey Velocity
        mon_vel = state["monkey"]["vel"]
        if mon_vel > 15:
            val_vel = 1
        elif mon_vel > -5:
            val_vel = 0
        elif mon_vel > -25:
            val_vel = -1
        else:
            val_vel = -2
        # Uncomment line 93 and comment lines 83-91 to do Trials 2-5 for State Feature Selection
        val_vel = state["monkey"]["vel"] // 15

        # Note: Feature 5 is Gravity (passed in as parameter)
        return (val_dist, val_top_diff, val_vel, val_danger, gravity)

    def action_callback(self, state):

        if state['score'] > self.high_score:
            print(self.epoch, '---', state['score'])
            self.high_score = state['score']

        # Infer gravity (Feature 5)
        if self.gravity == None and self.last_state != None:
            grav_temp = int(self.last_state['monkey']['vel'] - state['monkey']['vel'])
            if (grav_temp == 1 or grav_temp == 4):
                self.gravity = grav_temp
                if self.last_action == 0:
                    self.gravity_sure = True
            else:
                self.gravity = 4

        # Account for possibility of mis-update
        if self.last_state != None and self.gravity_sure == False:
            if self.last_action == 0:
                grav_temp = int(self.last_state['monkey']['vel'] - state['monkey']['vel'])
                if (grav_temp == 1 or grav_temp == 4):
                    self.gravity = grav_temp
                    self.gravity_sure = True

        self.iter += 1

        # Format to feed to Counter()
        state_tuple = self.tuple_generate(state, self.gravity)
        next_action = None

        if self.Q[state_tuple, 1] > self.Q[state_tuple, 0]:
            next_value = self.Q[state_tuple, 1]
            next_action = 1
        else:
            next_value = self.Q[state_tuple, 0]
            next_action = 0

        # Update Q
        if self.last_state is not None:
            last_state_tuple = self.tuple_generate(self.last_state, self.gravity)
            Q_last = self.Q[last_state_tuple, self.last_action]
            self.Q[last_state_tuple, self.last_action] =\
                Q_last - ETA * (Q_last - self.last_reward - GAMMA * next_value)

        # Epsilon greedy
        zero_times = self.A_at_S[state_tuple, 0]
        one_times = self.A_at_S[state_tuple, 1]
        total_times = zero_times + one_times
        if total_times == 0:
            eps = EPSILON
        else:
            eps = EPSILON * ((1 / total_times) ** 0.8)
        if npr.rand() < eps:
            if zero_times == 0:
                next_action = 0
            elif one_times < zero_times and one_times == 0:
                next_action = 1
            elif npr.rand() < RANDOM_PROB:
                next_action = 0
            else:
                next_action = 1

        self.last_state = state
        self.last_action = next_action
        self.A_at_S[state_tuple, next_action] += 1
        return next_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward

def run_games(learner, hist, iters=100, t_len=100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length=t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        # Reset the state of the learner.
        learner.reset()

    pg.quit()
    return


if __name__ == '__main__':

    # Select agent.
    agent = Learner()

    # Empty list to save history.
    hist = []

    # Run games.
    run_games(agent, hist, 100, 0)

    # Save history. Note: CHANGE NAME of npy file for each trial (to avoid overwriting data)
    np.save('hist', np.array(hist))
