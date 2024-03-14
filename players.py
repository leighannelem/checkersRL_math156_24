import numpy as np
import pickle

class Player:
    def __init__(self, name, exp_rate=0.3):
        self.name = name # string
        self.states = []  # record all positions taken
        self.lr = 0.02
        # exp_rate is epsilon
        # exp_rate = 0.3 means 70% of the time agent will take a greedy action and
        # 30% of the time will take a random action
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value
        self.actions = []


    def getHash(self, board):
        boardHash = str(board.flatten())
        return boardHash
    
    def chooseAction(self, moves, current_board, symbol): 
        if np.random.uniform(0,1) <=  self.exp_rate: 
            # take random action
            idx = np.random.choice(len(moves))
            action = moves[idx] 
        else:
            # take greedy action
            # i think this is the Q learning portion ... will need to explain
            value_max = -999 
            for m in moves:
                next_board = current_board.copy() 

                # the following is copied from updateState
                # TODO: WHEN WE CHANGE UPDATE STATE WILL ALSO NEED TO CHANGE THIS
                start_row = m[0]
                start_col = m[1]
                end_row = m[2]
                end_col = m[3]
                next_board[start_row][start_col] = 0
                # update new location, accounting for if it turns into a king
                if symbol == 1 and end_row == 0:
                    next_board[end_row][end_col] = 2 #*symbol
                elif symbol == -1 and end_row == 7:
                    next_board[end_row][end_col] = -2 
                else:
                    next_board[end_row][end_col] = symbol
                if abs(end_row - start_row) == 2:
                    # remove the piece that was jumped
                    next_board[(start_row + end_row) // 2][(start_col + end_col) // 2] = 0
                    # no double jumps (for now) <3

                next_boardHash = self.getHash(next_board)
                if self.states_value.get(next_boardHash) is None:
                    value = 0
                else:
                    value = self.states_value.get(next_boardHash)
                # print("value", value)
                if value >= value_max:
                    value_max = value
                    action = m
        # print("{} takes action {}".format(self.name, action))
        self.actions.append(action)
        return action

    
    # append a hash state
    def addState(self, state):
        self.states.append(state)

    # at the end of game, backpropogate and update states value and update the Q-Table
    def feedReward(self, reward):
        for i in reversed(range(len(self.states))):
            state = self.states[i]
            action = self.actions[i]  # assuming self.actions is a list of actions taken
            state_action = (state, action)
            if self.states_value.get(state_action) is None:
                self.states_value[state_action] = 0
            self.states_value[state_action] += self.lr*(self.decay_gamma*reward - self.states_value[state_action])
            reward = self.states_value[state_action]
    
    def calculate_q_value_difference(self, old_q_values):
        total_difference = 0
        for state_action, q_value in self.states_value.items():
            if state_action in old_q_values:
                total_difference += abs(old_q_values[state_action] - q_value)
            else:
                total_difference += abs(q_value)
        return total_difference

    def reset(self):
        self.states = []

    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.state_value = pickle.load(fr)
        fr.close()
            

class HumanPlayer:
    def __init__(self,name):
        self.name = name

    def chooseAction(self, moves):
        while True:
            start_row = int(input("Input the starting row of your piece:"))
            start_col = int(input("Input the starting col of your piece:"))
            end_row = int(input("Input the end row of your piece:"))
            end_col = int(input("Input the end col of your piece:"))
            action = (start_row, start_col, end_row, end_col)
            if action in moves:
                # i.e. if it is legal
                # lowkey this might be wrong idk
                # make sure that it checks if the move is legal
                return action

    # append hash state
    def addState(self, state):
        pass

    # at the end of the game, backpropogate and update states value
    def feedReward(self, reward):
        pass
    
    def reset(self):
        pass

class RandomPlayer:
    def __init__(self,name):
        self.name = name

    def chooseAction(self, moves):
        # take random action
        idx = np.random.choice(len(moves))
        action = moves[idx]
        return action

    # append hash state
    def addState(self, state):
        pass

    # at the end of the game, backpropogate and update states value
    def feedReward(self, reward):
        pass
    
    def reset(self):
        pass
            