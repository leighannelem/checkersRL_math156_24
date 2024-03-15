import matplotlib.pyplot as plt
from keras import Sequential, regularizers
from keras.layers import Dense
import tensorflow as tf
import numpy as np
import random
from tqdm import tqdm

from state import State
from players import Player, HumanPlayer, RandomPlayer

def concatenate(array1, array2):
    for i in range(len(array2)):
        array1.append(array2[i])
    return array1  


def GetModel(game, playtype, rounds=100):
    LEARNING_RATE = 0.5
    DISCOUNT_FACTOR = 0.95
    EXPLORATION = 0.3

    won = 0
    lost = 0
    draw = 0
    winrates = []
    loserates = []
    drawrates = []

    reward = 0
    rewards = 0
    cumulative_rewards = []

    states = []
    labels = np.zeros(1) # rewards

    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=64)) 
    model.add(Dense(16, activation='relu',  kernel_regularizer=regularizers.l2(0.1)))

    model.add(Dense(1, activation='relu',  kernel_regularizer=regularizers.l2(0.1)))
    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=["acc"])

    for i in tqdm(range(rounds)):
        #print("won", won, ", lost", lost, ", draw", draw)
        states = []
        for g in range(10):
            temp_states = []
            player = 1
            count = 0
            while True:
                count += 1
                end2 = 0
                temptemp_states = [] # i added this
                if count > 500:
                    draw += 1
                    break
                else:
                    if (player == 1):
                        game.playerSymbol = 1

                        moves = game.getLegalMoves(player)
                        if (len(moves) == 0): 
                            end2 = -1 # win goes to the other player
                            continue

                        if (random.random() < EXPLORATION):
                            idx = np.random.choice(len(moves))
                            move = moves[idx]
                            
                        else:
                            for m in moves:
                                features = game.getHashNew(m)
                                temptemp_states.append(features)
                            temptemp_tensor = tf.constant(temptemp_states)
                            scores = model.predict_on_batch(temptemp_tensor)
                            idx = np.argmax(scores)
                            move = moves[idx]

                        # Append features of the selected move to temp_data
                        tab = game.getHashNew(move)
                        game.updateState(move)
                        temp_states.append(tab)

                    elif (player == -1):
                        game.playerSymbol = -1

                        if (playtype == "self"):
                            moves = game.getLegalMoves(player)
                            if (len(moves) == 0): 
                                end2 = 1 
                                continue
                            if (random.random() < EXPLORATION):
                                idx = np.random.choice(len(moves))
                                game.updateState(moves[idx])
                            else:
                                for move in moves:
                                    features = game.getHashNew(move)
                                    temptemp_states.append(features)
                                temptemp_tensor = tf.constant(temptemp_states)
                                scores = model.predict_on_batch(temptemp_tensor)
                                idx = np.argmax(scores)
                                best_move = moves[idx]
                                game.updateState(best_move)
                        elif (playtype == "random"):
                            moves = game.getLegalMoves(player)
                            if (len(moves) == 0):
                                end2 = 1
                                continue
                            idx = np.random.choice(len(moves))
                            game.updateState(moves[idx])

                end = game.winner()

                if (end == 1 or end2 == 1): # if player 1 wins
                    won += 1
                    if len(temp_states[1:]) == 0:
                        game.reset()
                        break
                    reward = 10
                    temp_tensor = tf.constant(temp_states[1:])
                    old_prediction = model.predict_on_batch(temp_tensor)
                    optimal_future_value = np.ones(old_prediction.shape)
                    temp_labels = old_prediction + LEARNING_RATE * (reward + DISCOUNT_FACTOR * optimal_future_value - old_prediction )
                    states = concatenate(states, temp_states[1:])
                    labels = np.vstack((labels, temp_labels))
                    game.reset()
                    break
                elif (end == -1 or end2 == -1):
                    lost += 1
                    if len(temp_states[1:]) == 0:
                        game.reset()
                        break
                    reward = -10
                    temp_tensor = tf.constant(temp_states[1:])
                    old_prediction = model.predict_on_batch(temp_tensor)
                    optimal_future_value = np.ones(old_prediction.shape)
                    temp_labels = old_prediction + LEARNING_RATE * (reward + DISCOUNT_FACTOR * optimal_future_value - old_prediction )
                    states = concatenate(states, temp_states[1:])
                    labels = np.vstack((labels, temp_labels))
                    game.reset()
                    break

                player = -player
            rewards += reward
        states = tf.constant(states)
        #if (not isDraw): 
        model.fit(states[1:], labels[2:], epochs=16, batch_size=256, verbose=0)
        labels = np.zeros(1)

        winrate = int((won)/(won+draw+lost)*100)
        loserate = int((lost)/(won+draw+lost)*100)
        drawrate = int((draw)/(won+draw+lost)*100)
        winrates.append(winrate)
        loserates.append(loserate)
        drawrates.append(drawrate)
        cumulative_rewards.append(rewards)
        #model.save("model.keras")
    indices = list(range(len(winrates)))
    print("won", won/10, ", lost", lost/10, ", draw", draw/10)
    plt.plot(indices, winrates, marker='o', linestyle='-', label = "win rate")
    plt.plot(indices, loserates, marker='o', linestyle='-', label = "lose rate")
    plt.plot(indices, drawrates, marker='o', linestyle='-', label = "draw rate")
    plt.title('Rates of results')
    plt.xlabel('generations')
    plt.ylabel('wins [%]')
    plt.legend()
    plt.show()

    plt.plot(indices, cumulative_rewards, marker='o', linestyle='-')
    plt.title('Cumulative Rewards over Time')
    plt.xlabel('Iterations')
    plt.ylabel('Cumulative Rewards')
    plt.show()

