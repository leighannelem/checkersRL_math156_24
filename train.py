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


def GetModel(game, rounds=100, learning_rate=0.5, discount_factor = 0.95):
    won = 0
    lost = 0
    draw = 0
    winrates = []

    data = []
    labels = np.zeros(1)

    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=4)) 
    model.add(Dense(16, activation='relu',  kernel_regularizer=regularizers.l2(0.1)))

    model.add(Dense(1, activation='relu',  kernel_regularizer=regularizers.l2(0.1)))
    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=["acc"])

    for i in tqdm(range(rounds)):
        data = []
        isDraw = False
        for g in range(10):
            temp_data = []
            player = 1
            count = 0
            while True:
                count += 1
                end2 = 0
                if count > 500:
                    draw += 1
                    isDraw = True
                    break
                else:
                    if (player == 1):
                        game.playerSymbol = 1
                        moves = game.availableMoves() #p1
                        Leaf = tf.zeros((len(moves), 4)) 
                        for l in range(len(moves)):
                            tensor = moves[l] # the lth available move
                            Leaf = tf.tensor_scatter_nd_update(Leaf, [[l]], [tensor])
                        scores = model.predict_on_batch(Leaf)
                        if (len(moves) == 0): 
                            end2 = -1 # win goes to the other player
                            continue
                        i = np.argmax(scores)
                        game.updateState(moves[i]) 
                        tab = moves[i]
                        temp_data.append(tab) 
                    elif (player == -1):
                        game.playerSymbol = -1
                        moves = game.availableMoves()
                        Leaf = tf.zeros((len(moves), 4)) 
                        for l in range(len(moves)):
                            tensor = moves[l] # the lth available move
                            Leaf = tf.tensor_scatter_nd_update(Leaf, [[l]], [tensor])
                        scores = model.predict_on_batch(Leaf)
                        if (len(moves) == 0): 
                            end2 = 1 
                            continue
                        i = np.argmax(scores)
                        game.updateState(moves[i]) #switches to p2
                        #tab = moves[i]
                        #temp_data.append(tab) 

                end = game.winner()

                if (end == 1 or end2 == 1): # if player 1 wins
                    won += 1
                    reward = 10
                    temp_tensor = tf.constant(temp_data[1:])
                    old_prediction = model.predict_on_batch(temp_tensor)
                    optimal_future_value = np.ones(old_prediction.shape)
                    temp_labels = old_prediction + learning_rate * (reward + discount_factor * optimal_future_value - old_prediction )
                    data = concatenate(data, temp_data[1:])
                    labels = np.vstack((labels, temp_labels))
                    game.reset()
                    break
                elif (end == -1 or end2 == -1):
                    lost += 1
                    reward = -10
                    temp_tensor = tf.constant(temp_data[1:])
                    old_prediction = model.predict_on_batch(temp_tensor)
                    optimal_future_value = np.ones(old_prediction.shape)
                    temp_labels = old_prediction + learning_rate * (reward + discount_factor * optimal_future_value - old_prediction )
                    data = concatenate(data, temp_data[1:])
                    labels = np.vstack((labels, temp_labels))
                    game.reset()
                    break

                
                player = -player
        data = tf.constant(data)
        if (not isDraw): model.fit(data[1:], labels[2:], epochs=16, batch_size=256, verbose=0)
        labels = np.zeros(1)

        winrate = int((won)/(won+draw+lost)*100)
        winrates.append(winrate)
        model.save("model.keras")
    indices = list(range(len(winrates)))
    print("won", won/10, ", lost", lost/10, ", draw", draw/10)
    plt.plot(indices, winrates, marker='o', linestyle='-')
    plt.title('Rates of win')
    plt.xlabel('generations')
    plt.ylabel('wins [%]')
    plt.show()