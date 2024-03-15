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

'''def concatenate(array1, array2):
    return tf.concat([array1, array2], axis=0)'''

def GetModel(game, rounds=100):
    won = 0
    lost = 0
    draw = 0
    winrates = []

    LEARNING_RATE = 0.5
    DISCOUNT_FACTOR = 0.95
    EXPLORATION = 0.95

    data = []
    labels = np.zeros(1)

    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=4)) 
    model.add(Dense(16, activation='relu',  kernel_regularizer=regularizers.l2(0.1)))

    model.add(Dense(1, activation='relu',  kernel_regularizer=regularizers.l2(0.1)))
    model.compile(optimizer='nadam', loss='binary_crossentropy', metrics=["acc"])

    for i in tqdm(range(rounds)):
        print("won", won, ", lost", lost, ", draw", draw)
        data = []
        isDraw = False
        for g in range(10):

            temp_data = []
            player = 1
            count = 0
            while True:
                
                count += 1
                #print("count is", count)
                end2 = 0
                if count > 100:
                    draw += 1
                    isDraw = True
                    break
                else:
                    if (player == 1):
                        #print("player 1's turn")
                        game.playerSymbol = 1
                        moves = game.availableMoves() #p1
                        #print("available moves are", moves)
                        Leaf = tf.zeros((len(moves), 4)) 
                        for l in range(len(moves)):
                            tensor = moves[l] # the lth available move
                            Leaf = tf.tensor_scatter_nd_update(Leaf, [[l]], [tensor])
                        scores = model.predict_on_batch(Leaf)
                        #print("scores is", scores)
                        if (len(moves) == 0): 
                            end2 = -1 # win goes to the other player
                            continue
                        i = np.argmax(scores)
                        game.updateState(moves[i]) #switches to p2
                        tab = moves[i]
                        temp_data.append(tab) 
                    elif (player == -1):
                        #print("player 2's turn")
                        game.playerSymbol = -1
                        moves = game.availableMoves()
                        #print("available moves are", moves)
                        Leaf = tf.zeros((len(moves), 4)) 
                        for l in range(len(moves)):
                            tensor = moves[l] # the lth available move
                            Leaf = tf.tensor_scatter_nd_update(Leaf, [[l]], [tensor])
                        scores = model.predict_on_batch(Leaf)
                        #print("scores is", scores)
                        if (len(moves) == 0): 
                            end2 = 1 
                            continue
                        i = np.argmax(scores)
                        game.updateState(moves[i]) #switches to p2
                        #tab = moves[i]
                        #temp_data.append(tab) 
                #print("temp_data is", temp_data)

                end = game.winner()
                #print("end is", end)

                if (end == 1 or end2 == 1): # if player 1 wins
                    #print("player 1 won")
                    won += 1
                    reward = 10
                    temp_tensor = tf.constant(temp_data[1:])
                    #print("temp_tensor is", temp_tensor)
                    old_prediction = model.predict_on_batch(temp_tensor)
                    #print("old_prediction is", old_prediction)
                    #print("old_prediction shape is", old_prediction.shape)
                    optimal_future_value = np.ones(old_prediction.shape)
                    #print("optimal_future_value is", optimal_future_value)
                    #print("optimal_future_value shape is", optimal_future_value.shape)
                    temp_labels = old_prediction + LEARNING_RATE * (reward + DISCOUNT_FACTOR * optimal_future_value - old_prediction )
                    #print("temp_labels is", temp_labels)
                    #print("temp_labels shape is", temp_labels.shape)
                    data = concatenate(data, temp_data[1:])
                    #print("data is", data)
                    #print("data shape is", len(data))
                    labels = np.vstack((labels, temp_labels))
                    #print("labels is", labels)
                    #print("labels shape is", labels.shape)
                    game.reset()
                    break
                elif (end == -1 or end2 == -1):
                    lost += 1
                    reward = -10
                    temp_tensor = tf.constant(temp_data[1:])
                    #print("temp_tensor is", temp_tensor)
                    old_prediction = model.predict_on_batch(temp_tensor)
                    #print("old_prediction is", old_prediction)
                    #print("old_prediction shape is", old_prediction.shape)
                    optimal_future_value = np.ones(old_prediction.shape)
                    #print("optimal_future_value is", optimal_future_value)
                    #print("optimal_future_value shape is", optimal_future_value.shape)
                    temp_labels = old_prediction + LEARNING_RATE * (reward + DISCOUNT_FACTOR * optimal_future_value - old_prediction )
                    #print("temp_labels is", temp_labels)
                    #print("temp_labels shape is", temp_labels.shape)
                    data = concatenate(data, temp_data[1:])
                    #print("data is", data)
                    #print("data shape is", len(data))
                    labels = np.vstack((labels, temp_labels))
                    #print("labels is", labels)
                    #print("labels shape is", labels.shape)
                    game.reset()
                    break

                
                player = -player
        data = tf.constant(data)
        if (not isDraw): model.fit(data[1:], labels[2:], epochs=16, batch_size=256)
        labels = np.zeros(1)

        winrate = int((won)/(won+draw+lost)*100)
        winrates.append(winrate)
        model.save("model.keras")
    indices = list(range(len(winrates)))
    plt.plot(indices, winrates, marker='o', linestyle='-')
    plt.title('Rates of win')
    plt.xlabel('generations')
    plt.ylabel('wins [%]')
    plt.show()