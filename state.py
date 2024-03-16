import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class State:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        self.isEnd = False
        self.boardHash = None
        # makes board and sets player as red
        self.playerSymbol = 1 # red
        self.board = np.array([
            [0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0],
            [0, -1, 0, -1, 0, -1, 0, -1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0]])
        # negatives: black
        # positives: red

    # get unique hash of current board state
    def getHash(self):
        #self.boardHash = str(self.board.flatten())
        self.boardHash = self.board.flatten()
        return self.boardHash
    
    def winner(self):
        if not any(value > 0 for row in self.board for value in row):
            # no reds on board
            self.isEnd = True
            return -1 # black wins
        if not any(value < 0 for row in self.board for value in row):
            # no blacks on board
            self.isEnd = True
            return 1 # red wins
        
        if len(self.availableMoves()) == 0:
            if self.playerSymbol == 1:
                # red has no legal moves left
                self.isEnd = True
                return -1
            else:
                # black has no legal moves left
                self.isEnd = True
                return 1 
                
    # need this for getLegalMoves
    def is_valid_move(self, start_row, start_col, end_row, end_col):
        if any(coord < 0 or coord > 7 for coord in [start_row, start_col, end_row, end_col]):
            # invalid index
            return False
        if np.sign(self.board[start_row][start_col]) != self.playerSymbol: 
            # if your player does not have piece there
            return False
        if self.board[end_row][end_col] != 0: 
            # end spot is not empty
            return False
        if self.board[start_row][start_col] == -1 and end_row <= start_row:
            # black cannot move up
            return False
        if self.board[start_row][start_col] == 1 and end_row >= start_row:
            # red cannot move down
            return False
        if abs(end_row - start_row) != abs(end_col - start_col):
            # rows and cols moved should be the same
            return False
        if abs(end_row - start_row) > 2:
            # cannot move more than two rows up or down
            return False
        if abs(end_row - start_row) == 2:
            # check if its a valid jump
            # no double jumps (for now) :-)
            mid_row = (start_row + end_row) // 2
            mid_col = (start_col + end_col) // 2
            if np.sign(self.board[mid_row][mid_col]) != (-1 if self.playerSymbol == 1 else 1):
                return False
        return True

    def getLegalMoves(self, symbol):
        legal_moves = []
        for row in range(8):
            for col in range(8):
                if np.sign(self.board[row][col]) == symbol: 
                    for i in (-2, -1, 1, 2):
                        for j in range(-1, 2, 2):
                            end_row = row + i
                            end_col = col + i*j
                            if self.is_valid_move(row, col, end_row, end_col):
                                legal_moves.append((row, col, end_row, end_col))
                            
        return legal_moves

    def availableMoves(self):
        # Return available positions for current player
        # split into 2 functions bc idk how to get for the right player
        # combine this and getLegalMoves into one func if possible idk
        moves = self.getLegalMoves(self.playerSymbol)
        return moves
    
    def availableJumps(self, row, col):
        # available jumps for current player
        # purpose: double/triple jumps
        jumps = [(row, col, row, col)]
        for i in (-2, 2):
            for j in range(-1, 2, 2):
                end_row = row + i
                end_col = col + i*j
                if self.is_valid_move(row, col, end_row, end_col):
                    jumps.append((row, col, end_row, end_col))
        return jumps

    def updateState(self, move):
        # Update board state based on the action taken by the player
        start_row = move[0]
        start_col = move[1]
        end_row = move[2]
        end_col = move[3]
        # update new location, accounting for if it turns into a king
        if self.playerSymbol == 1 and end_row == 0:
            # turns into king
            self.board[end_row][end_col] = 2 #*self.playerSymbol
        elif self.playerSymbol == -1 and end_row == 7:
            # turns into king
            self.board[end_row][end_col] = -2 
        else:
            self.board[end_row][end_col] = self.board[start_row][start_col]
        if abs(end_row - start_row) == 2:
            # remove the piece that was jumped
            self.board[(start_row + end_row) // 2][(start_col + end_col) // 2] = 0
            # no double jumps (for now) <3
        self.board[start_row][start_col] = 0
        # switch player
        self.playerSymbol = -1 if self.playerSymbol == 1 else 1

    def getNewBoard(self, move):
        newboard = self.board.copy()
        start_row = move[0]
        start_col = move[1]
        end_row = move[2]
        end_col = move[3]
        # update new location, accounting for if it turns into a king
        if self.playerSymbol == 1 and end_row == 0:
            # turns into king
            newboard[end_row][end_col] = 2 
        elif self.playerSymbol == -1 and end_row == 7:
            # turns into king
            newboard[end_row][end_col] = -2 
        else:
            newboard[end_row][end_col] = newboard[start_row][start_col]
        if abs(end_row - start_row) == 2:
            # remove the piece that was jumped
            newboard[(start_row + end_row) // 2][(start_col + end_col) // 2] = 0
        newboard[start_row][start_col] = 0
        return newboard

    def getHashNew(self, move):
        return self.getNewBoard(move).flatten()
    
    # only when game ends
    def giveReward(self):
        # Assign rewards to players based on the game outcome
        result = self.winner()
        # backpropogate result
        if result == 1:             # red (1) wins
            self.p1.feedReward(1)
            self.p2.feedReward(0)
        else:                       # black (-1) wins
            self.p1.feedReward(0)
            self.p2.feedReward(1)
        # no option for a tie/stalemate in checkers

    def reset(self):
        # Reset the game state
        self.boardHash = None
        self.isEnd = False
        self.playerSymbol = 1 # red
        self.board = np.array([
            [0, -1, 0, -1, 0, -1, 0, -1],
            [-1, 0, -1, 0, -1, 0, -1, 0],
            [0, -1, 0, -1, 0, -1, 0, -1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0]])
        # negatives: black
        # positives: red
    
    def play(self, rounds=100):
        won = 0
        lost = 0
        winrates = []
        for i in tqdm(range(rounds)):
            #if i%1000 == 0:
            #if i%(rounds//50) == 0:
            #    print("Rounds {}".format(i))
            
            # tbh a lot of this logic is kind of confusing and i think this could be improved

            while not self.isEnd: #i.e. while the game has not ended
                # Player 1
                moves = self.availableMoves()
                p1_action = self.p1.chooseAction(moves, self.board, self.playerSymbol) 
                # take action and update board state
                self.updateState(p1_action)

                
                # TODO: add double and triple jumps from play_human
                

                board_hash = self.getHash()
                self.p1.addState(board_hash)
                # check board status if it is end

                # look at this closer later
                win = self.winner()
                if win is not None: #i.e. there is a winner
                    # self.showBoard()
                    won += 1
                    self.giveReward() # reward or penalize depending on outcome
                    self.p1.reset()
                    self.p2.reset()
                    self.reset()
                    break
                else:
                    # Player 2 takes their turn
                    moves = self.availableMoves()
                    p2_action = self.p2.chooseAction(moves, self.board, self.playerSymbol) 
                    # take action and update board state
                    self.updateState(p2_action)

                    # TODO: add double and triple jumps from play_human

                    board_hash = self.getHash()
                    self.p2.addState(board_hash)

                    win = self.winner()
                    if win is not None:
                        #self.showBoard()
                        lost += 1
                        self.giveReward()
                        self.p1.reset()
                        self.p2.reset()
                        self.reset()
                        break
            winrate = int((won)/(won+lost)*100)
            winrates.append(winrate)
        indices = list(range(len(winrates)))
        plt.plot(indices, winrates, marker='o', linestyle='-')

        plt.title('Rates of win')
        plt.xlabel('generations')
        plt.ylabel('wins [%]')
        
    def checkInside(self, x, y, ar):
        for tab in ar:
            if tab[0] == x and tab[1] == y:
                return True
        return False

    def minmax(self, player, RL = False):
        Leafs = []
        FinalMoves = []
        b1 = self.board.copy()
        s1 = self.score1
        s2 = self.score2
        moves1 = self.getLegalMoves(player)
        for m1 in moves1:
            self.board = b1
            self.score1 = s1
            self.score2 = s2
            self.updateState_1(m1)
            b2 = self.board.copy()
            s21 = self.score1
            s22 = self.score2
            moves2 = self.getLegalMoves(-player)
            if len(moves2) == 0:
                #Leafs.append((m1, 100, self.evaluate(player))) 
                Leafs.append((m1, self.evaluate(player))) 
            for m2 in moves2:
                self.board = b2
                self.score1 = s21
                self.score2 = s22
                self.updateState_1(m2)
                #Leafs.append((m1, self.getScore(), self.evaluate(player)))
                Leafs.append((m1, self.evaluate(player)))
        self.board = b1
        self.score1 = s1
        self.score2 = s2
        '''
        max = -999
        for leaf in Leafs:
            # leaf[1] is the score
            if leaf[1] > max:
                max = leaf[1]
                FinalMoves.clear()
                # leaf[0][2] is end row, leaf[0][3] is end col
                if not self.checkInside(leaf[0][2], leaf[0][3], FinalMoves) : 
                    FinalMoves.append(leaf[0])
            elif leaf[1] == max:
                if not self.checkInside(leaf[0][2], leaf[0][3], FinalMoves) : 
                    FinalMoves.append(leaf[0])'''
                    
        if RL:
            return Leafs
        return FinalMoves


    def play_human(self):
        # Play against a human
        while not self.isEnd:
            # Player 1
            moves = self.availableMoves()
            p1_action = self.p1.chooseAction(moves, self.board, self.playerSymbol)
            print("{} takes action {}".format(self.p1.name, p1_action))
            # take action and update board state
            self.updateState(p1_action)

            # Double Jump
            # check if the action was a jump 
            if abs(p1_action[0] - p1_action[2]) == 2:
                # if it was a jump, then get all available addl jumps at this new board state
                jumps = self.availableJumps(p1_action[0], p1_action[1])
                if len(jumps) > 1:
                    # choose an action out of the jumps
                    p1_action = self.p1.chooseAction(jumps, self.board, self.playerSymbol)
                    print("{} takes action {}".format(self.p1.name, p1_action))
                    self.updateState(p1_action)

                    # Triple Jump
                    # check if the action was a jump 
                    if abs(p1_action[0] - p1_action[2]) == 2:
                        # if it was a jump, then get all available addl jumps at this new board state
                        jumps = self.availableJumps(p1_action[0], p1_action[1])
                        if len(jumps) > 1:
                            # choose an action out of the jumps
                            p1_action = self.p1.chooseAction(jumps, self.board, self.playerSymbol)
                            print("{} takes action {}".format(self.p1.name, p1_action))
                            self.updateState(p1_action)


            self.showBoard()
            # check board status if it is end
            win = self.winner()
            if win is not None:
                print(self.p1.name, "wins!")
                self.reset()
                break
            else:
                # player 2
                moves = self.availableMoves()
                p2_action = self.p2.chooseAction(moves)
                print("{} takes action {}".format(self.p2.name, p2_action))
                self.updateState(p2_action)
                self.showBoard()

                # Double Jump
                # check if the action was a jump 
                if abs(p2_action[0] - p2_action[2]) == 2:
                    # if it was a jump, then get all available addl jumps at this new board state
                    jumps = self.availableJumps(p2_action[0], p2_action[1])
                    if len(jumps) > 1:
                        # choose an action out of the jumps
                        p2_action = self.p2.chooseAction(jumps)
                        print("{} takes action {}".format(self.p2.name, p2_action))
                        self.updateState(p2_action)
                        self.showBoard()

                        # Triple Jump
                        # check if the action was a jump 
                        if abs(p2_action[0] - p2_action[2]) == 2:
                            # if it was a jump, then get all available addl jumps at this new board state
                            jumps = self.availableJumps(p2_action[0], p2_action[1])
                            if len(jumps) > 1:
                                # choose an action out of the jumps
                                p2_action = self.p2.chooseAction(jumps)
                                print("{} takes action {}".format(self.p2.name, p2_action))
                                self.updateState(p2_action)
                                self.showBoard()

                win = self.winner()
                if win is not None:
                    print(self.p2.name, "wins!")
                    self.reset()
                    break
            
    
    def showBoard(self, board):
        # Display the current state of the board
        mapping = {1: 'r', 2: 'R', 0: ' ', -2: 'B', -1: 'b'}
        mapped_board = np.vectorize(lambda value: mapping.get(value, str(value)))(board)
        display_board = '   0  1  2  3  4  5  6  7\n'
        for i, row in enumerate(mapped_board):
            if i == len(mapped_board) - 1:  # Check if it's the last row
                display_board += f"{i} [{', '.join(row)}]"
            else:
                display_board += f"{i} [{', '.join(row)}]\n"
        print(display_board)
