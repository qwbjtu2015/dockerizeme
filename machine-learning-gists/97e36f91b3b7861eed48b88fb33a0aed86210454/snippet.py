import numpy as np
import random
import copy

'''
--------------
Specifications
--------------
-Task (T): Playing  Tic-Tac-Toe
-Perfromance(P): % of games won in Tournament
-Experience: Data from games played against self (Indirect feedback)
-Learning: It boils down to optimizing the function (Target function) that finds best Move 
	for a board State from a set of legal Moves.

---------------------------------
Design choices for Implementation
---------------------------------
-In the case of this game we learn an approximation of the ideal target function described above.
-Approximate Target Function (V): Maps board -> R
-Approximate Target Function Representation: w0+w1(x1)+w2(x2)+..... ws (weights) & xs (board features)		

'''

class ExperimentGenerator():
	''' Experiment Generator generates New problems. In this case it just returns the same initial game boards '''
	
	def __init__(self):
		
		self.initBoardState = [[' ',' ',' '],[' ',' ',' '],[' ',' ',' ']]

	def generateNewProblem(self):
		''' Returns initial Board State '''
		return(self.initBoardState)

	
class Player:
	''' Provides all the methods that a Player must execute. '''

	def __init__(self,playerSymbol,playerTargetFunctionWeightVector):
		
		self.playerSymbol = playerSymbol
		self.playerTargetFunctionWeightVector = playerTargetFunctionWeightVector

	def isGameOver(self,board,playerSymbol):
		''' Returns True if game is over else returns false '''

		flag = False
		# Game already over
		if(board == -1):
			flag = True		
		# Game won by either player
		elif((board[0][0] == board[0][1] == board[0][2] == playerSymbol)  or 
			(board[1][0] == board[1][1] == board[1][2] == playerSymbol) or
			(board[2][0] == board[2][1] == board[2][2] == playerSymbol) or
			(board[0][0] == board[1][0] == board[2][0] == playerSymbol) or
			(board[0][1] == board[1][1] == board[2][1] == playerSymbol) or
			(board[0][2] == board[1][2] == board[2][2] == playerSymbol) or
			(board[0][0] == board[1][1] == board[2][2] == playerSymbol) or
			(board[0][2] == board[1][1] == board[2][0] == playerSymbol) ): 
				flag = True
		# Board Full		
		elif(' ' not in np.array(board).flatten()):
			flag = True
		return(flag)	

	def lookForLegalMoves(self,boardState,playerSymbol):
		''' Returns a list of legal moves for a given board state.'''

		legalMoves = []
		for i in range(len(boardState[0])):
			for j in range(len(boardState[0])):
				if(boardState[i][j] == ' '):
					tempBoard = copy.deepcopy(boardState)
					tempBoard[i][j]=playerSymbol
					legalMoves.append(tempBoard)					
		return(legalMoves)	

	def extractFeatures(self,board,playerSymbol1,playerSymbol2):
		''' Returns and extracted feature Vector for a given board state '''

		w1,w2,w3,w4,w5,w6 = 0,0,0,0,0,0
		for i in range(3):
			# 1 Player-1's Symbol in a row in an open row
			if (((board[i][0] == playerSymbol1) and ( board[i][1] == board[i][2] == ' ')) or
				((board[i][2] == playerSymbol1) and ( board[i][0] == board[i][1] == ' '))):
				w1 = w1 + 1
			# 1 Player-1's Symbol in a row in an open col
			if (((board[0][i] == playerSymbol1) and ( board[1][i] == board[2][i] == ' ')) or
				((board[2][i] == playerSymbol1) and ( board[0][i] == board[1][i] == ' '))):
				w1 = w1 + 1		
			# 1 Player-2's Symbol in a row in an open row
			if (((board[i][0] == playerSymbol2) and ( board[i][1] == board[i][2] == ' ')) or
				((board[i][2] == playerSymbol2) and ( board[i][0] == board[i][1] == ' '))):
				w2 = w2 + 1
			# 1 Player-2's Symbol in a row in an open col
			if (((board[0][i] == playerSymbol2) and ( board[1][i] == board[2][i] == ' ')) or
				((board[2][i] == playerSymbol2) and ( board[0][i] == board[1][i] == ' '))):
				w2 = w2 + 1	
			# 2 Player-1's Symbols in a row
			if (((board[i][0] == board[i][1] == playerSymbol1) and ((board[i][2]) == ' ')) or
				((board[i][1] == board[i][2] == playerSymbol1) and ((board[i][0]) == ' '))):
				w3 = w3 + 1	
			# 2 Player-2's Symbols in a row
			if (((board[i][0] == board[i][1] == playerSymbol2) and ((board[i][2]) == ' ')) or
				((board[i][1] == board[i][2] == playerSymbol2) and ((board[i][0]) == ' '))):
				w4 = w4 + 1	
			# 3 Player-1's Symbols in a row with an open box
			if ((board[i][0] == board[i][1] == board[i][2] == playerSymbol1)):
				w5 = w5 + 1	
			# 3 Player-2's Symbols in a row with an open box
			if (board[i][0] == board[i][1] == board[i][2] == playerSymbol2 ):
				w6 = w6 + 1
		# Added 1 for bias		
		feature_vector = [1,w1,w2,w3,w4,w5,w6]		
		return(feature_vector)
		
	def boardPrint(self,board):
		''' Displays board in proper format'''
		
		print('\n')
		print(board[0][0] + '|' + board[0][1] + '|' + board[0][2])
		print("-----")
		print(board[1][0] + '|' + board[1][1] + '|' + board[1][2])
		print("-----")
		print(board[2][0] + '|' + board[2][1] + '|' + board[2][2])
		print('\n')	

	def calculateNonFinalBoardScore(self,weight_vector,feature_vector):
		''' Returns score/Value of a given non final board state '''

		weight_vector = np.array(weight_vector).reshape((len(weight_vector),1))
		feature_vector = np.array(feature_vector).reshape((len(feature_vector),1))
		boardScore = np.dot(weight_vector.T,feature_vector)
		return(boardScore[0][0])
	
	def chooseMove(self,board,playerSymbol1,playerSymbol2):
		''' Returns the best move from a set of legal moves for a given board state.'''

		legalMoves = self.lookForLegalMoves(board,playerSymbol1)
		legalMoveScores = [self.calculateNonFinalBoardScore(self.playerTargetFunctionWeightVector,
			self.extractFeatures(i,playerSymbol1,playerSymbol2)) for i in legalMoves]
		newBoard = legalMoves[np.argmax(legalMoveScores)]
		return(newBoard)		
					
	def chooseRandomMove(self,board,playerSymbol):
		''' Returns a random move from a set of legal moves for a given board state '''

		legalMoves = self.lookForLegalMoves(board,playerSymbol)
		newBoard = random.choice(legalMoves)
		return(newBoard)			


class PerformanceSystem:
	''' Performance System takes the initial Game board & returns Solution trace/Game History of the Game Play '''
	
	def __init__(self,initialBoard,playersTargetFunctionWeightVectors,playerSymbols):

		self.board = initialBoard
		self.playersTargetFunctionWeightVectors = playersTargetFunctionWeightVectors
		self.playerSymbols = playerSymbols
	
	def isGameOver(self,board,playerSymbol):
		''' Returns True if game is over else returns false '''

		flag = False
		# Game already over
		if(board == -1):
			flag = True
		# Game won by either player
		elif((board[0][0] == board[0][1] == board[0][2] == playerSymbol)  or 
			(board[1][0] == board[1][1] == board[1][2] == playerSymbol) or
			(board[2][0] == board[2][1] == board[2][2] == playerSymbol) or
			(board[0][0] == board[1][0] == board[2][0] == playerSymbol) or
			(board[0][1] == board[1][1] == board[2][1] == playerSymbol) or
			(board[0][2] == board[1][2] == board[2][2] == playerSymbol) or
			(board[0][0] == board[1][1] == board[2][2] == playerSymbol) or
			(board[0][2] == board[1][1] == board[2][0] == playerSymbol) ): 
				flag = True
		# Board full		
		elif(' ' not in np.array(board).flatten()):
			flag = True
		return(flag)
		
	def generateGameHistory(self):
		''' Returns Solution trace generated from pitting 2 players(agents) against each '''
		
		gameHistory = []
		gameStatusFlag = True
		player1 = Player(self.playerSymbols[0],self.playersTargetFunctionWeightVectors[0])
		player2 = Player(self.playerSymbols[1],self.playersTargetFunctionWeightVectors[1])
		tempBoard = copy.deepcopy(self.board)
		while(gameStatusFlag):			
			#tempBoard = player1.chooseRandomMove(tempBoard,player1.playerSymbol)
			tempBoard = player1.chooseMove(tempBoard,player1.playerSymbol,player2.playerSymbol)
			gameHistory.append(tempBoard)
			gameStatusFlag = not self.isGameOver(tempBoard,player1.playerSymbol)
			if(gameStatusFlag == False):
				break
			tempBoard = player2.chooseRandomMove(tempBoard,player2.playerSymbol)
			#tempBoard = player2.chooseMove(tempBoard,player2.playerSymbol,player1.playerSymbol)
			gameHistory.append(tempBoard)
			gameStatusFlag =  not self.isGameOver(tempBoard,player2.playerSymbol)					
		return(gameHistory)	


class Critic:
	''' Critic takes the Game History & generates training examples to be used by Generalizer.'''

	def __init__(self,gameHistory):
		self.gameHistory = gameHistory

	def extractFeatures(self,board,playerSymbol1,playerSymbol2):
		''' Returns and extracted feature Vector for a given board state '''

		w1,w2,w3,w4,w5,w6 = 0,0,0,0,0,0
		for i in range(3):
			# 1 Player-1's Symbol in a row in an open row
			if (((board[i][0] == playerSymbol1) and ( board[i][1] == board[i][2] == ' ')) or
				((board[i][2] == playerSymbol1) and ( board[i][0] == board[i][1] == ' '))):
				w1 = w1 + 1
			# 1 Player-1's Symbol in a row in an open col
			if (((board[0][i] == playerSymbol1) and ( board[1][i] == board[2][i] == ' ')) or
				((board[2][i] == playerSymbol1) and ( board[0][i] == board[1][i] == ' '))):
				w1 = w1 + 1		
			# 1 Player-2's Symbol in a row in an open row
			if (((board[i][0] == playerSymbol2) and ( board[i][1] == board[i][2] == ' ')) or
				((board[i][2] == playerSymbol2) and ( board[i][0] == board[i][1] == ' '))):
				w2 = w2 + 1	
			# 1 Player-2's Symbol in a row in an open col
			if (((board[0][i] == playerSymbol2) and ( board[1][i] == board[2][i] == ' ')) or
				((board[2][i] == playerSymbol2) and ( board[0][i] == board[1][i] == ' '))):
				w2 = w2 + 1			
			# 2 Player-1's Symbols in a row
			if (((board[i][0] == board[i][1] == playerSymbol1) and ((board[i][2]) == ' ')) or
				((board[i][1] == board[i][2] == playerSymbol1) and ((board[i][0]) == ' '))):
				w3 = w3 + 1
			# 2 Player-2's Symbols in a row
			if (((board[i][0] == board[i][1] == playerSymbol2) and ((board[i][2]) == ' ')) or
				((board[i][1] == board[i][2] == playerSymbol2) and ((board[i][0]) == ' '))):
				w4 = w4 + 1	
			# 3 Player-1's Symbols in a row with an open box
			if ((board[i][0] == board[i][1] == board[i][2] == playerSymbol1)):
				w5 = w5 + 1	
			# 3 Player-2's Symbols in a row with an open box
			if (board[i][0] == board[i][1] == board[i][2] == playerSymbol2 ):
				w6 = w6 + 1
		# Added 1 for bias		
		feature_vector = [1,w1,w2,w3,w4,w5,w6]		
		return(feature_vector)		
		


	def calculateNonFinalBoardScore(self,weight_vector,feature_vector):
		''' Returns score/Value of a given non final board state '''

		weight_vector = np.array(weight_vector).reshape((len(weight_vector),1))
		feature_vector = np.array(feature_vector).reshape((len(feature_vector),1))
		boardScore = np.dot(weight_vector.T,feature_vector)
		return(boardScore[0][0])

	def calculateFinalBoardScore(self,board,playerSymbol1,playerSymbol2):
		''' Returns score/Value of a given final board state '''

		# If game ends in a draw
		score = 0
		# If player-1 (i.e self) wins
		if((board[0][0] == board[0][1] == board[0][2] == playerSymbol1)  or 
			(board[1][0] == board[1][1] == board[1][2] == playerSymbol1) or
			(board[2][0] == board[2][1] == board[2][2] == playerSymbol1) or
			(board[0][0] == board[1][0] == board[2][0] == playerSymbol1) or
			(board[0][1] == board[1][1] == board[2][1] == playerSymbol1) or
			(board[0][2] == board[1][2] == board[2][2] == playerSymbol1) or
			(board[0][0] == board[1][1] == board[2][2] == playerSymbol1) or
			(board[0][2] == board[1][1] == board[2][0] == playerSymbol1) ):
			score = 100
		# If player-2 (i.e opponent) wins	
		elif((board[0][0] == board[0][1] == board[0][2] == playerSymbol2)  or 
			(board[1][0] == board[1][1] == board[1][2] == playerSymbol2) or
			(board[2][0] == board[2][1] == board[2][2] == playerSymbol2) or
			(board[0][0] == board[1][0] == board[2][0] == playerSymbol2) or
			(board[0][1] == board[1][1] == board[2][1] == playerSymbol2) or
			(board[0][2] == board[1][2] == board[2][2] == playerSymbol2) or
			(board[0][0] == board[1][1] == board[2][2] == playerSymbol2) or
			(board[0][2] == board[1][1] == board[2][0] == playerSymbol2) ):
			score = -100
		return(score)		
			

	def generateTrainingSamples(self,weight_vector,playerSymbol1,playerSymbol2):
		''' Returns training examples i.e a list of list of feature vectors & assosiated scores '''
		
		trainingExamples=[]
		for i in range(len(self.gameHistory)-1):
			feature_vector = self.extractFeatures(self.gameHistory[i+1],playerSymbol1,playerSymbol2)
			trainingExamples.append([feature_vector,self.calculateNonFinalBoardScore(weight_vector,feature_vector)])
		trainingExamples.append([self.extractFeatures(self.gameHistory[-1],playerSymbol1,playerSymbol2),
			self.calculateFinalBoardScore(self.gameHistory[-1],playerSymbol1,playerSymbol2)])
		return(trainingExamples)

	def arrayPrint(self,board):
		''' Displays a given 2D array in proper format '''

		print('\n')
		print(board[0][0] + '|' + board[0][1] + '|' + board[0][2])
		print("-----")
		print(board[1][0] + '|' + board[1][1] + '|' + board[1][2])
		print("-----")
		print(board[2][0] + '|' + board[2][1] + '|' + board[2][2])
		print('\n')

	def boardDisplay(self,playerSymbol1,playerSymbol2,gameStatusCount):
		''' Displayes board & returns a list containg Win/Loss/Draw counts '''

		for board in self.gameHistory:
			self.arrayPrint(board)		
		finalScore = self.calculateFinalBoardScore(self.gameHistory[-1],playerSymbol1,playerSymbol2)
		if(finalScore == 100):
			print(playerSymbol1 + " wins")
			gameStatusCount[0] = gameStatusCount[0] + 1
		elif(finalScore == -100):
			print(playerSymbol2 + " wins")
			gameStatusCount[1] = gameStatusCount[1] + 1
		else:
			print("Draw")
			gameStatusCount[2] = gameStatusCount[2] + 1	
		return(gameStatusCount)		 	
		

class Generalizer:
	''' It takes Training examples from Critic & suggests/improves the Hypothesis function (Approximate Target Function) '''
	
	def __init__(self,trainingExamples):
		self.trainingExamples = trainingExamples

	def calculateNonFinalBoardScore(self,weight_vector,feature_vector):
		''' Returns score/Value of a given non final board state '''

		weight_vector = np.array(weight_vector).reshape((len(weight_vector),1))
		feature_vector = np.array(feature_vector).reshape((len(feature_vector),1))
		boardScore = np.dot(weight_vector.T,feature_vector)
		return(boardScore[0][0])	

	def lmsWeightUpdate(self,weight_vector,alpha=0.4):
		''' Returns new Weight vector updated y learning from Training examples via LMS (Least Mean Squares) training rule.'''

		for trainingExample in self.trainingExamples:
			vTrainBoardState = trainingExample[1]
			vHatBoardState = self.calculateNonFinalBoardScore(weight_vector,trainingExample[0])
			weight_vector = weight_vector + (alpha * (vTrainBoardState - vHatBoardState) * np.array(trainingExample[0]))
		return (weight_vector)	
		

		
def train(numTrainingSamples = 10):
	''' Executions of Training & Testing phases'''

	# Training phase (Indirect Feedback via Computer v/s Computer)
	trainingGameCount = 0
	playerSymbols = ('X','O')
	playersTargetFunctionWeightVectors = [np.array([.5,.5,.5,.5,.5,.5,.5]),np.array([.5,.5,.5,.5,.5,.5,.5])]
	gameStatusCount = [0,0,0]
	#initWeight = playersTargetFunctionWeightVectors[0]


	while (trainingGameCount < numTrainingSamples):
		
		# Experiment Generator
		experimentGenerator = ExperimentGenerator()
		initialBoardState = experimentGenerator.generateNewProblem()
		
		# Performance System
		performanceSystem = PerformanceSystem(initialBoardState,playersTargetFunctionWeightVectors,playerSymbols)
		gameHistory = performanceSystem.generateGameHistory()
				
		# Critic
		critic = Critic(gameHistory)
		trainingExamplesPlayer1 = critic.generateTrainingSamples(playersTargetFunctionWeightVectors[0],
			playerSymbols[0],playerSymbols[1])
		trainingExamplesPlayer2 = critic.generateTrainingSamples(playersTargetFunctionWeightVectors[1],
			playerSymbols[1],playerSymbols[0])
		# Display board states
		gameStatusCount = critic.boardDisplay(playerSymbols[0],playerSymbols[1],gameStatusCount)
		
		# Generalizer
		generalizer = Generalizer(trainingExamplesPlayer1)
		playersTargetFunctionWeightVectors = [generalizer.lmsWeightUpdate(playersTargetFunctionWeightVectors[0]),
			generalizer.lmsWeightUpdate(playersTargetFunctionWeightVectors[1])]	 

		trainingGameCount = trainingGameCount+1

	# Epoch wise Game status	
	print("\nTraining Results: (" + "Player-1 Wins = " + str(gameStatusCount[0]) +
			", Player-2 Wins = " + str(gameStatusCount[1]) + ", Game Draws = " + str(gameStatusCount[2]) +
			")\n")	

	# Weight Learnt from previous games
	learntWeight =  list(np.mean(np.array([playersTargetFunctionWeightVectors[0],
		playersTargetFunctionWeightVectors[1]]),axis = 0))
	print("Final Learnt Weight Vector: \n"+ str(learntWeight))
		
	# Computer vs Human Games
	print("\nDo you want to play(y/n) v/s Computer AI")
	ans = input() 
	while(ans == "y"):

		experimentGenerator = ExperimentGenerator()
		boardState = experimentGenerator.generateNewProblem()
		gameStatusFlag = True
		computer = Player('X',learntWeight)
		gameHistory = []

		print('\nBegin Computer(X) v/s Human(O) Tic-Tac-Toe\n')
		while(gameStatusFlag):
			
			boardState = computer.chooseMove(boardState,computer.playerSymbol,'H')
			print('Computers\'s Turn:\n')
			computer.boardPrint(boardState)
			gameHistory.append(boardState)
			gameStatusFlag = not computer.isGameOver(boardState,computer.playerSymbol)
			if(gameStatusFlag == False):
				break

			print('Human\'s Turn:\n')	
			print('Enter X-coordinate(0-2):')
			x = int(input())
			print('Enter Y-coordinate(0-2):')
			y = int(input())
		
			boardState[x][y] = 'O'
			computer.boardPrint(boardState)
			gameHistory.append(boardState)
			gameStatusFlag =  not computer.isGameOver(boardState,'H')

		print("Do you want to continue playing(y/n).")
		ans = input()
		if(ans != 'y'):
			break 		

# Actual Execution of Tic-Tac-Toe Game 
train(10000)
