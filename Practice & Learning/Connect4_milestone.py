# Class: CS5 Gold
# File: hw11pr2.py
# Name: Michael Hess
# Description: Board AI
import random
import copy
def inarow_Neast(ch, r_start, c_start, A, N):
	"""Starting from (row, col) of (r_start, c_start)
	   within the 2d list-of-lists A (array),
	   returns True if there are N ch's in a row
	   heading east and returns False otherwise.
	"""
	H = len(A)
	W = len(A[0])
	if r_start < 0 or r_start > H - 1:
		return False  # out of bounds row
	if c_start < 0 or c_start + (N - 1) > W - 1:
		return False  # o.o.b. col
	# loop over each location _offset_ i
	for i in range(N):
		if A[r_start][c_start + i] != ch:  # a mismatch!
			return False
	return True  # all offsets succeeded, so we return True

def inarow_Nsouth(ch, r_start, c_start, A, N):
	"""Starting from (row, col) of (r_start, c_start)
	   within the 2d list-of-lists A (array),
	   returns True if there are N ch's in a row
	   heading south and returns False otherwise.
	"""
	H = len(A)
	W = len(A[0])
	if r_start < 0 or r_start + (N - 1) > H - 1:
		return False  # out of bounds row
	if c_start < 0 or c_start > W - 1:
		return False  # o.o.b. col
	# loop over each location _offset_ i
	for i in range(N):
		if A[r_start + i][c_start] != ch:  # a mismatch!
			return False
	return True  # all offsets succeeded, so we return True

def inarow_Nnortheast(ch, r_start, c_start, A, N):
	"""Starting from (row, col) of (r_start, c_start)
	   within the 2d list-of-lists A (array),
	   returns True if there are N ch's in a row
	   heading northeast and returns False otherwise.
	"""
	H = len(A)
	W = len(A[0])
	if r_start - (N - 1) < 0 or r_start > H - 1:
		return False  # out of bounds row
	if c_start < 0 or c_start + (N - 1) > W - 1:
		return False  # o.o.b. col
	# loop over each location _offset_ i
	for i in range(N):
		if A[r_start - i][c_start + i] != ch:  # a mismatch!
			return False
	return True  # all offsets succeeded, so we return True

def inarow_Nsoutheast(ch, r_start, c_start, A, N):
	"""Starting from (row, col) of (r_start, c_start)
	   within the 2d list-of-lists A (array),
	   returns True if there are N ch's in a row
	   heading southeast and returns False otherwise.
	"""
	H = len(A)
	W = len(A[0])
	if r_start < 0 or r_start + (N - 1) > H - 1:
		return False  # out of bounds row
	if c_start < 0 or c_start + (N - 1) > W - 1:
		return False  # o.o.b. col
	# loop over each location _offset_ i
	for i in range(N):
		if A[r_start + i][c_start + i] != ch:  # a mismatch!
			return False
	return True  # all offsets succeeded, so we return Truew


class Board:
	"""A data type representing a Connect-4 board
	   with an arbitrary number of rows and columns.
	"""

	def __init__(self, width, height):
		"""Construct objects of type Board, with the given width and height."""
		self.width = width
		self.height = height
		self.data = [[' '] * width for row in range(height)]

	# We do not need to return anything from a constructor!

	def __repr__(self):
		"""This method returns a string representation
		   for an object of type Board.
		"""
		s = ''  # the string to return
		for row in range(0, self.height):
			s += '|'
			for col in range(0, self.width):
				s += self.data[row][col] + '|'
			s += '\n'

		s += (2 * self.width + 1) * '-'  # bottom of the board

		# and the numbers underneath here

		return s  # the board is complete, return it

	def addMove(self, col, ox):
		"""
		:param col: the location to add the new move
		:param ox: X or O
		:return: no return, but adds the o/x to that column on the board from above
		"""
		x = 0
		if self.allowsMove(col):
			for row in range(self.height, 0, -1):
				if self.data[row - 1][col] == ' ':
					self.data[row - 1][col] = ox
					return


	def clear(self):
		"""
		:return: no return, but changes all entries on Board to ' '
		"""
		for row in range(self.height):
			for col in range(self.width):
				self.data[row][col] = ' '

	def setBoard(self, moveString):
		"""Accepts a string of columns and places
		   alternating checkers in those columns,
		   starting with 'X'.

		   For example, call b.setBoard('012345')
		   to see 'X's and 'O's alternate on the
		   bottom row, or b.setBoard('000000') to
		   see them alternate in the left column.

		   moveString must be a string of one-digit integers.
		"""
		nextChecker = 'X'  # start by playing 'X'
		for colChar in moveString:
			col = int(colChar)
			if 0 <= col <= self.width:
				self.addMove(col, nextChecker)
			if nextChecker == 'X':
				nextChecker = 'O'
			else:
				nextChecker = 'X'

	def allowsMove(self, col):
		"""
		:param c: column where move is going to be added
		:return: boolean, whether a move can be made in that column
		"""
		NCOL = self.width
		if col < 0 or col > NCOL:
			return False
		else:
			for row in range(0, self.height):
				if self.data[row][col] == ' ':
					return True
			return True

	def isFull(self):
		"""
		:return: Boolean of whether board is full or not
		"""
		for row in range(self.height):
			for col in range(self.width):
				if self.data[row][col] == ' ':
					return False
		return True

	def delMove(self, c):
		for row in range(self.height):
			if self.data[row][c] != ' ':
				self.data[row][c] = ' '
				print(self)
				return




	def winsFor(self, ox):
		"""
		:param ox: 'O' or 'X'
		:return: whether any wins exist for a certain player
		"""
		for row in range(self.height):
			for col in range(self.width):
				if inarow_Neast(ox, row, col, self.data, 4):
					return True
				if inarow_Nsouth(ox, row, col, self.data, 4):
					return True
				if inarow_Nsoutheast(ox, row, col, self.data, 4):
					return True
				if inarow_Nnortheast(ox, row, col, self.data, 4):
					return True
		return False

	def hostgame(self):
		"""
		plays the game until someone wins or there is a tie
		"""
		print("Welcome to Connect Four!")
		print(self)
		while not self.isFull():
			xmove = int(input("X's Move"))
			if not self.allowsMove(xmove):
				print("That is not a valid move")
				xmove = int(input("X's Move"))
				self.addMove(xmove, 'X')
			else:
				self.addMove(xmove, 'X')
			print(self)
			if self.winsFor('X'):
				print("X wins!")
				break
			if self.isFull():
				break
			omove = int(input("O's Move"))
			if not self.allowsMove(omove):
				print("That is not a valid move")
				omove = int(input("O's Move"))
				self.addMove(omove, 'O')
			else:
				self.addMove(omove, 'O')
			print(self)
			if self.winsFor('O'):
				print("O wins!")
				break
		if self.isFull():
			print("The board is full, but there are no winners.")
			print("It's a tie!")

	def colsToWin(self, ox):
		"""
		:param ox: 'X' or 'O'
		:return: a list with column indexes of winning moves
		"""
		LoWM = []
		for col in range(self.width):
			if self.allowsMove(col):
				self.addMove(col, ox)
				if self.winsFor(ox):
					LoWM += [col]
					self.delMove(col)
				else:
					self.delMove(col)
		return LoWM


	def getHumanMove(self):
		column = input("X: Which column will your move go?")
		if self.allowsMove(column):
			self.data[column][self.getLowestEmptySpace(column)] = 'X'
			print(self)
			return


	def aiMove(self, ox):
		potentialMoves = self.getPotentialMoves(ox, 2)
		# get the best fitness from the potential moves
		bestMoveFitness = -1
		for i in range(self.width):
			if potentialMoves[i] > bestMoveFitness and self.allowsMove(i):
				bestMoveFitness = potentialMoves[i]
		# find all potential moves that have this best fitness
		bestMoves = []
		for i in range(len(potentialMoves)):
			if potentialMoves[i] == bestMoveFitness and self.allowsMove(i):
				bestMoves.append(i)
		return random.choice(bestMoves)


	def getPotentialMoves(self, ox, ply):
		if ply == 0 or self.isFull():
			return [0] * self.width

		if ox == 'X':
			enemyTile = 'O'
		else:
			enemyTile = 'X'
		# Figure out the best move to make.
		potentialMoves = [0] * self.width
		for row in range(self.width):
			copyBoard = copy.deepcopy(self)
			if not copyBoard.allowsMove(row):
				continue
			else:
				copyBoard.addMove(row, ox)
				if copyBoard.winsFor(ox):
				# a winning move automatically gets a perfect fitness
					potentialMoves[row] = 1
					break  # don't bother calculating other moves
				else:
				# do other player's counter moves and determine best one
					if copyBoard.isFull():
						potentialMoves[row] = 0
					else:
						for counterMove in range(self.width):
							copyBoard2 = copy.deepcopy(copyBoard)
							if not copyBoard2.allowsMove(counterMove):
								continue
							copyBoard2.addMove(counterMove, enemyTile)
							if self.winsFor(ox):
								# a losing move automatically gets the worst fitness
								potentialMoves[row] = -1
								break
							else:
								# do the recursive call to getPotentialMoves()
								results = copyBoard2.getPotentialMoves(ox, ply - 1)
								potentialMoves[row] += (sum(results) / self.width) / self.width
		return potentialMoves

class Player:
	def __init__(self, ox, tbt, ply):
		self.ox = ox
		self.tbt = tbt
		self.ply = ply

	def __repr__(self):
		s = "Player for " + self.ox + "\n"
		s += "  with tiebreak type: " + self.tbt + "\n"
		s += "  and ply == " + str(self.ply) + "\n\n"
		return s

	def oppCh(self):
		if self.ox == 'X':
			return 'O'
		else:
			return 'X'

	def scoreBoard(self, b):
		score = 0
		if b.winsFor(self.ox):
			score =+ 100.0
		elif b.winsFor(self.oppCh()):
			score += 0.0
		else:
			score += 50.0
		return score

	def tiebreakMove(self, scores):
		LoMaxIndx = []
		index = 0
		for i in scores:
			if i == max(scores):
				LoMaxIndx += [index]
			index += 1
		if self.tbt == 'LEFT':
			return LoMaxIndx[0]
		elif self.tbt == 'RIGHT':
			return LoMaxIndx[-1]
		else:
			return random.choice(LoMaxIndx)

	def scoresFor(self, b):
		scores = [50]*b.width
		for column in range(b.width):
			if not self.allowsMove(column):
				scores[column] += -1
			if self.winsFor(self.ox):
				scores[column] += 100
			if self.winsFor(self.oppCh(self.ox)):
				scores[column] += 0
			if self.ply == 0:
				scores[column] += 0
			else:
				self.ply -= 1
				self.addMove(column)
				scores[column] += self.scoresFor()