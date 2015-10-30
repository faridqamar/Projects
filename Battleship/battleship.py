"""
This is a simple Battleship game to play in the console
The Battleship is 4 squares in length, and to hit you
must guess the row number and column number correctly 4 times.
The game ends when you sink the battleship.

 
the following are to come eventually:
- Make multiple battleships:
need to be careful because need to make sure that
battleships aren't placed on top of each other on the game board.
- Make game a two-player game.
- have more features like rematches, statistics and more!
"""

from random import randint

# initiate empty board
board = []

# create board with 10 rows and columns with "." in each square
for x in range(10):
    board.append(["."]*10)

# define a function to print the board nicely with row numbers
# and column numbers visible
def print_board(board):
    x = 1
    print "  1 2 3 4 5 6 7 8 9 10"
    for row in board:
        print x, " ".join(row)
        x += 1

print "***---------------------------***"
print "     Let's play Battleship!"
print "***---------------------------***"
print ""
print "Instructions: there is one battleship hidden"
print "in the board, of length 4. Insert the row"
print "number first and the column number after."
print ""
print_board(board)

# randomly pick the coordinates for the battleship
# 0 for horizontal, 1 for vertical
updown = randint(0, 1)
# pick either row or column from 1-10
rowcol = randint(1, 10)
# pick start of ship from 1-7
start = randint(1, 7)
# return x,y location of battleship
if updown == 0:
    random_col = [start, start+1, start+2, start+3]
    random_row = [rowcol, rowcol, rowcol, rowcol]
    #return [[[rowcol],[start]],[[rowcol][start+1]],[[rowcol],[start+2]],[[rowcol], [start+3]]]
elif updown == 1:
    random_row = [start, start+1, start+2, start+3]
    random_col = [rowcol, rowcol, rowcol, rowcol]        

#print "random_row", random_row
#print "random_col", random_col

end = 0
turn = 0
battleship_hit = 4
while (end < 1):
	# ask the user to guess row and column numbers
    guess_row = int(raw_input("Guess Row [1-10]:"))
    guess_col = int(raw_input("Guess Col [1-10]:"))

	# begin if sequence:
	# if coordinates are outside the board rerun loop
    if (guess_row < 1 or guess_row > 10) or (guess_col < 1 or guess_col > 10):
        print "Oops, that's not even in the ocean."
	# if coordinates already guessed, rerun loop
    elif (board[guess_row-1][guess_col-1] == "O" or board[guess_row-1][guess_col-1] == "X"):
        print "You guessed that one already."
    else:
		# if guess is correct, replace square with an X
        if (guess_row in random_row and guess_col in random_col):
            battleship_hit -= 1
            print "You hit my battleship!"
            print "hits left: %d" % battleship_hit
            board[guess_row-1][guess_col-1] = "X"
	    # if battleship sunk, end game
            if (battleship_hit == 0):
                print_board(board)
                print "Congratulations! You sunk my battleship!"
                break
        else:
            print "You missed my battleship!"
            board[guess_row-1][guess_col-1] = "O"
    print "Turn", turn+1
    print ""
    print_board(board)
    turn +=1
