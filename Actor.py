

import numpy as np

class Blob:
    def __init__(self, size):
        self.size = size
        self.x = np.random.randint(0, size)
        self.y = np.random.randint(0, size)

    def __str__(self):
        return f"Blob ({self.x}, {self.y})"

    def __sub__(self, other):
        return (self.x-other.x, self.y-other.y)

    def __eq__(self, other):  # This will allow us to see if the Blobs are over each other -
        return self.x == other.x and self.y == other.y

    #  Available Moves to Make
    def action(self, choice):
        '''
        Gives us 9 total movement options. (0,1,2,3,4,5,6,7,8)
        '''
        if choice == 0:
            #  Diag - Up / Right
            self.move(x=1, y=1)
        elif choice == 1:
            #  Diag - Down / Left
            self.move(x=-1, y=-1)
        elif choice == 2:
            #  Diag - Up / Left
            self.move(x=-1, y=1)
        elif choice == 3:
            #  Diag - Down / Right
            self.move(x=1, y=-1)
        elif choice == 4:
            #  Right Only
            self.move(x=1, y=0)
        elif choice == 5:
            #  Left Only
            self.move(x=-1, y=0)
        elif choice == 6:
            #  Up Only
            self.move(x=0, y=1)
        elif choice == 7:
            #  Down Only
            self.move(x=0, y=-1)
        elif choice == 8:
            #  Don't Move
            self.move(x=0, y=0)

    def move(self, x=False, y=False):

        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > self.size-1:
            self.x = self.size-1
        if self.y < 0:
            self.y = 0
        elif self.y > self.size-1:
            self.y = self.size-1