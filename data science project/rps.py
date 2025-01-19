import random

print("let play the game rock paper scissor ")
print("")
while True:
          print("what do yuo want to choose")
          print("1. rock ")
          print("2.paper ")
          print("3. scissors")
          x = input("enter your choice ")
          if(x == 1):
                  y = "rock"
          elif(x == 2):
                  y = "paper"
          elif(x == 3):
                  y = "scissors"

          print("your choice is ",y)
          print("now it's computer turn")
          
          z = random.randint(1,3)
          if (z == 1):
                  a = "rock"
          elif (z == 2):
                  a = "paper"
          elif (z == 3):
                  a = "scissors"
          print("computer choice is ",a)

# determine the winner

          if ( y == a):
                  result = "draw"
          elif( y == 1 and a == 2 or y == 2 and a == 1):
                  result == "it's paper"                
          elif(y == 2 and a == 3  or y == 3 and a == 2):
                  result == "it's scissor"
          elif(y == 3 and a == 1 or y == 1 and a == 3):
                  result = "it's rock"

# declaring result 

          if (result == y):
                  print("user wins")
          elif (result == a):
                  print("computer wins")
          elif (result == "draw"):
                  print("it,s draw")   
