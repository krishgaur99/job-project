import random
i = random.randint(1,100)
while True:
          try:
                guess = 0
                x = int(input("enter your number"))
                if (x == i):
                        guess = 1
                        print("correct number")
                        break  
                elif(x > i):
                        guess = 1
                        print("your number to higher")
                elif(x < i):
                        guess = 1
                        print("your nuber is lower") 
                else:
                        print("wrong input")               
          except:
                  print("give no. not any thingelse")        