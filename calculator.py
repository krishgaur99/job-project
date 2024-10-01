import math

print("1. addition")
print("2. subtraction")
print("3. multiplication")
print("4. division")
print("5. square root")
print("6. square")
print("7. cube root")
print("8. cube")
print("9. factorial")
print("10. to the power n")

while True:
          try:
                    a = input("enter your choice")
                    if(a == 1 or "addition"):
                            x = int(input("enter your first number"))
                            y = int(input("enter your second number"))
                            print(x+y)
                    elif(a == 2 or "subtraction"):
                            x = int(input("enter your first number"))
                            y = int(input("enter your second number"))
                            print(x-y)
                    elif(a == 3 or "multiplication"):
                            x = int(input("enter your first number"))
                            y = int(input("enter your second number"))
                            print(x*y)
                    elif(a == 4 or "division"):
                            x = int(input("enter your first number"))
                            y = int(input("enter your second number"))
                            print(x/y)
                    elif(a == 5 or "square root"):
                            x = int(input("enter your number"))
                            print(x**0.5)
                    elif(a == 6 or "square"):
                            x = int(input("enter your number"))
                            print(x**x)
                    elif(a == 7 or "cube root"):
                              x = int(input("enter your number"))
                              print(x**0.33)
                    elif(a == 8 or "cube"):
                            x = int(input("enter your number"))
                            print(x**x**x)  
                    elif(a == 9 or "factorial"):
                            x = int(input("enter your number"))
                            for i in range(x):
                                    print(x*x)
                    elif(a == 10 or "powern"):
                            x = int(input("enter your base number"))
                            y = int(input("enter your power number"))          
                    else:
                            print("lolem and lol")    
          except:
                  print("wrong input")         