l1=[]
while True:
        print("1. delete")
        print("2.add")
        x = (input("bata kya karna hai"))
        if (x == "1" or x == "hatao"):
                y = input("index value se ya name value se")
                if(y == "index value se" or "index value"):
                        q = int(input("index value bata"))
                        try:   
                           l1.pop(q)
                           print(l1)
                        except:
                                print("empty list")   
                elif(x == "name value se" or "naem value"):
                           w = int(input("name value bata"))
                           l1.remove(w)     
                else:
                        print("error value given")           
        elif(x == "2" or x == "daldo"):
                s = input("kya dalna chahate ho")
                l1.append(s)
                print(l1)
        else:
                print("chal be")        