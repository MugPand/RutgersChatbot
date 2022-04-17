import sys

def main():
    tempPrereq = ""     #from database
    prereq = tempPrereq.split(";")

    tempPrev = ""       #from input
    stud_prev = tempPrev.split(";")
    
    ret = []

    for i in range (len(prereq)):
        temp = prereq[i]
        
        if len(temp) == 10:
            contains = False
            for j in range (len(stud_prev)):
                if stud_prev[j] == temp:
                    contains = True
            if contains == False:
                ret.insert(temp)

        else:
            temp_list = temp.split("or")
            temp_ret = []

            for j in range (len(temp_list)):
                temp_item = temp_list[j]
                contains = False
                for k in range (len(stud_prev)):
                    if stud_prev[k] == temp_item:
                        contains = True
                if contains == False:
                    temp_ret.insert(temp_item)

            if len(temp_ret) != 0:
                ret.insert(temp_ret)


    sys.exit()


if __name__ == "__main__":
    main()



