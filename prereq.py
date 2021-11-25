import sys

class Course:    
    def __init__(self, name, courseID,prereq):
        self.name = name  
        self.courseID=courseID
        self.prereq=prereq
    
def course_And(prereq, student_prev):

    for x in range(len(student_prev)):
        if student_prev[x].courseID==prereq.courseID:
            return True

    return False


def course_Or(prereq,student_prev):
    for x in range(len(student_prev)):
        for y in range(len(prereq)):
            if student_prev[x].courseID==prereq[y].courseID:
                return True

    return False
    

def main():
    success = True
    a = Course("Data 101", "01:198:142",None)
    t =  Course("Intro to CS", "01:198:111",None)
    b = Course("Data management", "01:198:210", [(2,[a,t])])
    student_prev = [a,b,t]
    
    test= Course("Test","01:000:000",[(1,t),(2,[a,b])])

    for x in range(len(test.prereq)):
        temp=test.prereq[x]
        
        if temp[0]==1:
            success = course_And(temp[1], student_prev)
            if success==False:
                print("You have not taken "+temp[1].name)
                sys.exit()
        
        elif temp[0]==2:
            success = course_Or(temp[1], student_prev)
            if success==False:
                print("You have not taken "+temp[1][0].name)
                sys.exit()
            
        else:
            success=True


    print("You may take "+test.name)
    sys.exit()
            
    
if __name__ == "__main__":
    main()



