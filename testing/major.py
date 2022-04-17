import sys

def main():
    tempPrev = "01:198:111;01:198:112;01:198:205;01:198:206;01:198:211;01:198:344;01:750:203"       #from input
    stud_prev = tempPrev.split(";")
    
    track = "major"

    ret = []
    ctr = 0


    if track == 'major':
        tempPrereqCS = "01:198:111;01:198:112;01:198:205;01:198:206;01:198:211;01:198:344"
        prereqCS = tempPrereqCS.split(";")

        for i in range (len(prereqCS)):
            temp = prereqCS[i]
            for j in range (len(stud_prev)):
                if stud_prev[j] == temp:
                    ret.append(temp)
        if len(ret) != 6:
            print "Missing CS courses"
        ret = []


        tempPrereqMath = "01:640:151;01:640:152;01:640:250"
        prereqMath = tempPrereqMath.split(";")

        for i in range (len(prereqMath)):
            temp = prereqMath[i]
            for j in range (len(stud_prev)):
                if stud_prev[j] == temp:
                    ret.append(temp)
        if len(ret) != 3:
            print "Missing Math courses"
        ret = []



        tempScience = "01:750:203;01:750:204;01:750:205;01:750:206;01:160:159;01:160:160;01:160:171"
        science = tempScience.split(";");

        physics = []
        chem = []
        for i in range (len(science)):
            temp = science[i]
            for j in range (len(stud_prev)):
                if stud_prev[j] == temp:
                     if temp[3]=='7':
                        physics.append(temp)
                     else:
                        chem.append(temp)


        phys_complete = 0   #0=completed, 1=incomplete,determined, 2=incomplete,undetermined
        chem_complete = 0
        
        if len(physics) < 4:
            if len(chem) < len(physics):
                phys_complete = 1
            else:
                phys_complete = 2
        if len(chem) < 3:
            if len(physics) < len(chem):
                chem_complete = 1
            else:
                chem_complete = 2

        if phys_complete == 1:
            print "Missing Physics courses"
        if chem_complete == 1:
            print "Missing Chem courses"
        if phys_complete == 2 and chem_complete == 2:
            print "Missing Science courses"

         
        tempElective = "01:198:210;01:198:213;01:198:214;01:198:314;01:198:323;01:198:324;01:198:334;01:198:336;01:198:352;01:198:411;01:198:415;01:198:416;01:198:417;01:198:419;01:198:424;01:198:425;01:198:428;01:198:431;01:198:437;01:198:439;01:198:440;01:198:442;01:198:452;01:198:460;01:198:461;01:198:462"
        elect = tempElective.split(";")

        for i in range (len(elect)):
            temp = elect[i]
            for j in range (len(stud_prev)):
                if stud_prev[j] == temp:
                    ret.append(temp)

        if len(ret) < 7:
            print "Must take " + str(7-len(ret)) + " more CS electives"




    else:
        tempPrereq = "01:198:111;01:198:112;01:198:205;01:198:206;01:198:210;01:198:211;01:198:213;01:198:214;01:198:314;01:198:323;01:198:324;01:198:334;01:198:336;01:198:344;01:198:352;01:198:411;01:198:415;01:198:416;01:198:417;01:198:419;01:198:424;01:198:425;01:198:428;01:198:431;01:198:437;01:198:439;01:198:440;01:198:442;01:198:452;01:198:460;01:198:461;01:198:462"
        prereq = tempPrereq.split(";")

        for i in range (len(prereq)):
            temp = prereq[i]
            for j in range (len(stud_prev)):
                if stud_prev[j] == temp:
                    ret.append(temp)
                    if temp[7] == '3' or temp[7] == '4':
                        ctr += 1
    

        #print ret

        if len(ret) < 6:
            print "Must take " + str(6-len(ret)) + " more CS courses"
        if ctr < 2:
            print "Must take " + str(2-ctr) + " more 300 or 400 level courses"
        

    sys.exit()


if __name__ == "__main__":
    main()



