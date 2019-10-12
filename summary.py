def summary(T_age, T_gender, T_number, Detected, date):
    
    f = open("summary/summary_%s.txt" % date,"w+")
    
    f.write("\n_______________Summary_______________\n")
    f.write("                Dated : %s \n\n\n\n" % (date) )
    f.write("Total number of people just detected = %d\n\n" % Detected)
    f.write("Total number of people detected & Predicted = %d\n\n" % T_number)
    f.write("Gender: \n Male = %d     Female = %d\n\n" % (T_gender[0],T_gender[1]))
    f.write("Age: \n Infants = % d\n Age from 4 to 6 = %d\n Age from 8 to 12 = %d\n Age from 15 to 20 = %d\n Age from 25 to 32 = %d\n Age from 38 to 43 = %d\n Age from 48 to 53 = %d\n Age from 60 to 100 = %d\n" % (
        T_age[0], T_age[1], T_age[2], T_age[3], T_age[4], T_age[5], T_age[6], T_age[7]))
