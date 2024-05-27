import  csv
import  sys
import  os.path

value = []

with open("indexes.csv", 'r') as csv_file :
    try :
        dict_val = csv.reader(csv_file, delimiter = ",")
        for row in dict_val :
            value.append(row)
    except :
        sys.exit("Error: File {:} cannot be read".format(csv_file))

check = False
while check == False :
    mileage = input("Enter mileage: ")
    try :
        number_mileage = float(mileage) - 0
        if (number_mileage >= 0) :
            check = True
        else :
            print ("Error: negative mileage? Try again.")
    except :
        print ("Error: not a number, try again.")

# teta0 + teta1 * mileage = price is the equation of the linear regression model that was trained in the trainer.py file and saved in the indexes.csv file
# the price is the value that we want to predict based on the mileage value that the user entered in the input field of the predictor.py file
# the price is the dependent variable and the mileage is the independent variable
# how to predict the price value based on the mileage value: 
# 1. load the trained model from the indexes.csv file
# 2. calculate the price value using the equation of the linear regression model
# 3. print the predicted price value to the user
# the trained model is saved in the indexes.csv file as a list of lists
# the first element of the list is the teta0 parameter of the linear regression model
# the second element of the list is the teta1 parameter of the linear regression model
# the teta0 and teta1 parameters are the parameters that were found by the gradient descent algorithm in the trainer.py file
# teta0 is a slope parameter of the linear regression model
# teta1 is an intercept parameter of the linear regression model
print (float(value[0][0]) + (float(value[0][1]) * number_mileage))