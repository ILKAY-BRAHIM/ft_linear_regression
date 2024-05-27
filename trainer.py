import pandas as pd
import matplotlib.pyplot as plt

def calculateCostSquareError(theta0, theta1, mileage, price):
    """
    Function that calculates the cost square error for a linear regression.
    """
    m = len(mileage)
    # the cost function is the function that we want to minimize in order to find the best parameters theta0 and theta1 that will make the best predictions
    # the cost function is the sum of the square errors between the predicted value and the actual value of the price of the car for each data point in the dataset
    return (1 / (2 * m)) * sum((theta1 * mileage + theta0 - price) ** 2)

def updateParams(mileage, price, theta0, theta1, lerning_rate):
    """
    Function that simultaneously updates theta0 and theta1 using the partial derivatives.
    """
    m = len(mileage)
    # why do we need to calculate the error? the error is the difference between the predicted value and the actual value of the price of the car for each data point in the dataset
    # the error is used to update the parameters theta0 and theta1 in order to minimize the cost function
    # the cost function is the function that we want to minimize in order to find the best parameters theta0 and theta1 that will make the best predictions
    # the cost function is the sum of the square errors between the predicted value and the actual value of the price of the car for each data point in the dataset
    # The error is used to compute the gradient (partial derivatives) of the cost function with respect to the parameters theta0 and theta1 These gradients tell us how to adjust the parameters to reduce the error.
    # The error is also used to compute the cost function, typically the Mean Squared Error (MSE), which measures the overall performance of the model.
    # intercet = theta0, slope = theta1
    error = theta1 * mileage + theta0 - price
    temp_theta0 = theta0 - lerning_rate * (1 / m) * sum(error)
    temp_theta1 = theta1 - lerning_rate * (1 / m) * sum(error * mileage)
    return temp_theta0, temp_theta1

def executeGradientDescentAlgo(mileage, price, lerning_rate, num_iterations):
    """
    Executes the gradient descent algorithm for a specified number of iterations.
    """
    theta0 = 0
    theta1 = 0
    cost_history = []

    for _ in range(num_iterations):
        theta0, theta1 = updateParams(mileage, price, theta0, theta1, lerning_rate)
        cost = calculateCostSquareError(theta0, theta1, mileage, price)
        cost_history.append(cost)

    return theta0, theta1, cost_history

def main():
    # Load dataset
    # the dataset contains two columns: the price of the car and the mileage of the car
    # the price of the car is the dependent variable and the mileage of the car is the independent variable
    # the goal is to train a linear regression model that will predict the price of the car based on the mileage of the car
    # using libraries like pandas to load the dataset is a good practice because it makes the code more readable and easier to maintain

    df = pd.read_csv('data.csv')
    price = df['price']
    mileage = df['km']
    
    # -----------------------------------------------------Normalize data (if necessary)----------------------------------------------
    # Mean of the data set is the average value of the data set. The formula is: sum(x) / n
    mileage_mean = mileage.mean()
    # Standard deviation of the data set is a measure of how spread out the values are from the mean value (the average value). the formula is: sqrt((sum((x - mean)^2) / n)), n = number of elements
    mileage_std = mileage.std()
    # Normalized data is the data that has been scaled to a common scale. The formula is: (x - mean) / std dev in order to have a mean of 0 and a standard deviation of 1  (z-score normalization)
    # i need to normalize the data because the mileage values are much higher than the price values and this can lead to a slow convergence of the gradient descent algorithm
    # to normalize the data i need to subtract the mean value from each element and then divide by the standard deviation
    # performing this operation will scale the data to have a mean of 0 and a standard deviation of 1
    # perporming this operation will also make the gradient descent algorithm converge faster because the cost function will be more symmetric and the gradient descent algorithm will be able to find the minimum value of the cost function faster
    mileage_normalized = (mileage - mileage_mean) / mileage_std

    
    # ----------------------------------------------------Hyperparameters--------------------------------------------------------------
    lerning_rate = 0.01
    num_iterations = 1000

    # ----------------------------------------------------Run gradient descent-----------------------------------------------------------
    theta0, theta1, cost_history = executeGradientDescentAlgo(mileage_normalized, price, lerning_rate, num_iterations)
    
    # ----------------------------------------------------Denormalize parameters-----------------------------------------------------------
    # the parameters theta0 and theta1 that were found by the gradient descent algorithm are the parameters of the linear regression model that was trained on the normalized data
    # in order to make predictions on the original data i need to denormalize the parameters theta0 and theta1
    # to denormalize the parameters theta0 and theta1 i need to multiply theta1 by the standard deviation of the mileage and then divide by the mean of the mileage
    # i also need to divide theta0 by the standard deviation of the mileage
    # the reason i need to divide theta0 by the standard deviation of the mileage is because the mean of the normalized data is 0 and the standard deviation of the normalized data is 1
    # the reason i need to multiply theta1 by the standard deviation of the mileage and then divide by the mean of the mileage is because the mean of the normalized data is 0 and the standard deviation of the normalized data is 1

    theta0 = theta0 - (theta1 * mileage_mean / mileage_std)
    theta1 = theta1 / mileage_std
    # ----------------------------------------------------Print results-------------------------------------------------------------------
    print("Training model result (price = theta0 + theta1 * mileage):")
    print("theta0:", theta0)
    print("theta1:", theta1)

    # ----------------------------------------------------Save parameters to a file-------------------------------------------------------
    with open('indexes.csv', 'w') as file:
        file.write(str(theta0) + ',' + str(theta1))
    
    # -----------------------------------------------------Plot results--------------------------------------------------------------------
    plt.figure(figsize=(14, 5))
    
    # -------------------------------------Plot the data and the linear regression model-------------------------------------------------
    plt.subplot(1, 2, 1)
    plt.scatter(mileage, price, label='Data points')
    plt.plot(mileage, theta1 * mileage + theta0, color='green', label='Linear regression')
    plt.xlabel('Mileage')
    plt.ylabel('Price')
    plt.title('Mileage vs. Price')
    plt.legend()
    
    # ---------------------------------------------Plot the cost function history-----------------------------------------------------------
    plt.subplot(1, 2, 2)
    plt.plot(range(num_iterations), cost_history, color='green')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost function history')
    
    plt.show()

if __name__ == "__main__":
    main()
