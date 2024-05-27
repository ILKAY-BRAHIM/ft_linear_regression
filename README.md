# ft_linear_regression

> [!NOTE]
> install requirements using pip install -r requirements.txt

## Usage

Clone and change directory to project, then
```bash
python -m venv env
source myenv/bin/activate
python trainer.py data.csv
python predictor.py
```
## algo steps
### step 1 Load dataset
    the dataset contains two columns: the price of the car and the mileage of the car
    the price of the car is the dependent variable and the mileage of the car is the independent variable
    the goal is to train a linear regression model that will predict the price of the car based on the mileage of the car
    using libraries like pandas to load the dataset is a good practice because it makes the code more readable and easier to maintain
### step 2 Normalize data (if necessary)
    i need to normalize the data because the mileage values are much higher than the price values and this can lead to a slow convergence of the gradient descent algorithm
    Normalized data is the data that has been scaled to a common scale. The formula is: (x - mean) / std dev in order to have a mean of 0 and a standard deviation of 1  (z-score normalization)
    performing this operation will scale the data to have a mean of 0 and a standard deviation of 1
    perporming this operation will also make the gradient descent algorithm converge faster because the cost function will be more symmetric and the gradient descent algorithm will be able to find the minimum value of the cost function faster
### step 3 gradient descent and const fuction
  1 Gradient Descent is used to find the local minimum of a differentiable function. The basic idea is to take steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point. <br>
        ---Initialize parameters (weights, biases) with random values. <br>
        ---Compute the gradient of the cost function with respect to each parameter.<br>
        ---Update the parameters by subtracting the gradient multiplied by a learning rate from the current parameter values.<br>
        ---Repeat steps 2 and 3 until convergence (the changes in the cost function are very small).<br>
    
<img width="281" alt="Screen Shot 2024-05-27 at 12 27 54 PM" src="https://github.com/ILKAY-BRAHIM/ft_linear_regression/assets/88441828/58f938d4-52ff-4fb7-8b8d-465335a6cce8"> <br>

  2 A cost function (or loss function) measures how well the model's predictions match the actual data. In machine learning, a common cost function for linear regression is the Mean Squared Error (MSE).
        
  <img width="326" alt="Screen Shot 2024-05-27 at 12 24 24 PM" src="https://github.com/ILKAY-BRAHIM/ft_linear_regression/assets/88441828/28f7bab3-2d83-4f42-b737-87617315e135"> <br>
  ### step4 Denormalize parameters
  --- the parameters theta0 and theta1 that were found by the gradient descent algorithm are the parameters of the linear regression model that was trained on the normalized data <br>
  --- in order to make predictions on the original data i need to denormalize the parameters theta0 and theta1 <br>
  --- to denormalize the parameters theta0 and theta1 i need to multiply theta1 by the standard deviation of the mileage and then divide by the mean of the mileage <br>
  --- i also need to divide theta0 by the standard deviation of the mileage <br>
  --- the reason i need to divide theta0 by the standard deviation of the mileage is because the mean of the normalized data is 0 and the standard deviation of the normalized data is 1 <br>
  --- the reason i need to multiply theta1 by the standard deviation of the mileage and then divide by the mean of the mileage is because the mean of the normalized data is 0 and the standard deviation of the normalized data is 1 <br>
  ### step 5 save slope(teta1) and intercet(teta0)
  Plot the data and the linear regression model <br>
<img width="634" alt="Screen Shot 2024-05-27 at 12 39 38 PM" src="https://github.com/ILKAY-BRAHIM/ft_linear_regression/assets/88441828/3a3479f4-8a7c-49dd-814e-1bcb3fd9dcf1"> <br>

  Plot the cost function history <br>
  <img width="639" alt="Screen Shot 2024-05-27 at 12 39 25 PM" src="https://github.com/ILKAY-BRAHIM/ft_linear_regression/assets/88441828/4b2e8772-5702-47c6-bf7e-475b5c380d3c">

