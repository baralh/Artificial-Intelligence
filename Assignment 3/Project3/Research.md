# Links to good info on the web

[Polynomial Regression](https://towardsdatascience.com/polynomial-regression-bbe8b9d97491)

[Polynomial Regression Wikipedia page](https://en.wikipedia.org/wiki/Polynomial_regression)

[Uni-Variate Polynomial Regression in Python (from scratch)](https://towardsdatascience.com/implementation-of-uni-variate-linear-regression-in-python-using-gradient-descent-optimization-from-3491a13ca2b0) looks like a good reference.

[Another person's attempt in python (on GitHub) without using skLearn](https://github.com/pickus91/Polynomial-Regression-From-Scratch/blob/master/polynomial_regression.py)
___

To generate a higher-order equation (than linear) we can add powers of the original features as new features. 

The linear model,

<img src="https://github.com/coffee247/AI-Homework/blob/master/Project3/Images/CodeCogsEqn%20(1).png">

can be transformed to 

<img src="https://github.com/coffee247/AI-Homework/blob/master/Project3/Images/CodeCogsEqn%20(2).png">

This is still considered to be linear model as the coefficients/weights associated with the features are still linear. x² is only a feature. However the curve that we are fitting is quadratic in nature.

To generate polynomial features (here 2nd degree polynomial)
  * create another column in the feature table whose features are raised to the n^th power of the original column
  
  EXPLANATION:
  
  <img src="https://github.com/coffee247/AI-Homework/blob/master/Project3/Images/CodeCogsEqn%20(4).png">

----

# Normalizing the data 

* using min-max scaling

In min-max scaling each feature value is rescaled to be in the range from 0 to 1. The formula for applying min-max scaling to some attribute value x_i is:

<img src="https://github.com/coffee247/AI-Homework/blob/master/Project3/Images/CodeCogsEqn.png">

<img src="https://github.com/coffee247/AI-Homework/blob/master/Project3/Images/CodeCogsEqn%20(5).png">

