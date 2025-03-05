Linear regression is a machine learning algorithm that attempts to determine a linear relationship between a set of inputs and the output value.
$$
y_i = \beta_{0} + \beta_{1} x_{i1} + \cdots + \beta_{p} x_{ip} + \varepsilon_i
 = \mathbf{x}^\mathsf{T}_i\boldsymbol\beta + \varepsilon_i,
 \qquad i = 1, \ldots, n
$$

Here, we assume that $$\varepsilon_i$$'s are identical and independent normal variables. Using the training data, linear regression seeks the estimates of the coefficients that minimizes the least squares error. 
Note that the "linear" in linear regression refers to linearity of the coefficients, which means that input values $$ x_{i}$$'s can be transformed (e.g. log-transform). 


This is one of the most intuitive and simple algorithms. However, the simplicity means that a linear model may not be appropriate for many data sets. Linear regression assumes that data points are independent, which would not work for something like a time series data. 

In the following notebook, we will apply linear regression to analyze the relationship between length and maximum speed of roller coasters. First, we will fit an initial model and do some diagnostics to see if the model is appropriate. Then, we will separate the data into 2 different categories and refit linear models.
