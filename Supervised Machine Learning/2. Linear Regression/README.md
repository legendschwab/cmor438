**Linear regression** is a machine learning algorithm that attempts to determine a linear relationship between a set of inputs and their corresponding output values.  

$$
y_i = \beta_{0} + \beta_{1} x_{i1} + \cdots + \beta_{p} x_{ip} + \varepsilon_i
= \mathbf{x}^\mathsf{T}_i\boldsymbol\beta + \varepsilon_i, \quad i = 1, \ldots, n
$$

Here, we assume that $\varepsilon_i$ are independent and identically distributed (i.i.d.) normal variables. Using the training data $$(\overrightarrow{x}_i, y_i)$$ linear regression seeks to estimate the coefficients that minimize the least squares error, which is the sum of squared errors between observed data points and the predicted values.

Note that the term **"linear"** in linear regression refers to the linearity of the coefficients, meaning that input values $$x_{i}$$ can be transformed (e.g., log transformation) while still maintaining the linearity of the model. 

This is one of the most intuitive and simple algorithms. However, its simplicity also means that a linear model may not be appropriate for many datasets. For instance, **linear regression assumes that data points are independent**, which makes it unsuitable for time series data or other scenarios where observations are correlated.  

In the following notebook, we will apply **linear regression** to analyze the relationship between **the height and maximum speed of roller coasters**. First, we will fit an initial model and perform diagnostic checks to assess its appropriateness. Then, we will remove some outliers based on the type of the roller coaster to improve the linear model. 
