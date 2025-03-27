---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
myst:
  substitutions:
    ref_test: 1
---

# <i class="fa-solid fa-puzzle-piece"></i> Regularization

Building on subset selection, an alternative approach is to include all *p* predictors in the model but apply regularization—shrinking the **coefficient estimates toward zero** relative to the least squares estimates. This reduces model complexity without fully discarding variables. Though it introduces some bias, it often lowers variance and improves test performance. 

3 approaches are commonly used to regularize the predictors:
- Ridge Regression (L2 Regression)
- Lasso Regression (L1 Regression)
- Elastic Net Regression


----------------------------------------------------------------
### *Todays data with many predictors - Hitters dataset*
For pracitcal demonstration, we will use again the `Hitters` dataset. 

```{code-cell} 
# import packages
import statsmodels.api as sm 
import matplotlib.pyplot as plt
import seaborn as sns

# get dataset
hitters = sm.datasets.get_rdataset("Hitters", "ISLR").data

# keeping a total of 10 variables - the outcome Salary and 9 predictors.
# Keeping only Salary and 9 predictors
hitters_subset = hitters[["Salary", "CHits", "CAtBat", "CRuns", "CWalks", "Assists", "Hits", "HmRun", "Years", "Errors"]].copy()

# make sure the rows, containing missing values, are dropped
hitters_subset.dropna(inplace=True)

hitters_subset.head()
```
```{code-cell} 
from sklearn.model_selection import train_test_split

# Defining target and features
X = hitters_subset.drop(columns=["Salary"])
y = hitters_subset["Salary"]

# split into training and test data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```
Let's also take a look at the correlation between predictors to check for potential multicollinearity, which can affect the stability of linear regression models.

```{code-cell} 
# Correlation heatmap for hitters_subset
plt.figure(figsize=(8, 5))
sns.heatmap(hitters_subset.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Hitters Subset", fontsize=16)
plt.show()
```
The heatmap reveals strong correlations between several predictors, indicating multicollinearity — making ridge regression an appropriate modeling choice. While subset selection can reduce overfitting, it may still be unstable in the presence of multicollinearity, as it tends to arbitrarily select among highly correlated predictors.


## TO MICHA: SHOULD WE INCLUDE THIS? Makind subset selection unbrauchbar for this dataset


## Ridge Regression
To understand ridge regression, lets have a look at the formula. As you will see ridge regression is very similiar to ordinary least squares fitting, but includes an tuning parameter that needs to be determined seperately.

```{margin}
Lamda is a tuning parameter that controls the strength of the penalty! 
```

$$
 \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2
$$

Where: 
- $ \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $ is the **residual sum of squares (RSS)**  

- $ \lambda \sum_{j=1}^{p} \beta_j^2 $ is the **L2 penalty** on the coefficients 



```{admonition} The tuning parameter λ
:class: note 

λ controls the degree of regulariztation and the relative impact of the penalty on the parameter estimates.

- λ=0: penalty term has no effect (Ridge Regression will produce least square estimates)
- As λ increases, the impact of the shirnkage penalty grows

 Thus, selecting a good value of lambda is crucial. For this, we can use cross-validation.
```
**Step 1:** 
Before we implement ridge regression, we need to standardize the variables since ridge regression is senstivite to scaling. This can be done by using `StandardScaler`from `sklearn`package. 

```{code-cell} 
# scale the data to have mean 0 and stdev 1
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Use the same scaler to transform test data
X_test_scaled = scaler.transform(X_test)
# This ensures both sets are standardized consistently, but only the training data 
# is used to compute the scaling parameters.
```

**Step 2:**
Next, we set up a range of values for λ. The graph nicely visualize how the beta values change with increasing lamda. 

```{code-cell} ipython3
:tags: [remove-input]

from sklearn.linear_model import Ridge

#initialize list to store coefficient values
coef=[]
alphas = range(1,40)

for a in alphas:
  ridgereg=Ridge(alpha=a)
  ridgereg.fit(X_train_scaled,y_train)
  coef.append(ridgereg.coef_)

# Make plot of Beta as a function of Lamda
fig=plt.figure()
ax=fig.add_subplot(111)
ax.plot(alphas,coef)
ax.set_xlabel('Lambda (Regularization Parameter)')
ax.set_ylabel('Beta (Predictor Coefficients)')
ax.set_title('Ridge Coefficients vs Regularization Parameters')
ax.axis('tight')
```

```{code-cell}
# set range 
lambda_range= range(1,40)
```

**Step 3:** 
For each value of lambda, ridge regression is performed on the training data. Cross-validation is then used to identify the optimal lambda that minimizes prediction error, and the final model is fitted using this best value.

```{margin}
In 'scikit-learn' the lamda parameter is called alpha! Don't get confused by that.
```

```{code-cell} 
from sklearn.linear_model import RidgeCV
import pandas as pd

# Fit Ridge regression through cross validation
ridge_cv = RidgeCV(alphas=lambda_range, store_cv_values=True)
ridge_cv.fit(X_train_scaled, y_train) 

print(f"The optimal alpha value for our analysis ends up being {ridge_cv.alpha_}.")

# Create a DataFrame to display predictor names and their corresponding coefficients
coef_table = pd.DataFrame({
    'Predictor': X_train.columns,
    'Ridge Coefficient': ridge_cv.coef_
})

# Sort table by absolute value of Ridge Coefficient
coef_table = coef_table.reindex(coef_table['Ridge Coefficient'].abs().sort_values(ascending=False).index)

print(coef_table)

```
