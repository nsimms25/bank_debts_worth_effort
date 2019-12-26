"""
Using data on expected and actual recovery amounts in order to predict which debts are are worth the work to recover.

Data is located in bank_data.csv Use Kruskal-Wallis, chi2 test and sm api in scipy stats in order to find the p_value
and finally the coefficient that relates to the worth of the debts.
"""

# Import modules
import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt

# Read in dataset
df = pd.read_csv('bank_data.csv')

# Print the first few rows of the DataFrame
df.head()

# Scatter plot of Age vs. Expected Recovery Amount
plt.scatter(x=df['expected_recovery_amount'], y=df['age'], c="g", s=2)
plt.xlim(0, 2000)
plt.ylim(0, 60)
plt.xlabel('Expected Recovery Amount')
plt.ylabel('Age')
plt.legend(loc=2)
plt.show()

# Compute average age just below and above the threshold
era_900_1100 = df.loc[(df['expected_recovery_amount'] < 1100) &
                      (df['expected_recovery_amount'] >= 900)]
by_recovery_strategy = era_900_1100.groupby(['recovery_strategy'])
by_recovery_strategy['age'].describe().unstack()

# Perform Kruskal-Wallis test
Level_0_age = era_900_1100.loc[df['recovery_strategy'] == "Level 0 Recovery"]['age']
Level_1_age = era_900_1100.loc[df['recovery_strategy'] == "Level 1 Recovery"]['age']
stats.kruskal(Level_0_age, Level_1_age)

# KruskalResult(statistic=3.4572342749517513, pvalue=0.06297556896097407)

# Number of customers in each category
crosstab = pd.crosstab(df.loc[(df['expected_recovery_amount'] < 1100) &
                              (df['expected_recovery_amount'] >= 900)]['recovery_strategy'],
                       df['sex'])
print(crosstab)

# Chi-square test
chi2_stat, p_val, dof, ex = stats.chi2_contingency(crosstab)
print(p_val)

'''
sex                Female  Male
recovery_strategy              
Level 0 Recovery       32    57
Level 1 Recovery       39    55
0.5377947810444592
'''

# Scatter plot of Actual Recovery Amount vs. Expected Recovery Amount
plt.scatter(x=df['expected_recovery_amount'], y=df['actual_recovery_amount'], c="g", s=2)
plt.xlim(900, 1100)
plt.ylim(0, 2000)
plt.xlabel("Expected Recovery Amount")
plt.ylabel("Actual Recovery Amount")
plt.legend(loc=2)
plt.show()

# Compute average actual recovery amount just below and above the threshold
by_recovery_strategy['actual_recovery_amount'].describe().unstack()

# Perform Kruskal-Wallis test
Level_0_actual = era_900_1100.loc[df['recovery_strategy'] == 'Level 0 Recovery']['actual_recovery_amount']
Level_1_actual = era_900_1100.loc[df['recovery_strategy'] == 'Level 1 Recovery']['actual_recovery_amount']
stats.kruskal(Level_0_actual, Level_1_actual)

# Repeat for a smaller range of $950 to $1050
era_950_1050 = df.loc[(df['expected_recovery_amount'] < 1050) &
                      (df['expected_recovery_amount'] >= 950)]
Level_0_actual = era_950_1050.loc[df['recovery_strategy'] == 'Level 0 Recovery']['actual_recovery_amount']
Level_1_actual = era_950_1050.loc[df['recovery_strategy'] == 'Level 1 Recovery']['actual_recovery_amount']
stats.kruskal(Level_0_actual, Level_1_actual)

# KruskalResult(statistic=30.246000000000038, pvalue=3.80575314300276e-08)

# Import statsmodels
import statsmodels.api as sm

# Define X and y
X = era_900_1100['expected_recovery_amount']
y = era_900_1100['actual_recovery_amount']
X = sm.add_constant(X)

# Build linear regression model
model = sm.OLS(y, X).fit()
predictions = model.predict(X)

# Print out the model summary statistics
print(model.summary())

'''
                              OLS Regression Results                              
==================================================================================
Dep. Variable:     actual_recovery_amount   R-squared:                       0.261
Model:                                OLS   Adj. R-squared:                  0.256
Method:                     Least Squares   F-statistic:                     63.78
Date:                    Wed, 11 Dec 2019   Prob (F-statistic):           1.56e-13
Time:                            05:29:16   Log-Likelihood:                -1278.9
No. Observations:                     183   AIC:                             2562.
Df Residuals:                         181   BIC:                             2568.
Df Model:                               1                                         
Covariance Type:                nonrobust                                         
============================================================================================
                               coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------------------
const                    -1978.7597    347.741     -5.690      0.000   -2664.907   -1292.612
expected_recovery_amount     2.7577      0.345      7.986      0.000       2.076       3.439
==============================================================================
Omnibus:                       64.493   Durbin-Watson:                   1.777
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              185.818
Skew:                           1.463   Prob(JB):                     4.47e-41
Kurtosis:                       6.977   Cond. No.                     1.80e+04
==============================================================================
'''

# Create indicator (0 or 1) for expected recovery amount >= $1000
df['indicator_1000'] = np.where(df['expected_recovery_amount'] < 1000, 0, 1)
era_900_1100 = df.loc[(df['expected_recovery_amount'] < 1100) &
                      (df['expected_recovery_amount'] >= 900)]

# Define X and y
X = era_900_1100[['expected_recovery_amount', 'indicator_1000']]
y = era_900_1100['actual_recovery_amount']
X = sm.add_constant(X)

# Build linear regression model
model = sm.OLS(y, X).fit()

# Print the model summary
print(model.summary())

'''
                              OLS Regression Results                              
==================================================================================
Dep. Variable:     actual_recovery_amount   R-squared:                       0.314
Model:                                OLS   Adj. R-squared:                  0.307
Method:                     Least Squares   F-statistic:                     41.22
Date:                    Wed, 11 Dec 2019   Prob (F-statistic):           1.83e-15
Time:                            05:29:16   Log-Likelihood:                -1272.0
No. Observations:                     183   AIC:                             2550.
Df Residuals:                         180   BIC:                             2560.
Df Model:                               2                                         
Covariance Type:                nonrobust                                         
============================================================================================
                               coef    std err          t      P>|t|      [0.025      0.975]
--------------------------------------------------------------------------------------------
const                        3.3440    626.274      0.005      0.996   -1232.440    1239.128
expected_recovery_amount     0.6430      0.655      0.981      0.328      -0.650       1.936
indicator_1000             277.6344     74.043      3.750      0.000     131.530     423.739
==============================================================================
Omnibus:                       65.977   Durbin-Watson:                   1.906
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              186.537
Skew:                           1.510   Prob(JB):                     3.12e-41
Kurtosis:                       6.917   Cond. No.                     3.37e+04
==============================================================================
'''

# Redefine era_950_1050 so the indicator variable is included
era_950_1050 = df.loc[(df['expected_recovery_amount'] < 1050) &
                      (df['expected_recovery_amount'] >= 950)]

# Define X and y
X = era_950_1050[['expected_recovery_amount', 'indicator_1000']]
y = era_950_1050['actual_recovery_amount']
X = sm.add_constant(X)

# Build linear regression model
model = sm.OLS(y, X).fit()

# Print the model summary
model.summary()

'''
OLS Regression Results
   Dep. Variable: actual_recovery_amount	         R-squared:	0.283
           Model:	OLS	                        Adj. R-squared:	0.269
          Method:	Least Squares	               F-statistic:	18.99
            Date:	Wed, 11 Dec 2019	    Prob (F-statistic):	1.12e-07
            Time:	05:29:16	                Log-Likelihood:	-692.92
No. Observations:	99	                                   AIC:	1392.
    Df Residuals:	96	                                   BIC:	1400.
        Df Model:	2	
 Covariance Type: nonrobust	
                            coef	      std         err	  t	      P>|t|    [0.025	0.975]
                    const	-279.5243	1840.707	-0.152	0.880	-3933.298	3374.250
  expected_recovery_amount	0.9189	    1.886	     0.487	0.627	-2.825	    4.663
            indicator_1000	286.5337	111.352	     2.573	0.012	65.502	    507.566
             
      Omnibus:	39.302	       Durbin-Watson:	1.955
Prob(Omnibus):	0.000	    Jarque-Bera (JB):	82.258
         Skew:	1.564	            Prob(JB):	1.37e-18
     Kurtosis:	6.186	           Cond. No.:	6.81e+04
'''
