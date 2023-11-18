#............................................................
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np

# For Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# For Statistics
from scipy import stats
import statsmodels.formula.api as sm
import statsmodels.api as s
from sklearn.linear_model import LinearRegression
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
#.................................................................

# Importing dataset
df = pd.read_csv('C:\\Users\\Admin\\Projects\\Hypothesis_testing\\Admission_Predict.csv')
df.head()
print(df.info())

#I have observed some spaces at column names...
df.columns = df.columns.str.replace(' ', '')
print(df.columns)

fig,axes = plt.subplots(1,2,figsize=(30,6))
fig.suptitle('Distribution of TOEFL Score using box plot')
axes[0].set_title('Histogram')
axes[1].set_title('Quartile information')
sns.boxplot(ax=axes[1],data=df, x=df['TOEFLScore'],y=df['ChanceofAdmit'])
sns.distplot(ax=axes[0],a=df['TOEFLScore'],color='olive',kde=True,rug=True,norm_hist=True)
plt.show()

sns.distplot(a=df['GREScore'],color='olive',kde=True,rug=True,norm_hist=True)
plt.title("Distribution of GRE Score using box plot")
plt.xlabel(" Score")
plt.show()

sns.pairplot(data=df.iloc[:,1:])
plt.show()

### Hypotheseis
 #* Null hypothesis------------H0 = $\mu GRE = \mu TOEFL = \mu CGPA = \mu SOP = \mu LOR = \mu Research = \mu Rating$ ----Not impact on admission
 #* Alternate hypothesis-----Ha = $\mu GRE = \mu TOEFL = \mu CGPA = \mu SOP = \mu LOR = \mu Research = \mu Rating$   ----Impact on admission

reg     = ols(formula='ChanceofAdmit ~ GREScore',data=df)
regfit  = reg.fit()
print(regfit.summary())

print(anova_lm(regfit))

#**SST = SSR + SSE**
#* SST(Total sum of square)      = 6.527 + 3.41 
#* SSR(Regression sum of square) = 6.527(1st independent variable) 
#* SSE(Error sum of square)      = 3.412 error 

#**R square value 0.657**

# Adding more variable
reg1 = ols(formula="ChanceofAdmit ~ TOEFLScore + GREScore + CGPA +Research +SOP+LOR + UniversityRating ",data=df)
Fit1 = reg1.fit()
print(Fit1.summary())

print(anova_lm(Fit1))

#Interpretation
#Observation : After adding a variable into model decrease our SSE while SSR increases
#R-Square value also increase 0.822
#P-value 8.21e-180 is less than 0.05 alpha value hence we reject null hypothesis.
#TOEFL,GRE and CGPA plays main role in selection as they have high sum squared value

# Selecting top 3 variable
reg2 = ols(formula="ChanceofAdmit ~ TOEFLScore + CGPA + GREScore ",data=df)
Fit2 = reg2.fit()
print(Fit2.summary())
print(anova_lm(Fit2,typ=1))

#####  TOEFL,GRE and CGPA plays main role in selection as they have high sum squared value

#* E(y) = $\beta 0 +\beta 1* x1 + \beta 2* x2 \beta 3*x3$
#* Here $\beta0$ is intercept $\beta 1$ is variable TOEFL while $\beta 2$ is 2nd variable for CGPA & $\beta 3$is for GRE.
#* If $\beta 1 $is positive it means that TOEFL plays more role in admission followed by CGPA and GRE.

influence     = Fit2.get_influence()
resid_student = influence.resid_studentized_external
res           = Fit2.resid
probplot      = s.ProbPlot(res,stats.norm,fit =True)
fig           = probplot.qqplot(line='45')
plt.title('qqplot - Residuals of OLS fit')
plt.show()


# Linear Regression through scikit learn 
X = df.iloc[:,1:8].values
X = np.array(X)
df['chance'] = [0 if df['ChanceofAdmit'][i]<0.65 else 1 for i in range(len(df['ChanceofAdmit']))]
y = df['chance']
y = np.array(y)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify =y,random_state=21)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)

def Counter(y):
    c0 = 0 # class 0
    c1 = 0 # class 1
    for i in y:
        if i == 0:
            c0 +=1
        else:
            c1 +=1
    return c0,c1

# SMOTE: Synthetic minority oversampling Technique

from imblearn.over_sampling import SMOTE
counter = Counter(y_train)
print('Before ','0 class=',counter[0],'1=class',counter[1])

smote = SMOTE(sampling_strategy="minority", k_neighbors=5, random_state=44)
x_smote_train, y_smote_train = smote.fit_resample(X_train,y_train)


counter = Counter(y_smote_train)
print('After ','0 class=',counter[0],'1=class',counter[1])

print(x_smote_train.shape,y_smote_train.shape)

lr = LinearRegression()
lr.fit(x_smote_train,y_smote_train)
test_result = lr.predict(X_test)
test_result = [round(test_result[i]) for i in range(len(test_result))]
print(accuracy_score(test_result,y_test))
print(confusion_matrix(test_result,y_test))

