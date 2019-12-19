#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_excel(r'D:\UMD\Summer Term\Internship\Heart_Disease_Dataset.xlsx')
print("=====================================Head=====================================")
print(df.head())
print(df.describe())


# In[2]:


print(df.info())


# In[3]:


print(df.isnull().sum())


# In[4]:


df.dropna(axis=0,inplace=True)


# In[5]:


df.hist(bins=20, figsize=(20,15))


# In[6]:


df['TenYearCHD'].value_counts(dropna=False)


# In[7]:


plot = df['TenYearCHD'].value_counts().plot('pie', figsize=(5,3),  autopct='%1.1f%%')
plt.title('10 year Chronic disease rate', fontsize=12)


# In[8]:


print(df.shape)
print(list(df.columns))


# In[9]:


plt.figure(figsize=(5,5))
sns.countplot('male', hue='TenYearCHD', data=df)
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["No disease", "Have disease"])
plt.ylabel('Frequency')


# In[10]:


plt.figure(figsize=(15,3))
sns.countplot('age', hue='TenYearCHD', data=df)
plt.title('Heart Disease Frequency for Age')
plt.xlabel('Age')
plt.xticks(rotation=0)
plt.legend(["No disease", "Have disease"])
plt.ylabel('Frequency')


# In[11]:


#1 = Some High School; 2 = High School or GED; 3 = Some College or Vocational School; 4 = college
plt.figure(figsize=(5,3))
sns.countplot('education', hue='TenYearCHD', data=df)
#plt.title('Heart Disease Frequency for education')
plt.xlabel('Education (1 = Some High School; 2 = High School or GED; 3 = Some College or Vocational School; 4 = college)')
plt.xticks(rotation=0)
plt.legend(["No disease", "Have disease"])
plt.ylabel('Frequency')


# In[12]:


plt.figure(figsize=(5,5))
sns.countplot('currentSmoker', hue='TenYearCHD', data=df)
plt.xlabel('Current smoker (0 = Not, 1 = Smoker)')
plt.xticks(rotation=0)
plt.legend(["No disease", "Have disease"])
plt.ylabel('Frequency')


# In[13]:


plt.figure(figsize=(20,5))
sns.countplot('cigsPerDay', hue='TenYearCHD', data=df)
plt.title('Heart Disease Frequency for Cigars per day')
plt.xlabel('Cigars per day')
plt.xticks(rotation=0)
plt.legend(["No disease", "Have disease"])
plt.ylabel('Frequency')


# In[14]:


plt.figure(figsize=(5,3))
sns.countplot('BPMeds', hue='TenYearCHD', data=df)
#plt.title('Heart Disease Frequency for patient was on BP medication')
plt.xlabel('BP Medication before')
plt.xticks(rotation=0)
plt.legend(["No disease", "Have disease"])
plt.ylabel('Frequency')


# In[15]:


plt.figure(figsize=(5,3))
sns.countplot('prevalentStroke', hue='TenYearCHD', data=df)
#plt.title('Heart Disease Frequency Vs Stroke before')
plt.xlabel('Prevalent Stroke')
plt.xticks(rotation=0)
plt.legend(["No disease", "Have disease"])
plt.ylabel('Frequency')


# In[16]:


plt.figure(figsize=(5,3))
sns.countplot('prevalentHyp', hue='TenYearCHD', data=df)
#plt.title('Heart Disease Frequency patient had stroke before')
plt.xlabel('Prevalent Hypertension')
plt.xticks(rotation=0)
plt.legend(["No disease", "Have disease"])
plt.ylabel('Frequency')


# In[17]:


plt.figure(figsize=(5,3))
sns.countplot('diabetes', hue='TenYearCHD', data=df)
plt.xlabel('Diabetes')
plt.xticks(rotation=0)
plt.legend(["No disease", "Have disease"])
plt.ylabel('Frequency')


# In[18]:


df.plot.scatter(x='totChol', y='TenYearCHD')


# In[19]:


df.plot.scatter(x='sysBP', y='TenYearCHD')


# In[20]:


plt.scatter(x=df.age[df.TenYearCHD==1], y=df.totChol[(df.TenYearCHD==1)], c="red")
plt.scatter(x=df.age[df.TenYearCHD==0], y=df.totChol[(df.TenYearCHD==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Total Cholestral")
plt.show()


# In[21]:


plt.scatter(x=df.age[df.TenYearCHD==1], y=df.sysBP[(df.TenYearCHD==1)], c="red")
plt.scatter(x=df.age[df.TenYearCHD==0], y=df.sysBP[(df.TenYearCHD==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("systolic blood pressure ")
plt.show()


# In[22]:


plt.scatter(x=df.age[df.TenYearCHD==1], y=df.diaBP[(df.TenYearCHD==1)], c="red")
plt.scatter(x=df.age[df.TenYearCHD==0], y=df.diaBP[(df.TenYearCHD==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Diastolic blood pressure")
plt.show()


# In[23]:


plt.scatter(x=df.age[df.TenYearCHD==1], y=df.BMI[(df.TenYearCHD==1)], c="red")
plt.scatter(x=df.age[df.TenYearCHD==0], y=df.BMI[(df.TenYearCHD==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("BMI")
plt.show()


# In[24]:


plt.scatter(x=df.age[df.TenYearCHD==1], y=df.heartRate[(df.TenYearCHD==1)], c="red")
plt.scatter(x=df.age[df.TenYearCHD==0], y=df.heartRate[(df.TenYearCHD==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Heart rate")
plt.show()


# In[25]:


plt.scatter(x=df.age[df.TenYearCHD==1], y=df.glucose[(df.TenYearCHD==1)], c="red")
plt.scatter(x=df.age[df.TenYearCHD==0], y=df.glucose[(df.TenYearCHD==0)])
plt.legend(["Disease", "Not Disease"])
plt.xlabel("Age")
plt.ylabel("Heart rate")
plt.show()


# In[26]:


sns.pairplot(data=df)


# In[27]:


df.corr()
sns.heatmap(df.corr())


# In[28]:


#Highly Correlated Variables:
#prevalentHyp & sysBP,currentSmoker & cigsPerDay, diabetes & glucose, sysBP & prevalentHyp & diaBP, diaBP
df.drop('sysBP', axis=1, inplace=True)
df.drop('diabetes', axis=1, inplace=True)
df.drop('currentSmoker', axis=1, inplace=True)
df.drop('glucose', axis=1, inplace=True)
df.drop('diaBP', axis=1, inplace=True)


# In[29]:


plt.figure(figsize=(20,5))
sns.countplot('cigsPerDay', hue='TenYearCHD', data=df)
plt.title('Heart Disease Frequency for cigsPerDay')
plt.xlabel('cigsPerDay')
plt.xticks(rotation=0)
plt.legend(["No disease", "Have disease"])
plt.ylabel('Frequency')


# In[30]:


sns.heatmap(df.corr())
df.corr()


# In[31]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df.info()


# In[32]:


#split dataset in features and target variable
feature_cols = ['male', 'education', 'cigsPerDay', 'age','BPMeds','prevalentStroke','prevalentHyp','totChol','BMI','heartRate']
X = df[feature_cols] # Features
y = df.TenYearCHD # Target variable


# In[33]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[34]:


# Logistic Regression Model
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train,y_train)

y_pred=logreg.predict(X_test)


# In[35]:


# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[36]:


import numpy as np
class_names=[0,1] 
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[37]:


#To retrieve the intercept:
print(logreg.intercept_)
#For retrieving the slope:
print(logreg.coef_)


# In[38]:


log_coeff_df = pd.DataFrame(logreg.coef_[0], X.columns, columns=['Coefficient'])
log_coeff_df


# In[39]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[40]:


y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.plot([0, 1], [0, 1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()


# In[41]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[42]:


# Create Decision Tree classifer object
from sklearn.tree import DecisionTreeClassifier 
clf_tree = DecisionTreeClassifier()

clf_tree = clf_tree.fit(X_train,y_train)
y_pred = clf_tree.predict(X_test)


# In[43]:


# Model Accuracy
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[44]:


print(clf_tree.feature_importances_)


# In[45]:


import pandas as pd
feature_imp_tree = pd.Series(clf_tree.feature_importances_,index=feature_cols).sort_values(ascending=False)
feature_imp_tree


# In[48]:


# Visualizing Predictors
plt.figure(figsize=(10,5))
sns.barplot(x=feature_imp_tree, y=feature_imp_tree.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[49]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[50]:


#ROC Curve
y_pred_proba_tree = clf_tree.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_tree)
auc = metrics.roc_auc_score(y_test, y_pred_proba_tree)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.plot([0, 1], [0, 1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()


# In[51]:


# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)


# In[52]:


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

clf_rf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets
clf_rf.fit(X_train,y_train)

y_pred=clf_rf.predict(X_test)


# In[53]:


import pandas as pd
feature_imp_rf = pd.Series(clf_rf.feature_importances_,index=feature_cols).sort_values(ascending=False)
feature_imp_rf


# In[54]:


#Visualizing important features
import pandas as pd
feature_imp_rf1 = pd.Series(clf_rf.feature_importances_,index=feature_cols).sort_values(ascending=False)
feature_imp_rf1
# Creating a bar plot
sns.barplot(x=feature_imp_rf1, y=feature_imp_rf1.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()


# In[55]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[56]:


y_pred_proba_rf = clf_rf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba_rf)
auc = metrics.roc_auc_score(y_test, y_pred_proba_rf)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.plot([0, 1], [0, 1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=4)
plt.show()


# In[ ]:




