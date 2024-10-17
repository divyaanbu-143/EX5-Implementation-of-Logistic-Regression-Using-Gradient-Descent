# EX 5 Implementation of Logistic Regression Using Gradient Descent
# DATE:
## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1. Initialize weights 𝜃 as a zero or random vector of size 𝑛, and set bias b=0.
 2. For logistic regression, the hypothesis ℎ𝜃(𝑥) is given by the sigmoid function: 1 hθ(x)=.............. 1+e −(θ T X+b)
 3. The cost function for logistic regression is the binary cross-entropy:
 4. Gradient Descent to Minimize Cost Function

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Divya A
RegisterNumber:  2305002007
*/
```
 import pandas as pd
 import numpy as np
 data=pd.read_csv("/content/ex45Placement_Data.csv")
 data.head()
 data1=data.copy()
 data1.head()
 data1=data1.drop(['sl_no','salary'],axis=1)
 data1
 from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
 data1["gender"]=le.fit_transform(data1["gender"])
 data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
 data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
 data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
 data1["degree_t"]=le.fit_transform(data1["degree_t"])
 data1["workex"]=le.fit_transform(data1["workex"])
 data1["specialisation"]=le.fit_transform(data1["specialisation"])
 data1["status"]=le.fit_transform(data1["status"])
 data1
 X=data1.iloc[:,: -1]
 Y=data1["status"]
 theta=np.random.randn(X.shape[1])
 y=Y
 def sigmoid(z):
 return 1/(1+np.exp(-z))
 def loss(theta,X,y):
 h=sigmoid(X.dot(theta))
 return -np.sum(y*np.log(h)+(1-y)*log(1-h))
 def gradient_descent(theta,X,y,alpha,num_iterations):
 m=len(y)
 for i in range(num_iterations):
 h=sigmoid(X.dot(theta))
 gradient=X.T.dot(h-y)/m
 theta-=alpha*gradient
 return theta
 theta=gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
 def predict(theta,X):
 h=sigmoid(X.dot(theta))
 y_pred=np.where(h>=0.5,1,0)
 return y_pred
 y_pred=predict(theta,X)
 accuracy=np.mean(y_pred.flatten()==y)
 print("Accuracy:",accuracy)
print("Predicted:\n",y_pred)
 print("Actual:\n",y.values)
 xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
 y_prednew=predict(theta,xnew)
 print("Predicted Result:",y_prednew)

## Output:
![image](https://github.com/user-attachments/assets/b9070025-ae6c-4bf4-bb64-25f96ba03998)
![image](https://github.com/user-attachments/assets/836e5443-20b7-4dbe-96a4-4d219bfa4662)
![image](https://github.com/user-attachments/assets/4c78a674-a89e-4b49-b8fc-1b7a07cfe3b3)


![image](https://github.com/user-attachments/assets/06912682-4ec1-42f0-9fb4-18616a5e03d5)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

