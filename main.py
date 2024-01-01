import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler #(standardize data to common range)
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import pickle


df=pd.read_csv("diabetes.csv")

df.head()

df["Outcome"].value_counts()

df.shape

df.info()

df.describe()

df.groupby("Outcome").mean()
# this info is very important for us
# you can see person having more glucose level,insulin,age is diabetic

df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]] = df[["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]].replace(0, np.NaN)
df.isnull()

# Replacing NaN with mean values
df["Glucose"].fillna(df["Glucose"].mean(), inplace = True)
df["BloodPressure"].fillna(df["BloodPressure"].mean(), inplace = True)
df["SkinThickness"].fillna(df["SkinThickness"].mean(), inplace = True)
df["Insulin"].fillna(df["Insulin"].mean(), inplace = True)
df["BMI"].fillna(df["BMI"].mean(), inplace = True)

# all data except labels
X= df.drop(columns="Outcome",axis=1)
y=df["Outcome"]
print(X)



#DATA STANDARDIZATION


# for better prediction we will standardize them
scaler=StandardScaler() 
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)

X=standardized_data
y=df["Outcome"]

print(X)
print(y)



# TRAIN TEST SPILIT

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,stratify=y, random_state=2)
# stratify on "Outcome" column to maintain similar proportion of diabetic or non-diabetic
# random-state is like an index number for partiular splitting of data

print(X.shape,X_train.shape,X_test.shape)

#MODEL TRAINING
classifier=svm.SVC(kernel='linear')

#TRAINING THE SVM CLASSIFIER
classifier.fit(X_train,y_train)

#EVALUATION OF MODEL

# Dự đoán kết quả trên tập huấn luyện
X_train_prediction = classifier.predict(X_train)  # Thay classifier và X_train bằng tên mô hình và dữ liệu thực tế của bạn
training_data_accuracy_score = accuracy_score(y_train, X_train_prediction)

# Tạo biểu đồ tròn
labels = ['Correct Predictions', 'Incorrect Predictions']
sizes = [training_data_accuracy_score, 1 - training_data_accuracy_score]
colors = ['green', 'red']
explode = (0.1, 0)  # Đánh bùng phần Correct Predictions

plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Đảm bảo biểu đồ được vẽ dưới dạng hình tròn
plt.title('Training Data Accuracy')
plt.show()


# Dự đoán kết quả trên tập huấn luyện
X_test_prediction = classifier.predict(X_test)  # Thay classifier và X_train bằng tên mô hình và dữ liệu thực tế của bạn
testing_data_accuracy_score = accuracy_score(y_test, X_test_prediction)

# Tạo biểu đồ tròn
labels = ['Correct Predictions', 'Incorrect Predictions']
sizes = [testing_data_accuracy_score, 1 - testing_data_accuracy_score]
colors = ['blue', 'red']
explode = (0.1, 0)  # Đánh bùng phần Correct Predictions

plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')  # Đảm bảo biểu đồ được vẽ dưới dạng hình tròn
plt.title('Training Data Accuracy')
plt.show()

#SAVING THE TRAINED MODEL

filename="diabetesmodel.sav"
pickle.dump(classifier,open(filename,'wb'))

# loading the saved model
loaded_model=pickle.load(open("diabetesmodel.sav",'rb'))

#MAKING A PREDICTIVE SYSTEM

input_data=(1,85,66,29,0,26.6,0.351,31)

# change input_data to numpy array
inp_data_as_numpy_arr=np.asarray(input_data)
inp_data_reshape=inp_data_as_numpy_arr.reshape(1,-1)

# standardize the input data
std_data=scaler.transform(inp_data_reshape)

# prediction
prediction = loaded_model.predict(std_data)
print(f"Predicted Value: {prediction}")
if(prediction[0]==0):
    print("This person is Non-Diabetic")
else:
    print("This person is Diabetic")
