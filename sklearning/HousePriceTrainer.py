from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split as splitter
import pandas as pd

#split .6 has the best results, from my outputs


housing = fetch_california_housing()
# for i in housing:
#     print(i)
#     if(i=='data'):
#         for j in housing.__getitem__(i):
#             print(j)
#     else:
#         print(housing.__getitem__(i))

# print(housing)

x1=housing.__getitem__('data')
# print(x1)
y=housing.__getitem__('target')
# print(y)

x=x1[:,[0]]
# print(x)

j=0
temp = []
for i in range(21):
    if(i>18):
        break

    X_train, X_test, Y_train, Y_test = splitter(x, y, test_size=(.05*(i+1)), random_state=42)

    model=LinearRegression()
    model.fit(X_train, Y_train)

    j=0
    correct=0
    incorrect=0
    for eachTest in X_test:
        sample = eachTest.reshape(1, -1)
        prediction = model.predict(sample)[0]
        
        predicted = 1 if prediction >= 0.5 else 0
        actual = Y_test[j]


        if predicted==actual:
            correct=correct+1
        else:
            incorrect=incorrect+1
        
        j = j+1
    
    temp.append({
        "Split": i,
        "Correct": correct,
        "Incorrect": incorrect,
        "Accuracy": (correct/(correct+incorrect)),
        "test_size":(.05*(i+1))
        })
df = pd.DataFrame(temp)
print(df)
max_accuracy = df['Accuracy'].max()
spliter_index = df['Accuracy'].idxmax()
print(f"Maximum Accuracy is {max_accuracy} for spliter at index {spliter_index}")



    

