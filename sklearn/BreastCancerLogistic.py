from sklearn.linear_model import LogisticRegression as LR
import pandas as pd
from sklearn.model_selection import train_test_split as splitter

ds = pd.read_csv("sklearn/Breast_cancer_dataset.csv")

X = ds[["radius_mean", "texture_mean", "symmetry_worst", "fractal_dimension_worst"]].values

Y=ds.diagnosis.replace({'M': 0, 'B': 1})

X_train, X_test, Y_train, Y_test = splitter(X, Y, test_size=0.35, random_state=42)


model=LR()
model.fit(X_train, Y_train)

j=0
correct=0
incorrect=0
for i in X_test:
    x_sample = i.reshape(1, -1)
    prediction = model.predict(x_sample)[0]
    
    predicted = 1 if prediction >= 0.5 else 0
    confidence = max(min(abs(prediction - 0.5) * 2, 1), 0)
    actual = Y_test.iloc[j]

    print(f"{j}: Predicted: {predicted} (Confidence: {confidence:.2f}), Actual: {actual} {"Correct" if predicted==actual else "Incorrect"}")

    if predicted==actual:
        correct=correct+1
    else:
        incorrect=incorrect+1

    j=j+1


print(f"Correct:{correct} ,Incorrect:{incorrect}, Accuracy of Model:{correct/(correct+incorrect)}")
score = model.score(X_test, Y_test)
print(f"score: {score}")


    