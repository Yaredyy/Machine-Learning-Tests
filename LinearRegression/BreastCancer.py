from sklearn.linear_model import LinearRegression as LR
import pandas as pd
import random

ds = pd.read_csv("LinearRegression/Breast_cancer_dataset.csv")

X = ds[["radius_mean", "texture_mean", "symmetry_worst", "fractal_dimension_worst"]].values

Y=ds.diagnosis.replace({'M': 0, 'B': 1})

model=LR()
model.fit(X,Y)

for i in range(100):
    j = random.randint(0, len(X) - 1)
    x_sample = X[j].reshape(1, -1)
    prediction = model.predict(x_sample)[0]

    predicted_class = 1 if prediction >= 0.5 else 0
    confidence = max(min(abs(prediction - 0.5) * 2, 1), 0)  # Clamp between 0 and 1
    actual = Y[j]

    print(f"Predicted: {predicted_class} (Confidence: {confidence:.2f}), Actual: {actual}")

    