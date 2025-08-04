from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split as splitter

#.1 split has the best results, from my outputs


ds = pd.read_csv("sklearning\Breast_cancer_dataset.csv")

X = ds[["radius_mean", "texture_mean", "symmetry_worst", "fractal_dimension_worst"]].values

Y=ds.diagnosis.replace({'M': 0, 'B': 1})



for i in range(21):
    if(i>18):
        break

    X_train, X_test, Y_train, Y_test = splitter(X, Y, test_size=(.05*(i+1)), random_state=42)

    model=SVC(probability=True)
    model.fit(X_train, Y_train)

    j=0
    correct=0
    incorrect=0
    for eachTest in X_test:
        sample = eachTest.reshape(1, -1)
        prediction = model.predict(sample)[0]
        
        predicted = 1 if prediction >= 0.5 else 0
        actual = Y_test.iloc[j]


        if predicted==actual:
            correct=correct+1
        else:
            incorrect=incorrect+1
        
        j = j+1
    
    print(f"Split:{i+1} Correct: {correct} | Incorrect: {incorrect} | Accuracy: {(correct/(correct+incorrect)):.3f} | Split:{.05*(i+1)}")