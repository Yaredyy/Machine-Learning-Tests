from sklearn.linear_model import LogisticRegression as LR
import pandas as pd
from sklearn.model_selection import train_test_split as splitter
import seaborn as sns
import matplotlib.pyplot as plt
#.1 split has the best results, from my outputs

ds = pd.read_csv("sklearning/BreastCancer/Breast_cancer_dataset.csv")
# print(ds.columns)
ds.diagnosis=ds.diagnosis.replace({'M': 0, 'B': 1})

# temp=pd.DataFrame()
# j=0
# for i in ds.columns:
#     if i=='diagnosis':
#         continue
#     temp[f"{i}"]=ds[f"{i}"].values
#     j += 1
#     if (j==15 or i==ds.columns[ds.shape[1]-2]):
#         temp['diagnosis']=ds['diagnosis'].values
#         sns.heatmap(temp.corr())
#         plt.show()
#         j=0
#         temp=pd.DataFrame()
    
    


# print(ds.isnull().sum())
# print(ds.shape)


X = ds[['fractal_dimension_mean','texture_se','smoothness_se','symmetry_se','fractal_dimension_se']].values

Y=ds['diagnosis']

temp = []
for i in range(21):
    if(i>18):
        break

    X_train, X_test, Y_train, Y_test = splitter(X, Y, test_size=(.05*(i+1)), random_state=42)

    model=LR()
    model.fit(X_train, Y_train)

    j=0
    correct=0
    incorrect=0
    for eachTest in X_test:
        sample = eachTest.reshape(1, -1)
        prediction = model.predict(sample)[0]
        actual = Y_test.iloc[j]


        if prediction==actual:
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
    print(f"Split:{i+1} Correct: {correct} | Incorrect: {incorrect} | Accuracy: {(correct/(correct+incorrect)):.3f} | Split:{.05*(i+1)}")

df = pd.DataFrame(temp)
print(df)
max_accuracy = df['Accuracy'].max()
spliter_index = df['Accuracy'].idxmax()
print(f"Maximum Accuracy is {max_accuracy} for spliter at index {spliter_index} of split {(.05*(spliter_index+1))}")

df.to_csv("sklearning\BreastCancer\BreastCancerLogisticResult5.csv")