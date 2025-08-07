import pandas as pd
from sklearn.model_selection import train_test_split as split
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib


baseFolder=input("Input Base folder: ")
baseFolder="Tensor/PreDiabetes/Models/"+baseFolder

# Load the dataset
df = pd.read_csv('Tensor/PreDiabetes/Prediabetes.csv')



Y=df['diabetes']


#Drop non-relevant columns
X = df.drop(columns=['diabetes'])

X = X.apply(pd.to_numeric, errors='coerce').astype('float32')

# Normalize between 0 and 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

size = X_scaled.shape[1]

temp = []
best_accuracy = 0
best_model = None

for j in range(19):  # 0 to 18 splits
    test_size = 0.05 * (j + 1)
    X_train, X_test, Y_train, Y_test = split(X_scaled, Y, test_size=test_size, random_state=42)


    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(size,)),
        tf.keras.layers.Dense(size, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(int(np.floor(size/2)), activation='relu'),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(int(np.floor(size/4)), activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    model.fit(X_train, Y_train, epochs=10, verbose=0)

    predictions = model.predict(X_test)
    probs = predictions
    predicted_classes = np.argmax(probs, axis=1)

    Y_test = Y_test.reset_index(drop=True)

    correct = np.sum(predicted_classes == Y_test)
    incorrect = len(predicted_classes) - correct

    for i in range(len(predicted_classes)):
        actual = Y_test.iloc[i]
        predicted = predicted_classes[i]
        confidence = probs[i][predicted]

        result = "✅ Correct" if predicted == actual else "❌ Wrong"
        print(f"Fight {i+1}: Predicted = {'Diabetes' if predicted == 1 else 'No Diabetes'} "
              f"(Conf: {confidence:.2f}) | Actual = {'Diabetes' if actual == 1 else 'No Diabetes'} → {result}")

    accuracy = correct / (correct + incorrect)
    print(f"Split:{j+1} Correct: {correct} | Incorrect: {incorrect} | Accuracy: {(correct/(correct+incorrect)):.4f}% | Split:{test_size}")
    temp.append({
        "Split": j + 1,
        "Correct": correct,
        "Incorrect": incorrect,
        "Accuracy": accuracy,
        "test_size": test_size
    })
    
    if accuracy == 1.0:
        perfect_filename = baseFolder+f"/Perfects/perfect_ufc_model_{j + 1}.h5"
        model.save(perfect_filename)
        print(f"Saved perfect model for split {j + 1} as {perfect_filename}")

    if accuracy >= best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_split = j + 1
        

out = pd.DataFrame(temp)
print(out)

spliter_index = out['Accuracy'].idxmax()
print(f"Maximum Accuracy is {out['Accuracy'][spliter_index]} for spliter at index {spliter_index} of split {(.05*(spliter_index+1))}")

if best_model is not None:
    best_model.save(baseFolder+"/Best_ufc_model.h5")
    print("Saved best model as Best_ufc_model.h5")

out.to_csv(baseFolder+"/SessionSummary.csv")
joblib.dump(scaler, baseFolder + "/scaler.save")