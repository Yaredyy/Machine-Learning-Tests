import pandas as pd
from sklearn.model_selection import train_test_split as split
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#Maximum Accuracy is 0.6614597360624832 for spliter at index 11 of split .6

# Load the dataset
df = pd.read_csv('Tensor/ufc-master.csv')

# Convert Winner column to binary
df['Winner'] = df['Winner'].apply(lambda x: 1 if x == 'Red' else 0)

Y=df['Winner']

# Drop non-relevant columns
X = df.drop(columns=['Winner', 'RedFighter', 'BlueFighter', 'Date', 'Location', 'Country', 'Finish', 'FinishDetails'])

X = X.apply(pd.to_numeric, errors='coerce').astype('float32')
X.fillna(0, inplace=True)

pd.set_option('display.max_columns', None)
# print(df)
# print(Y)
# print("--------------------------------------------------------")
# print(X.columns)
# print(X.head(1))
print("--------------------------------------------------------")
# print(df.columns)
# print(df.head(1))

# Normalize between 0 and 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


temp = []
for j in range(21):
    if(j>18):
        break

    X_train, X_test, Y_train, Y_test = split(X_scaled, Y, test_size=(.05*(j+1)), random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(110, activation='relu'),
        tf.keras.layers.Dense(2),
    ])

    model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

    model.fit(X_train, Y_train, epochs=10)

    test_loss, test_acc = model.evaluate(X_test,  Y_test, verbose=2)
    predictions = model.predict(X_test)

    probs = tf.nn.softmax(predictions).numpy()
    predicted_classes = np.argmax(probs, axis=1)

    # Reset Y_test index to align with predictions
    Y_test = Y_test.reset_index(drop=True)

    correct=0
    incorrect=0

    for i in range(len(predicted_classes)):
        actual = Y_test[i]
        predicted = predicted_classes[i]
        confidence = probs[i][predicted]

        if predicted == actual:
            result = "✅ Correct"
            correct = correct + 1
        else:
            result = "❌ Wrong"
            incorrect = incorrect + 1
        print(f"Fight {i+1}: Predicted = {'Red' if predicted == 1 else 'Blue'} "
            f"(Conf: {confidence:.2f}) | Actual = {'Red' if actual == 1 else 'Blue'} → {result}")
    temp.append({
        "Split": j+1,
        "Correct": correct,
        "Incorrect": incorrect,
        "Accuracy": (correct/(correct+incorrect)),
        "test_size":(.05*(j+1))
        })
        
df = pd.DataFrame(temp)
print(df)
max_accuracy = df['Accuracy'].max()
spliter_index = df['Accuracy'].idxmax()
print(f"Maximum Accuracy is {max_accuracy} for spliter at index {spliter_index} of split {(.05*(j+1))}")