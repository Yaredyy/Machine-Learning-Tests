import pandas as pd
from sklearn.model_selection import train_test_split as split
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Modle one with:
#model = tf.keras.Sequential([
#        tf.keras.layers.Dense(110, activation='relu'),
#        tf.keras.layers.Dense(2),
#    ])
# Had maximum Accuracy is 0.6614597360624832 for spliter at index 11 of split .6

#New layers with old data had:
#Maximum Accuracy is 0.6541879881497441 for spliter at index 11 of split .6


#with cleaner data, new model has:
#


baseFolder=input("Input Base folder: ")
baseFolder="Tensor/Models/"+baseFolder

# Load the dataset
df = pd.read_csv('Tensor/ufc-master.csv')

# Convert Winner column to binary
df['Winner'] = df['Winner'].apply(lambda x: 1 if x == 'Red' else 0)

Y=df['Winner']


# Drop non-relevant columns
X = df.drop(columns=['Winner', 'RedFighter', 'BlueFighter', 'Date', 'Location', 'Country', 'Finish', 'FinishDetails'])
X = df.drop(columns=[
    # Raw stats if keeping differences
    'RedAge', 'BlueAge',
    'RedHeightCms', 'BlueHeightCms',
    'RedReachCms', 'BlueReachCms',
    'RedCurrentWinStreak', 'BlueCurrentWinStreak',
    'RedCurrentLoseStreak', 'BlueCurrentLoseStreak',
    'RedAvgSigStrLanded', 'BlueAvgSigStrLanded',
    'RedAvgSigStrPct', 'BlueAvgSigStrPct',
    'RedAvgSubAtt', 'BlueAvgSubAtt',
    'RedAvgTDLanded', 'BlueAvgTDLanded',
    'RedAvgTDPct', 'BlueAvgTDPct',

    # Odds / Expected Value
    'RedOdds', 'BlueOdds',
    'RedExpectedValue', 'BlueExpectedValue',
    'RedDecOdds', 'BlueDecOdds',
    'RKOOdds', 'BKOOdds',
    'RSubOdds', 'BSubOdds',

    # Post-fight outcome info
    'FinishRound', 'FinishRoundTime',
    'TotalFightTimeSecs',

    # Rankings
    'RPFPRank', 'BPFPRank', 'BetterRank',
    'RWFlyweightRank', 'RWFeatherweightRank',
    'RWBantamweightRank', 'RHeavyweightRank',
    'RLightHeavyweightRank', 'RMiddleweightRank',
    'RWelterweightRank', 'RLightweightRank',
    'RFeatherweightRank', 'RBantamweightRank',
    'RFlyweightRank', 'BWFeatherweightRank',
    'BWStrawweightRank', 'BWBantamweightRank',
    'BHeavyweightRank', 'BLightHeavyweightRank',
    'BMiddleweightRank', 'BWelterweightRank',
    'BLightweightRank', 'BFeatherweightRank',
    'BBantamweightRank', 'BFlyweightRank',

    # Ambiguous
    'EmptyArena', 'BMatchWCRank', 'RMatchWCRank'
])

X = X.apply(pd.to_numeric, errors='coerce').astype('float32')
X.fillna(0, inplace=True)

# Normalize between 0 and 1
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# pd.set_option('display.max_columns', None)
# print(df)
# print(Y)
# print("--------------------------------------------------------")
# print(X.columns)
# print(X.head(1))
# print("--------------------------------------------------------")
# print(df.columns)
# print(df.head(1))



temp = []
best_accuracy = 0
best_model = None

for j in range(19):  # 0 to 18 splits
    test_size = 0.05 * (j + 1)
    X_train, X_test, Y_train, Y_test = split(X_scaled, Y, test_size=test_size, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_scaled.shape[1],)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    model.fit(X_train, Y_train, epochs=10, verbose=0)

    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=2)
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
        print(f"Fight {i+1}: Predicted = {'Red' if predicted == 1 else 'Blue'} "
              f"(Conf: {confidence:.2f}) | Actual = {'Red' if actual == 1 else 'Blue'} → {result}")

    accuracy = correct / (correct + incorrect)
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
        

df = pd.DataFrame(temp)
print(df)
max_accuracy = df['Accuracy'].max()
spliter_index = df['Accuracy'].idxmax()
print(f"Maximum Accuracy is {max_accuracy} for spliter at index {spliter_index} of split {(.05*(spliter_index+1))}")

if best_model is not None:
    best_model.save(baseFolder+"/Best_ufc_model.h5")
    print("Saved best model as Best_ufc_model.h5")

df.to_csv(baseFolder+"/SessionSummary.csv")