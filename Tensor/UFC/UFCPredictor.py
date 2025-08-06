import tensorflow as tf
import numpy as np
import joblib

# List of your 57 input column names
columns = [
    'RedFighter', 'BlueFighter', "TitleBout", "WeightClass", "Gender", "NumberOfRounds", "BlueDraws",
    "BlueLongestWinStreak", "BlueLosses", "BlueTotalRoundsFought", "BlueTotalTitleBouts",
    "BlueWinsByDecisionMajority", "BlueWinsByDecisionSplit", "BlueWinsByDecisionUnanimous",
    "BlueWinsByKO", "BlueWinsBySubmission", "BlueWinsByTKODoctorStoppage", "BlueWins",
    "BlueStance", "BlueWeightLbs", "RedDraws", "RedLongestWinStreak", "RedLosses",
    "RedTotalRoundsFought", "RedTotalTitleBouts", "RedWinsByDecisionMajority",
    "RedWinsByDecisionSplit", "RedWinsByDecisionUnanimous", "RedWinsByKO",
    "RedWinsBySubmission", "RedWinsByTKODoctorStoppage", "RedWins", "RedStance",
    "RedWeightLbs", "LoseStreakDif", "WinStreakDif", "LongestWinStreakDif", "WinDif",
    "LossDif", "TotalRoundDif", "TotalTitleBoutDif", "KODif", "SubDif", "HeightDif",
    "ReachDif", "AgeDif", "SigStrDif", "AvgSubAttDif", "AvgTDDif", "RWStrawweightRank",
    "BWFlyweightRank"
]

def get_user_input():
    values = []
    print("Enter the values for each feature (float expected).")
    for col in columns:
        while True:
            try:
                val = float(input(f"{col}: "))
                values.append(val)
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
    return np.array(values).reshape(1, -1)  # Shape (1, 57)

def main():
    # Load model and scaler (paths must match your upload files)
    model = tf.keras.models.load_model('Tensor/UFC/perfect_ufc_model.h5')
    scaler = joblib.load('Tensor/scaler.save')

    user_input = get_user_input()
    scaled_input = scaler.transform(user_input)

    preds = model.predict(scaled_input)
    predicted_class = int(np.argmax(preds))
    confidence = float(np.max(preds))

    class_name = "Red" if predicted_class == 1 else "Blue"
    print(f"\nPrediction: {class_name} wins with confidence {confidence:.2f}")

if __name__ == "__main__":
    main()
