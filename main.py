import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("student-mat.csv", sep=';')

print("Dataset preview:")
print(df.head())

df['pass_fail'] = (df['G3'] >= 10).astype(int)


X = df[['studytime', 'absences', 'failures', 'G1', 'G2']]
y = df['pass_fail']



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("\nModel Accuracy on Test Set:", round(accuracy*100, 2), "%")

print("\nEnter new student data to predict Pass/Fail:")
studytime = int(input("Study time (1-4): "))
absences = int(input("Number of absences: "))
failures = int(input("Past class failures: "))
G1 = int(input("First period grade (G1): "))
G2 = int(input("Second period grade (G2): "))


new_student = pd.DataFrame([[studytime, absences, failures, G1, G2]],
                           columns=['studytime', 'absences', 'failures', 'G1', 'G2'])


prediction = model.predict(new_student)
probability = model.predict_proba(new_student)

if prediction[0] == 1:
    print("\nPrediction: PASS")
else:
    print("\nPrediction: FAIL")

print("Probability of passing:", round(probability[0][1]*100, 2), "%")

