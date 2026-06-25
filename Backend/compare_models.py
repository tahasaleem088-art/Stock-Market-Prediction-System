models = {
    "Random Forest": {
        "Accuracy": 0.60,
        "F1 Score": 0.65
    },
    "LSTM": {
        "Accuracy": 0.66,
        "F1 Score": 0.79
    },
    "GRU": {
        "Accuracy": 0.66,
        "F1 Score": 0.79
    }
}

print("\n📊 MODEL COMPARISON\n")

for model, metrics in models.items():
    print(f"{model}")
    print(f"Accuracy : {metrics['Accuracy']}")
    print(f"F1 Score : {metrics['F1 Score']}")
    print("-" * 30)

best_model = max(models, key=lambda x: models[x]["F1 Score"])

print(f"\n🏆 Best Performing Model: {best_model}")
import matplotlib.pyplot as plt

model_names = ["RF", "LSTM", "GRU"]
accuracy = [0.60, 0.66, 0.66]
f1 = [0.65, 0.79, 0.79]

x = range(len(model_names))

plt.figure(figsize=(8,5))

plt.plot(x, accuracy, marker='o', label="Accuracy")
plt.plot(x, f1, marker='o', label="F1 Score")

plt.xticks(x, model_names)

plt.xlabel("Models")
plt.ylabel("Score")
plt.title("Model Performance Comparison")

plt.legend()

plt.show()