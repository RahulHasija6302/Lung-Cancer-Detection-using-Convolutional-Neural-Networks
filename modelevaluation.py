

//Model Evaluation

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Evaluate the trained model's performance on the testing set
y_pred = model.predict(X_test)
y_pred_binary = np.round(y_pred).astype(int).flatten()

# Calculate metrics such as accuracy, precision, recall, and F1 score to assess the model's effectiveness
accuracy = accuracy_score(y_test, y_pred_binary)
precision = precision_score(y_test, y_pred_binary)
recall = recall_score(y_test, y_pred_binary)
f1 = f1_score(y_test, y_pred_binary)

print("Accuracy: {:.4f}".format(accuracy))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1 Score: {:.4f}".format(f1))

# Visualize the evaluation results using appropriate plots or charts
cm = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(cm, annot=True, cmap="Blues", fmt="g")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
