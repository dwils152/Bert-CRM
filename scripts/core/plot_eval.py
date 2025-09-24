import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay

# Load data from CSV (ensure the CSV file is in the same directory as this script)
data = pd.read_csv('predictions_labels.csv')  # file should have columns "probs" and "labels"

# Extract the probability predictions and true labels
y_probs = data['probs'].values
y_true = data['labels'].values

# Calculate ROC curve and ROC area
fpr, tpr, _ = roc_curve(y_true, y_probs)
roc_auc = auc(fpr, tpr)

# Calculate Precision-Recall curve and Average Precision (AUPRC)
precision, recall, _ = precision_recall_curve(y_true, y_probs)
auprc = average_precision_score(y_true, y_probs)

# Compute predicted labels using a threshold of 0.5
y_pred = (y_probs >= 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred)

# Create subplots for the ROC curve, PR curve, and confusion matrix
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Plot ROC curve
axs[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
axs[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axs[0].set_xlim([0.0, 1.0])
axs[0].set_ylim([0.0, 1.05])
axs[0].set_xlabel('False Positive Rate')
axs[0].set_ylabel('True Positive Rate')
axs[0].set_title('Receiver Operating Characteristic')
axs[0].legend(loc="lower right")

# Plot Precision-Recall curve
axs[1].plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {auprc:.2f})')
axs[1].set_xlim([0.0, 1.0])
axs[1].set_ylim([0.0, 1.05])
axs[1].set_xlabel('Recall')
axs[1].set_ylabel('Precision')
axs[1].set_title('Precision-Recall Curve')
axs[1].legend(loc="lower left")

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot(ax=axs[2], cmap=plt.cm.Blues)
axs[2].set_title('Confusion Matrix')

plt.tight_layout()
plt.savefig('plots.png', dpi=600)
