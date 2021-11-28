import numpy as np
import sklearn.metrics as sk
import matplotlib.pyplot as plt

# Generates ROC curve and finds other metrics 
# ROC curve plots TPR (true positive rate [recall]) against FPR (flase positive rate) as the threshold increases
# The closer to the top left corner of the graph the line is the better. A linear result of x = y means the model is a coin flip.
# ground_truth is a boolean arraylike containing the actual validation data
# detections is a float arraylike containing the detection scores that will be thresholded
def getROC(ground_truth, detections):    

    # Generate array of threshold values (start of interval, end of interval, number of thresholds)
    thresholds = np.linspace(0.1, 140, 1000)


    # Generate lists to append findings
    cmatricies = []
    tp_rates = []
    fp_rates = []
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    # Iterate over every threshold 
    for threshold in thresholds:

        # get prediction
        if not threshold:
            continue
        
        # Make predictions using threshold
        predictions = detections < threshold
        
        # calculate confusion matrix
        cm = sk.confusion_matrix(np.invert(ground_truth), predictions)
        cmatricies.append(cm)

        # get True Nagatives, False Positives, False Negatives, and True Positives 
        # to calculate True Positive Rate and False Positve Rate
        (TN, FP, FN, TP) = cm.ravel()

        # Sensitivity, recall, or true positive rate
        TPR = TP / (TP + FN)
        tp_rates.append(TPR)

        # False positive rate
        FPR = FP / (FP + TN)
        fp_rates.append(FPR)

        # Accuracy: What percent of the predictions are correct
        accuracy = (TP + TN)/(TP + FP + FN + TN)

        # Precision: What percent of the positive predictions are correct
        #   Can you trust your model when it says TRUE
        # Avoid 0/0 by setting precision to 1 if there are no positive predictions
        if (not TP and not FP):
            precision = 1
        else:
            precision = TP/(TP + FP)
        
        # Recall/Sensitivity: What percent of the positives were predicted to be positive
        #   Can you trust your model when it says FALSE
        recall = TP/(TP + FN)

        # 7) F-1 Score: The balance between precision and recall
        f1 = 2*(precision*recall)/(precision + recall)

        # A
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
    plt.figure(figsize = [10,8])
    plt.figure(figsize = [10,8])
    plt.plot(fp_rates, tp_rates, label='ROC curve', color='b')
    plt.plot([0, 1], [0, 1], label='Random Classifier (AUC = 0.5)', linestyle='--', lw=2, color='r')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.title('ROC Curve')
    plt.show()

    # Display 
    plt.figure(figsize = [20,10])
    plt.subplot(141);plt.plot(thresholds, accuracies);plt.title('Accuracy')
    plt.subplot(142);plt.plot(thresholds, precisions);plt.title('Precision')
    plt.subplot(143);plt.plot(thresholds, recalls);plt.title('Recall')
    plt.subplot(144);plt.plot(thresholds, f1s);plt.title('F1')
    plt.show()

    return np.array(thresholds), np.array(accuracies), np.array(precisions), np.array(recalls), np.array(f1s), np.array(cmatricies)