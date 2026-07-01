from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_accuracy(y_true, y_pred):
    """Calculate overall accuracy"""
    return accuracy_score(y_true, y_pred)

def calculate_precision(y_true, y_pred, average='weighted'):
    """Calculate precision score"""
    return precision_score(y_true, y_pred, average=average, zero_division=0)

def calculate_recall(y_true, y_pred, average='weighted'):
    """Calculate recall score"""
    return recall_score(y_true, y_pred, average=average, zero_division=0)

def calculate_f1(y_true, y_pred, average='weighted'):
    """Calculate F1 score"""
    return f1_score(y_true, y_pred, average=average, zero_division=0)

def calculate_all_metrics(y_true, y_pred, average='weighted'):
    """Calculate accuracy, precision, recall, and F1."""
    return {
        "accuracy": calculate_accuracy(y_true, y_pred),
        "precision": calculate_precision(y_true, y_pred, average=average),
        "recall": calculate_recall(y_true, y_pred, average=average),
        "f1": calculate_f1(y_true, y_pred, average=average)
    }
