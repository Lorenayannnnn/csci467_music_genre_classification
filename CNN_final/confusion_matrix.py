from sklearn.metrics import confusion_matrix
from sklearn.metrics import hamming_loss

def report_classification_accuracy(pred_labels, true_labels):

    index_2_genre = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
    
    cm = confusion_matrix(pred_labels, true_labels)

    import numpy as np
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    acc_results = cm.diagonal()
    for idx, class_name in enumerate(index_2_genre):
        print(class_name, acc_results[idx])
    
    print('hamming loss:', hamming_loss(true_labels, pred_labels) )

