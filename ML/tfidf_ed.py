import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import euclidean_distances

dataset_name_list = ["datasets/Textual/AB/","datasets/Textual/AG/","datasets/Textual/DA/",
                     "datasets/Textual/DS/", "datasets/Textual/WA/","datasets/Dirty/DA/",
                     "datasets/Dirty/DS/", "datasets/Dirty/WA/"]

for dataset_name in dataset_name_list:
    for dataset_index in range(10):
        print(dataset_name)
        print(dataset_index)
        train = pd.read_csv('../'+dataset_name+'train_'+str(dataset_index)+'.csv')
        valid = pd.read_csv('../'+dataset_name+'valid_'+str(dataset_index)+'.csv')
        test = pd.read_csv('../'+dataset_name+'test_'+str(dataset_index)+'.csv')

        vectorizer = TfidfVectorizer()
        vectorizer.fit(pd.concat([train['left_'],train['right_']]))
        m1 = vectorizer.transform(valid['left_'])
        m2 = vectorizer.transform(valid['right_'])

        len = m1.shape[0]

        scores = []

        for i in range(len):
            score = euclidean_distances(m1[i:i+1], m2[i:i+1])
            scores.append(score[0][0])

        #Evaluation Results
        predict = np.array(scores)
        label = np.array(valid['label'])

        threshold = []
        fscore = []
        for i in np.arange(0,2,0.1):
            threshold.append(i)
            fscore.append(precision_recall_fscore_support(label, np.array(predict < i), average='binary')[2])


        m1 = vectorizer.transform(test['left_'])
        m2 = vectorizer.transform(test['right_'])

        len = m1.shape[0]

        scores = []

        for i in range(len):
            score = euclidean_distances(m1[i:i+1], m2[i:i+1])
            scores.append(score[0][0])

        predict = np.array(scores)
        label = np.array(test['label'])

        # Calculate precision, recall, F-score
        precision, recall, fscore, _ = precision_recall_fscore_support(
            label,
            np.array(predict < threshold[fscore.index(max(fscore))]),
            average='binary'
        )

        # Format values to two decimal places and scale to percentage
        precision = round(precision * 100, 2)
        recall = round(recall * 100, 2)
        fscore = round(fscore * 100, 2)

        # Write the formatted results to the file
        with open('parameter_tuning_tfidf_ed.txt', 'a') as f:
            f.write(f"Precision: {precision} Recall: {recall} F-score: {fscore}\n")