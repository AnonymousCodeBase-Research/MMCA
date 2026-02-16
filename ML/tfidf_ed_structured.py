import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import euclidean_distances

# Dynamically identify attribute pairs from dataset columns
def infer_attribute_pairs(columns):
    """Infer attribute pairs dynamically based on column names."""
    attributes = []
    left_columns = [col for col in columns if col.startswith("left_")]
    right_columns = [col for col in columns if col.startswith("right_")]

    # Match left and right columns by their suffix (e.g., "title", "price", etc.)
    for left_col in left_columns:
        suffix = left_col.replace("left_", "")
        matching_right_col = f"right_{suffix}"
        if matching_right_col in right_columns and suffix != "id":  # Exclude ID columns
            attributes.append((left_col, matching_right_col))
    return attributes

# Compute average Euclidean distance for all attribute pairs
def compute_aggregated_distance(df, vectorizer, attributes):
    """Compute aggregated average Euclidean distances for all attribute pairs."""
    aggregated_scores = []
    num_attributes = len(attributes)  # Number of attribute pairs

    for index in range(len(df)):
        total_distance = 0
        for left_attr, right_attr in attributes:
            # Get TF-IDF vectors for the current row
            m1 = vectorizer.transform([df.iloc[index][left_attr]])
            m2 = vectorizer.transform([df.iloc[index][right_attr]])
            # Compute Euclidean distance and add to total
            total_distance += euclidean_distances(m1, m2)[0][0]
        # Compute average distance
        avg_distance = total_distance / num_attributes
        aggregated_scores.append(avg_distance)
    return np.array(aggregated_scores)

# Main processing loop
dataset_name_list = ["datasets/Structured/AB/", "datasets/Structured/AG/", "datasets/Structured/DA/",
                     "datasets/Structured/DS/", "datasets/Structured/WA/"]

for dataset_name in dataset_name_list:
    for dataset_index in range(10):
        print(dataset_name)
        print(dataset_index)

        # Load the datasets
        train = pd.read_csv('../' + dataset_name + 'train_' + str(dataset_index) + '.csv')
        valid = pd.read_csv('../' + dataset_name + 'valid_' + str(dataset_index) + '.csv')
        test = pd.read_csv('../' + dataset_name + 'test_' + str(dataset_index) + '.csv')

        # Infer attribute pairs dynamically from dataset columns
        attributes = infer_attribute_pairs(train.columns)
        print(f"Inferred attributes: {attributes}")

        # Replace NaN values with empty strings and ensure all columns are strings
        for df in [train, valid, test]:
            for attr_pair in attributes:
                df[attr_pair[0]] = df[attr_pair[0]].fillna("").astype(str)
                df[attr_pair[1]] = df[attr_pair[1]].fillna("").astype(str)

        # Initialize TF-IDF vectorizer and combine text from all attributes
        vectorizer = TfidfVectorizer()
        combined_train = pd.concat([train[attr[0]] for attr in attributes] + [train[attr[1]] for attr in attributes])
        vectorizer.fit(combined_train)

        # Compute aggregated Euclidean distances for validation and test datasets
        valid_scores = compute_aggregated_distance(valid, vectorizer, attributes)
        test_scores = compute_aggregated_distance(test, vectorizer, attributes)

        # Validation set evaluation
        valid_label = np.array(valid['label'])
        thresholds = np.arange(0, 2, 0.1)
        fscore = []
        for threshold in thresholds:
            fscore.append(precision_recall_fscore_support(valid_label, valid_scores < threshold, average='binary')[2])

        # Find the best threshold
        best_threshold = thresholds[np.argmax(fscore)]

        # Test set evaluation using the best threshold
        test_label = np.array(test['label'])
        precision, recall, fscore, _ = precision_recall_fscore_support(
            test_label, test_scores < best_threshold, average='binary'
        )

        # Format values to two decimal places and scale to percentage
        precision = round(precision * 100, 2)
        recall = round(recall * 100, 2)
        fscore = round(fscore * 100, 2)

        # Write the formatted results to the file
        with open('parameter_tuning_tfidf_ed.txt', 'a') as f:
            f.write(f"Dataset: {dataset_name} Index: {dataset_index} ")
            f.write(f"Precision: {precision} Recall: {recall} F-score: {fscore}\n")