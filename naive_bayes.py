'''
Description: Python utils file to use Naive Bayes to get derivation graphs
Author: Vishesh Prasad
Modification Log:
    August 18, 2024: create file and transfer code in
'''



# Import Modules
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold



"""
extract_features_and_labels(equations, words_between_equations, equation_indexing, adjacency_list)
Input: equations -- list of equations that were successfully extracted
       words_between_equations -- list of words that occur between equations
       equation_indexing -- list of equations in the order they were found from the article
       adjacency_list (optional) -- adjacency list used to extract labels
Return: features -- extracted features of equations and words between equations 
        labels -- labels of if one equation is connected to another and the direction (+1 if 'i' points to 'j', -1 if 'j' points to 'i', and 0 for no connection)
Function: Feature and label extraction for naive bayes where a feature contains all words that occur between two equations and the two equations themselves amd label specifies their connection
"""
def extract_features_and_labels(equations, words_between_equations, equation_indexing, adjacency_list=None):
    features = []
    labels = []
    for i in range(len(equation_indexing)):
        for j in range(i+1, len(equation_indexing)):
            # Feature extraction
            # Words before 1st equation
            feature_vector = words_between_equations[j] + " "
            # 1st equation
            for k in range(len(equations[equation_indexing[i]]['equations'])):
                feature_vector += equations[equation_indexing[i]]['equations'][k]['alttext'] + " " 
            # Words between the equations
            for k in range(i + 1, j):
                feature_vector += words_between_equations[k] + " "
            # 2nd equation
            for k in range(len(equations[equation_indexing[j]]['equations'])):
                feature_vector += equations[equation_indexing[j]]['equations'][k]['alttext'] + " "
            # Words after the 2nd equation
            feature_vector += words_between_equations[j + 1] if j + 1 < len(words_between_equations) else ""

            if adjacency_list is not None:
                # Label extraction
                label = 0
                if equation_indexing[j] in adjacency_list.get(equation_indexing[i], []):
                    label = 1
                elif equation_indexing[i] in adjacency_list.get(equation_indexing[j], []):
                    label = -1
                labels.append(label)
            features.append(feature_vector)

    if adjacency_list is not None:
        return features, labels
    else:
        return features



"""
bayes_classifier(article_ids, articles_used, extracted_equations, extracted_words_between_equations,
                 extracted_equation_indexing, k_folds)
Input: article_ids               -- dict of all articles from mdgd.json
       articles_used             -- list of article IDs where equations were extracted correctly
       extracted_equations       -- list of equation dicts per article
       extracted_words_between_equations -- list of word lists per article
       extracted_equation_indexing       -- list of equation orderings per article
       k_folds                   -- number of folds for cross-validation
Return: true_adjacency_lists      -- ground-truth adjacency lists for every test article across all folds
        predicted_adjacency_lists -- predicted adjacency lists for every test article across all folds
        []                        -- (no meaningful train-split concept in k-fold; kept for interface compatibility)
Function: k-fold cross-validated Naive Bayes classifier; every article is evaluated exactly once as a test article
"""
def bayes_classifier(article_ids, articles_used, extracted_equations, extracted_words_between_equations, extracted_equation_indexing, k_folds):
    num_articles = len(articles_used)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    true_adjacency_lists = []
    uncleaned_predicted_adjacency_lists = []

    for train_indices, test_indices in kf.split(range(num_articles)):
        # Build training features and labels for this fold
        train_features = []
        train_labels = []
        for i in train_indices:
            features, labels = extract_features_and_labels(
                extracted_equations[i],
                extracted_words_between_equations[i],
                extracted_equation_indexing[i],
                article_ids[articles_used[i]]["Adjacency List"]
            )
            train_features.extend(features)
            train_labels.extend(labels)

        # Fit a fresh vectorizer and classifier on this fold's training data
        vectorizer = CountVectorizer()
        X_train = vectorizer.fit_transform(train_features)
        classifier = MultinomialNB()
        classifier.fit(X_train, train_labels)

        # Predict for each article in the test fold
        for i in test_indices:
            equation_indexing = extracted_equation_indexing[i]
            features = extract_features_and_labels(
                extracted_equations[i],
                extracted_words_between_equations[i],
                equation_indexing
            )
            predictions = classifier.predict(vectorizer.transform(features))

            predicted_adjacency_list = {eq: [] for eq in equation_indexing}
            pred_idx = 0
            for j in range(len(equation_indexing)):
                for k in range(j + 1, len(equation_indexing)):
                    if predictions[pred_idx] == 1:
                        predicted_adjacency_list[equation_indexing[j]].append(equation_indexing[k])
                    elif predictions[pred_idx] == -1:
                        predicted_adjacency_list[equation_indexing[k]].append(equation_indexing[j])
                    pred_idx += 1

            uncleaned_predicted_adjacency_lists.append(predicted_adjacency_list)
            true_adjacency_lists.append(article_ids[articles_used[i]]["Adjacency List"])

    # Normalise empty adjacency lists to [None]
    cleaned_predicted_adjacency_lists = []
    for pred_adj in uncleaned_predicted_adjacency_lists:
        cleaned_predicted_adjacency_lists.append({
            eq: (neighbors if neighbors else [None])
            for eq, neighbors in pred_adj.items()
        })

    return true_adjacency_lists, cleaned_predicted_adjacency_lists, []