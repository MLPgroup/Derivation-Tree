'''
Description: Python utils file to use Naive Bayes to get derivation graphs
Author: Vishesh Prasad
Modification Log:
    August 18, 2024: create file and transfer code in
'''



# Import Modules
import random
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer



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
                if equation_indexing[j] in adjacency_list[equation_indexing[i]]:
                    label = 1
                elif equation_indexing[i] in adjacency_list[equation_indexing[j]]:
                    label = -1
                labels.append(label)
            features.append(feature_vector)

    if adjacency_list is not None:
        return features, labels
    else:
        return features



"""
bayes_classifier(article_ids, articles_used, extract_equations, extracted_words_between_equations)
Input: article_ids -- dictionary with info on all articles from articles.json
       articles_used -- list of articles where equations were extracted correctly
       extracted_equations -- list of equations that were successfully extracted
       extracted_words_between_equation -- list of list of words that occur between equations
       extracted_equation_indexing -- list of list of equations in the order they were found from the article
       bayes_training_percentage -- percentage of dataset to use for training of Naive Bayes model
Return: true_adjacency_lists -- list of labeled adjacency lists used in the test phase of the naive bayes algorithm
        predicted_adjacency_lists -- list of predicted adjacency lists resulting from the test phase of the naive bayes algorithm
        train_article_ids -- list of article ids used to train the classifier
Function: Predict adjacency list using the naive bayes algorithm
"""
def bayes_classifier(article_ids, articles_used, extracted_equations, extracted_words_between_equations, extracted_equation_indexing, bayes_training_percentage):
    # Initialize lists to store true and predicted adjacency lists
    true_adjacency_lists = []
    predicted_adjacency_lists = []

    # Split the data set into test and train
    num_articles = len(articles_used)
    # train_random_indices = range(int(num_articles * (bayes_training_percentage * 1.0 / 100)))
    train_size = int(num_articles * (bayes_training_percentage / 100))
    train_random_indices = random.sample(range(num_articles), train_size)

    train_features = []
    train_labels = []
    train_article_ids = []

    for i in train_random_indices:
        equations = extracted_equations[i]
        words_between_eqs = extracted_words_between_equations[i]
        equation_indexing = extracted_equation_indexing[i]

        features, labels = extract_features_and_labels(equations, words_between_eqs, equation_indexing, article_ids[articles_used[i]]["Adjacency List"])

        train_features.extend(features)
        train_labels.extend(labels)

        train_article_ids.append(article_ids[articles_used[i]]["Article ID"])

    # Train the Naive Bayes classifier
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(train_features)
    y_train = train_labels

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # Predict connections for the remaining articles
    for i in range(num_articles):
        if i not in train_random_indices:
            equations = extracted_equations[i]
            words_between_eqs = extracted_words_between_equations[i]
            equation_indexing = extracted_equation_indexing[i]

            features = extract_features_and_labels(equations, words_between_eqs, equation_indexing)
            X_test = vectorizer.transform(features)

            # Predict labels
            predictions = classifier.predict(X_test)
            predicted_adjacency_list = {equation_id: [] for equation_id in equation_indexing}
            predicted_index = 0
            # Extract predictions to form adjacency list
            for j in range(len(equation_indexing)):
                for k in range(j+1, len(equation_indexing)):
                    if predictions[predicted_index] == 1:
                        predicted_adjacency_list[equation_indexing[j]].append(equation_indexing[k])
                    elif predictions[predicted_index] == -1:
                        predicted_adjacency_list[equation_indexing[k]].append(equation_indexing[j])
                    predicted_index += 1

            predicted_adjacency_lists.append(predicted_adjacency_list)
            true_adjacency_lists.append(article_ids[articles_used[i]]["Adjacency List"])

    return true_adjacency_lists, predicted_adjacency_lists, train_article_ids