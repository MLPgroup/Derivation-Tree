'''
Description: Python code to parse json file and store manually parsed articles
Author: Vishesh Prasad
Modification Log:
    October 2, 2023: Created file and wrote initial foundational code to store articles in dictionary
    October 22, 2023: Create function to use parser code
'''

def get_manually_parsed_articles():
    # Import modules
    import json

    # Open json file and store into dicionary
    with open('articles.json') as json_file:
        # Load list of articles
        json_data = json.load(json_file)

        # Dictionary of manually parsed articles
        articles = json_data['Manually Parsed Articles']
        manually_parsed_articles = {}
        for article in articles:
            manually_parsed_articles[article['Article ID']] = article
    
    return manually_parsed_articles
