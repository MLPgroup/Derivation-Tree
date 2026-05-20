import json

# Read the articles.json file
with open('articles.json', 'r') as f:
    data = json.load(f)

# Extract all Article IDs from Manually Parsed Articles
article_ids = [article["Article ID"] for article in data["Manually Parsed Articles"]]

# Create output structure
output = {"article_ids": article_ids}

# Save to a new JSON file
with open('article_ids.json', 'w') as f:
    json.dump(output, f, indent=4)

print(f"Extracted {len(article_ids)} article IDs")
print(f"Saved to article_ids.json")
