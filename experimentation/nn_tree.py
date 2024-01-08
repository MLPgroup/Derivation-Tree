import torch
import torch.nn as nn
import torch.optim as optim
from bs4 import BeautifulSoup
import spacy

# Sample HTML file with LaTeX equations
html_content = """
<html>
  <head>
    <title>Derivation Tree Example</title>
  </head>
  <body>
    <div id="equation1">
      <label>Equation 1:</label>
      \(a = b + c\)
    </div>
    <div id="equation2">
      <label>Equation 2:</label>
      \(d = a \times e\)
    </div>
    <!-- Add more equations as needed -->
  </body>
</html>
"""

# Parse HTML content using BeautifulSoup
soup = BeautifulSoup(html_content, 'html.parser')

# Extract LaTeX equations
equations = [div.get_text() for div in soup.find_all('div')]

# Dummy training data with correct derivation trees
training_data = [
    {'input': 'Equation 1', 'output': 'Equation 2'},
    # Add more training data as needed
]

# Tokenize equations using spaCy
nlp = spacy.load("en_core_web_sm")

def tokenize_equation(equation):
    return [token.text for token in nlp(equation)]

# Create vocabulary
vocab = set()
for equation in equations:
    vocab.update(tokenize_equation(equation))
vocab = list(vocab)

# Map tokens to indices
token2index = {token: i for i, token in enumerate(vocab)}
index2token = {i: token for i, token in enumerate(vocab)}

# Convert equations to token indices
equations_indices = [[token2index[token] for token in tokenize_equation(equation)] for equation in equations]

# Define a simple neural network model
class DerivationTreeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DerivationTreeModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.sum(x, dim=1)  # Sum embeddings along the sequence dimension
        x = self.fc(x)
        return x

# Initialize the model
input_size = len(vocab)
hidden_size = 64
output_size = len(equations)
model = DerivationTreeModel(input_size, hidden_size, output_size)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    for data in training_data:
        input_equation = equations_indices[equations.index(data['input'])]
        output_equation = equations_indices[equations.index(data['output'])]

        # Convert to PyTorch tensors
        input_tensor = torch.tensor(input_equation, dtype=torch.long).view(1, -1)
        target_tensor = torch.tensor(output_equation, dtype=torch.long)

        # Forward pass
        output = model(input_tensor)

        # Compute loss
        loss = criterion(output, target_tensor)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

# Save the trained model if needed
torch.save(model.state_dict(), 'derivation_tree_model.pth')
