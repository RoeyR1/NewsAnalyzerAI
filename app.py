import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import os

# Define the model class


class HeadlineScoreModel(nn.Module):
    def __init__(self, n_regression_outputs, n_categories):
        super(HeadlineScoreModel, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.regression_head = nn.Linear(
            self.bert.config.hidden_size, n_regression_outputs)
        self.classification_heads = nn.ModuleList([
            nn.Linear(self.bert.config.hidden_size, 1) for _ in range(n_categories)
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the first token ([CLS])
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.drop(pooled_output)
        regression_output = self.regression_head(output)
        classification_outputs = [head(output)
                                  for head in self.classification_heads]
        # Shape: (batch_size, n_categories)
        classification_output = torch.cat(classification_outputs, dim=1)
        return regression_output, classification_output


# Function to handle predictions
def predict_headline(model, tokenizer, category_names, headline, device, max_len=183):
    model.eval()
    encoding = tokenizer.encode_plus(
        headline,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        regression_output, classification_output = model(
            input_ids=input_ids, attention_mask=attention_mask)

    regression_output = regression_output.cpu().numpy()[0]
    classification_output = torch.sigmoid(
        classification_output).cpu().numpy()[0]

    regression_results = {
        'Sentiment': regression_output[0],
        'Intensity': regression_output[1],
        'Urgency': regression_output[2],
        'Controversy': regression_output[3]
    }

    classification_results = {}
    for idx, category in enumerate(category_names):
        classification_results[category] = bool(
            classification_output[idx] >= 0.5)

    return regression_results, classification_results


# Load model and tokenizer
def load_model():
    model_path = 'model/distilbert_news_classifier_epoch16.pth'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeadlineScoreModel(n_regression_outputs=4, n_categories=5)
    model.load_state_dict(torch.load(
        model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    category_names = ['Health', 'Environment',
                      'Technology', 'Finance', 'Politics']
    return model, tokenizer, category_names, device


# Function to analyze a headline and return the results
def analyze_headline(headline, model, tokenizer, category_names, device):
    try:
        regression_results, classification_results = predict_headline(
            model=model,
            tokenizer=tokenizer,
            category_names=category_names,
            headline=headline,
            device=device
        )
        # Handle scores out of range
        if regression_results['Sentiment'] < 0:
            regression_results['Sentiment'] = 0
        if regression_results['Intensity'] > 10:
            regression_results['Intensity'] = 10

        output = f"""📊 Analysis Results:

Metrics:
- Sentiment: {regression_results['Sentiment']:.2f}/10
- Intensity: {regression_results['Intensity']:.2f}/10
- Urgency: {regression_results['Urgency']:.2f}/10
- Controversy: {regression_results['Controversy']:.2f}/10

📑 Categories: {', '.join([cat for cat, val in classification_results.items() if val]) or 'None'}
"""
        return output
    except Exception as e:
        return f"Error analyzing headline: {str(e)}"


# Main function to run chatbot in the terminal
def main():
    print("Welcome to the News Headline Analyzer chatbot!")
    model, tokenizer, category_names, device = load_model()

    while True:
        try:
            headline = input(
                "\nEnter a news headline (or 'quit' to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if headline.lower() == 'quit':
            print("Goodbye!")
            break
        if not headline:
            print("Empty input. Please enter a valid headline.")
            continue

        # Get the analysis results for the headline
        output = analyze_headline(
            headline, model, tokenizer, category_names, device)

        # Print the output to the terminal
        print(output)


# Run the chatbot
if __name__ == "__main__":
    main()
