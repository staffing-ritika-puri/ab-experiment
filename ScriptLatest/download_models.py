from transformers import (
    BertTokenizer, BertModel,
    AutoTokenizer, AutoModelForSequenceClassification
)

# Download BERT base uncased
print("Downloading bert-base-uncased...")
BertTokenizer.from_pretrained("bert-base-uncased")
BertModel.from_pretrained("bert-base-uncased")
print("✓ BERT downloaded")

# Download NLI model
print("Downloading microsoft/deberta-large-mnli...")
AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli")
print("✓ DeBERTa MNLI downloaded")
