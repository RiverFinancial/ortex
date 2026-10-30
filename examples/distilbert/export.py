"""
### Install dependencies:

   $ pip install transformers
   $ pip install optimum
   $ pip install "transformers[onnx]"

"""

from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import DistilBertTokenizer

save_directory = "./models/distilbert-onnx/"

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = ORTModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english", export=True
)
print(model)

model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
