# pre-trained
# Ignore insignificant warnings (ex: deprecations)
import warnings
warnings.filterwarnings('ignore')

# Set a seed for reproducibility
import torch

def fix_torch_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fix_torch_seed()

model_path_or_name = "upstage/TinySolar-248m-4k-code-instruct"

from transformers import AutoModelForCausalLM
tiny_general_model = AutoModelForCausalLM.from_pretrained(
    model_path_or_name,
    device_map="cpu", # change to auto if you have access to a GPU
    torch_dtype=torch.bfloat16
)

from transformers import AutoTokenizer
tiny_general_tokenizer = AutoTokenizer.from_pretrained(
    model_path_or_name
)

prompt = "I am an engineer. I love"

inputs = tiny_general_tokenizer(prompt, return_tensors="pt")

from transformers import TextStreamer
streamer = TextStreamer(
    tiny_general_tokenizer,
    skip_prompt=True, # If you set to false, the model will first return the prompt and then the generated text
    skip_special_tokens=True
)

outputs = tiny_general_model.generate(
    **inputs,
    streamer=streamer,
    use_cache=True,
    max_new_tokens=128,
    do_sample=False,
    temperature=0.0,
    repetition_penalty=1.1
)

inputs = tiny_general_tokenizer("I love you", return_tensors="pt") # Use tiny_general_tokenizer instead of tokenizer

from transformers import TextStreamer
streamer = TextStreamer(tiny_general_tokenizer)  # Use tiny_general_tokenizer

generated_tokens = tiny_general_model.generate(
    **inputs,
    streamer=streamer,
    use_cache=True,
    max_new_tokens=128,
    do_sample=False,
    temperature=0.0,
    repetition_penalty=1.1
)
# Decode generated tokens using tiny_general_tokenizer
generated_text = tiny_general_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(generated_text)

prompt =  "def find_max(numbers):"

inputs = tiny_general_tokenizer(
    prompt, return_tensors="pt"
).to(tiny_general_model.device)

streamer = TextStreamer(
    tiny_general_tokenizer,
    skip_prompt=True, # Set to false to include the prompt in the output
    skip_special_tokens=True
)

outputs = tiny_general_model.generate(
    **inputs,
    streamer=streamer,
    use_cache=True,
    max_new_tokens=128,
    do_sample=False,
    temperature=0.0,
    repetition_penalty=1.1
)

model_path_or_name = "upstage/TinySolar-248m-4k-py"

tiny_custom_model = AutoModelForCausalLM.from_pretrained(
    model_path_or_name,
    device_map="cpu",
    torch_dtype=torch.bfloat16,
)
tiny_custom_tokenizer = AutoTokenizer.from_pretrained(
    model_path_or_name
)

prompt = "def find_max(numbers):"

inputs = tiny_custom_tokenizer(
    prompt, return_tensors="pt"
).to(tiny_custom_model.device)

streamer = TextStreamer(
    tiny_custom_tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)
outputs = tiny_custom_model.generate(
    **inputs, streamer=streamer,
    use_cache=True,
    max_new_tokens=128,
    do_sample=False,
    repetition_penalty=1.1
)

def find_max(numbers):
   max = 0
   for num in numbers:
       if num > max:
           max = num
   return max

   find_max([1,3,5,1,6,17,2])

   **Data preparation**
   an Upstage tool called **Dataverse** which can help you with data cleaning.

   import warnings
   warnings.filterwarnings("ignore")

   !pip install datasets # Install the missing 'datasets' module.
    import datasets

   pretraining_dataset = datasets.load_dataset(
    "upstage/Pretraining_Dataset",
    split="train"
     )

     print(pretraining_dataset)

     pretraining_dataset = pretraining_dataset.select_columns(
    ["text"]
    )
     print(pretraining_dataset[0]["text"][:500])

     instruction_dataset = datasets.load_dataset(
    "c-s-ale/alpaca-gpt4-data",
    split='train'
    )
    print(instruction_dataset)

    i=0
     print("Instruction: " + instruction_dataset[i]["instruction"]
      + "\nInput: " + instruction_dataset[i]["input"]
      + "\nOutput: " + instruction_dataset[i]["output"])


   # Import some required packages
import os
import requests

# Path to directory to store python scripts
code_dir = "./code"

urls = [
"https://www.coursera.org/projects/genai-for-business-analysis-fine-tuning-llms"
"https://www.coursera.org/projects/create-survey-analyze-results-with-surveymonkey"
"https://www.coursera.org/learn/transformer-models-and-bert-model"
"https://www.coursera.org/projects/deep-learning-with-pytorch-image-segmentation"
"https://www.coursera.org/projects/machine-learning-with-chatgpt-image-classification-model"
"https://www.coursera.org/learn/introduction-to-generative-ai/lecture/TJ28r/introduction-to-generative-ai"
]

# Import some required packages
import os
import requests # Added import statement

# Path to directory to store python scripts
code_dir = "./code" # Moved code_dir definition to the top

urls = [
"https://www.coursera.org/projects/genai-for-business-analysis-fine-tuning-llms", # Fixed missing comma
"https://www.coursera.org/projects/create-survey-analyze-results-with-surveymonkey",
"https://www.coursera.org/learn/transformer-models-and-bert-model",
"https://www.coursera.org/projects/deep-learning-with-pytorch-image-segmentation",
"https://www.coursera.org/projects/machine-learning-with-chatgpt-image-classification-model",
"https://www.coursera.org/learn/introduction-to-generative-ai/lecture/TJ28r/introduction-to-generative-ai"
]

for url in urls:
    print(f"Working on url: {url}")
    response = requests.get(url)
    file_name = os.path.basename(url)
    file_path = os.path.join(code_dir, file_name)

    # Create the directory if it doesn't exist
    os.makedirs(code_dir, exist_ok=True)

    with open(file_path, "wb") as file:
        file.write(response.content)

        files = os.listdir(code_dir)
    for file in files:
       print(file)

       code_dataset = []
for file in os.listdir(code_dir):
    code_dataset.append(
        {'text': open(os.path.join(code_dir, file), 'r').read()}
    )

    code_dataset = datasets.Dataset.from_list(code_dataset)
print(code_dataset)

dataset = datasets.concatenate_datasets(
    [pretraining_dataset, code_dataset]
)
print(dataset)

DATA CLEANING

dataset.num_rows

import heapq

def paragraph_length_filter(x):
    """Returns False iff a page has too few lines or lines are too short."""
    lines = x['text'].split('\n')
    if (
        len(lines) < 3
        or min(heapq.nlargest(3, [len(line) for line in lines])) < 3
    ):
        return False
    return True

    dataset = dataset.filter(
    paragraph_length_filter,
    load_from_cache_file=False
)

dataset.num_rows

def find_duplicates(paragraphs):
    """
    Use this function to find the number of repetitions
    in the paragraphs.
    """
    unique_x = set()
    duplicate_chars = 0
    duplicate_elements = 0
    for element in paragraphs:
        if element in unique_x:
            duplicate_chars += len(element)
            duplicate_elements += 1
        else:
            unique_x.add(element)
    return duplicate_elements, duplicate_chars

    import re
    def paragraph_repetition_filter(x):
    """
    Returns False iff a page has too many repetitions.
    """
    text = x['text']
    paragraphs = re.compile(r"\n{2,}").split(text.strip())                # Split by paragraphs (2 or more newlines)
    paragraphs_duplicates, char_duplicates = find_duplicates(paragraphs)  # Find number of duplicates in paragraphs
    if paragraphs_duplicates / len(paragraphs) > 0.3:
        return False
    if char_duplicates / len(text) > 0.2:
        return False
    return True

    dataset = dataset.filter(
    paragraph_repetition_filter,
    load_from_cache_file=False
    )

dataset.num_rows

### Deduplication

def deduplication(ds):
    def dedup_func(x):
        """Use this function to remove duplicate entries"""
        if x['text'] in unique_text:
            return False
        else:
            unique_text.add(x['text'])
            return True

    unique_text = set()

    ds = ds.filter(dedup_func, load_from_cache_file=False, num_proc=1)
    return ds

dataset = deduplication(dataset)

dataset.num_rows

### Quality filter - Language
!pip install fasttext
!pip install transformers datasets
import fasttext
import os
import urllib
from fasttext.FastText import _FastText

def english_language_filter(ds):
    # Download the model if it doesn't exist
    model_path = 'lid.176.bin'
    if not os.path.exists(model_path):
        !wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin

    # load language detection model
    model = fasttext.load_model(model_path) # Load the model using fasttext.load_model

    def is_english(x):
        # Predict language of the text and probability
        language, score = model.predict(x['text'].replace("\n", ""))

        language = language[0].replace("__label__", "") # Remove "__label__" from the predicted language
        return score > 0.4 and language == "en" # change code here if building a model in another language

    ds = ds.filter(is_english, load_from_cache_file=False, num_proc=1)
    return ds

dataset = english_language_filter(dataset)

def download_model(model_url, model_path):
    """Download the fastText language detection model if not already present."""
    if not os.path.exists(model_path):
        print("Downloading model...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Model downloaded.")
    else:
        print("Model already exists.")

def load_model(model_path):
    """Load the fastText language detection model."""
    return fasttext.load_model(model_path)

def is_english(text, model, threshold=0.4):
    """Check if the given text is in English."""
    predictions = model.predict(text.replace("\n", "")) # this line was missing and is causing the error
    language = predictions[0][0].split("__")[2]  # Extract the language code
    score = predictions[1][0]  # Get the confidence score
    return score > threshold and language == "en"

def filter_english_texts(dataset, model):
    """Filter out texts that are not in English from the dataset."""
    filtered_dataset = [entry for entry in dataset if is_english(entry['text'], model)]
    return filtered_dataset

def main():
    # Define model path and URL
    model_url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
    model_path = 'lid.176.bin'

    # Download and load the model
    download_model(model_url, model_path)
    model = load_model(model_path)

    # Filter the dataset
    filtered_dataset = filter_english_texts(dataset, model)
    print("Filtered Dataset:", filtered_dataset)

if __name__ == "__main__":
    main()

    dataset.num_rows

    file_path = "./data/preprocessed_dataset.parquet"
dataset.to_parquet(file_path)

# Lesson 3: Data Packaging
## 1. Tokenizing and creating input_ids

Start by loading the dataset from the previous lesson:
import datasets

dataset = datasets.load_dataset(
    "parquet",
    data_files="./data/preprocessed_dataset.parquet",
    split="train"
)
print(dataset)

dataset = dataset.shard(num_shards=10, index=0)
print(dataset)

from transformers import AutoTokenizer

# If the model is on the Hugging Face Hub, use the repository ID
model_path_or_name = "yifeihu/TB-OCR-preview-0.1"

# If the model is local, make sure the directory structure is correct and contains the necessary files
# model_path_or_name = "./models/SOLAR-10.7B-v1.0"

tokenizer = AutoTokenizer.from_pretrained(
    model_path_or_name,
    use_fast=True, # Changed to use_fast=True as this model supports fast tokenizers
    token='<your_token>' # Add your Hugging Face token here
)

tokenizer.tokenize("My name is Keerthi")

def tokenization(example):
    # Tokenize
    tokens = tokenizer.tokenize(example["text"])

    # Convert tokens to ids
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Add <bos>, <eos> tokens to the front and back of tokens_ids
    # bos: begin of sequence, eos: end of sequence
    token_ids = [
        tokenizer.bos_token_id] \
        + token_ids \
        + [tokenizer.eos_token_id
    ]
    example["input_ids"] = token_ids

    # We will be using this column to count the total number of tokens
    # in the final dataset
    example["num_tokens"] = len(token_ids)
    return example

    dataset = dataset.map(tokenization, load_from_cache_file=False)
print(dataset)

sample = dataset[2]
print("text", sample["text"][:30]) #
print("\ninput_ids", sample["input_ids"][:30])
print("\nnum_tokens", sample["num_tokens"])

import numpy as np
np.sum(dataset["num_tokens"])

## 2. Packing the data

input_ids = np.concatenate(dataset["input_ids"])
print(len(input_ids))

max_seq_length = 32

total_length = len(input_ids) - len(input_ids) % max_seq_length
print(total_length)

total_length = len(input_ids) - len(input_ids) % max_seq_length
print(total_length)

input_ids = input_ids[:total_length]
print(input_ids.shape)

input_ids_reshaped = input_ids.reshape(-1, max_seq_length).astype(np.int32)
input_ids_reshaped.shape

type(input_ids_reshaped)

input_ids_list = input_ids_reshaped.tolist()
packaged_pretrain_dataset = datasets.Dataset.from_dict(
    {"input_ids": input_ids_list}
)
print(packaged_pretrain_dataset)


