from datasets import load_dataset
import pickle
import pandas as pd
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the dataset
df = pd.read_parquet("hf://datasets/rag-datasets/rag-mini-bioasq/data/passages.parquet/part.0.parquet")

# Combine all passages into one big passage
combined_text = " ".join(df['passage'].tolist())

# Create a single Document object with the combined text
combined_document = Document(page_content=combined_text)

# Split the combined document into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=600,
    chunk_overlap=100,
    encoding_name='cl100k_base'
)

# Perform the splitting
split_documents = text_splitter.split_documents([combined_document])

# Save the split documents
with open("bioasq_split_documents.pickle", 'wb') as fw:
    pickle.dump(split_documents, fw)

with open("bioasq_split_documents.pickle", 'rb') as fw:
    split_documents = pickle.load(fw)

# Output the number of chunks
print(f"Number of split documents: {len(split_documents)}")
