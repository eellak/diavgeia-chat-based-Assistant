path = "/Users/geoant13/Documents/GitHub/Diavgeia/diaygeia/entrypoints/data"
#path = "/home/diav_user/diaygeia/diaygeia/entrypoints/data"
#path = "/Users/geoant13/Desktop/diavgeia/txts"

import os
from elasticsearch import Elasticsearch

###
# Create a connection to Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Define the index configuration and mappings (if needed)
index_config = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "content": {"type": "text"}
        }
    }
}

# Index name
index_name = "diaygeia"

# Create the index if it doesn't exist
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=index_config)


# Iterate over downloaded data and index each document
for file_name in os.listdir(path):
    if file_name.endswith(".txt"):
        with open(os.path.join(path, file_name), "r", encoding="utf-8") as file:
            key = file_name[:-4]  # Extract key from file name (remove ".txt" extension)
            content = file.read()
            doc = {
                "content": content
            }
            # Indexing the document
            es.index(index=index_name, id=key, document=doc)

print("Indexing complete.")

