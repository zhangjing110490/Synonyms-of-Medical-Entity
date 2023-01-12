# Generate Synonyms for Medical Entities based on Pretrained Word Embeddings

## Table of Contents

[comment]: <> (- [Security]&#40;#security&#41;)
* [Logic](#Logic)
* [Usage](#Usage)



[comment]: <> (- [Usage]&#40;#usage&#41;)
[comment]: <> (- [API]&#40;#api&#41;)
[comment]: <> (## Security)

## Logic
Step 1: Load medical entities from your own corpus file \
Step 2: Load Pre-trained Medical Word Embeddings and filter those not in your corpus file \
Step 3: Calculate the cosine similarity between each medical pair and use those with high similarity as synonyms \
Step 4: Save synonyms for later use


## Usage
1. Put your own data files containing entities you are interested in "Corpus" folder. \
2. Put your own pretrained word embedding file in "WordEmbeddings" folder. \
3. Specify parameters like threshold, max_num
4. Execute "generate_synonyms.py"