# Below is an example of converting text to embeddings.
# Each result returns a fixed length vector that represents the semantics of the text.
import numpy as np

question = "What kinds of pets do I like?"
document = "My favorite pet is a cat."

from langchain_openai import OpenAIEmbeddings
embd = OpenAIEmbeddings()
query_result = embd.embed_query(question)
document_result = embd.embed_query(document)

# Following this, we can compare the similarity between the two vectors.
# For the OpenAI Embedding model it is recommended to use cosine similarity. https://platform.openai.com/docs/guides/embeddings/frequently-asked-questions
# The closer to 1 the result is, the more similar the two vectors are.

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

similarity = cosine_similarity(query_result, document_result)
print("Cosine Similarity:", similarity)