# Below is an example of converting text into tokens.
# This is useful for understand how large documents are before embedding them. 
# Tokenisation is a useful method because context windows are measured in tokens.
# ~4 chars = 1 token
import tiktoken

question = "What kinds of pets do I like?"
document = "My favorite pet is a cat."

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

print(num_tokens_from_string(question, "cl100k_base"))