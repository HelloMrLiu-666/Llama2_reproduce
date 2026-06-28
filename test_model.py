from my_model import RMSNorm
from transformers import AutoTokenizer

def test_RMS_NORM():
    RMS_NORM=RMSNorm(dim: int, eps: float = 1e-6)
    tokenizer = AutoTokenizer.from_pretrained("TinyPixel/Llama-2-7B-bf16-sharded")


if __name__ == '__main__':