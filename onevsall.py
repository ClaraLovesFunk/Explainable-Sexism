import torch
from transformers import pipeline, AutoModel, AutoTokenizer

if __name__ == "__main__":
    device = 0 if torch.cuda.is_available() else -1
    model = AutoModel.from_pretrained("microsoft/deberta-v2-xxlarge")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xxlarge")
    pipe = pipeline(
            'feature-extraction',
            model = model,
            tokenizer = tokenizer,
            device = device,
    )
    res = pipe("Test sentence")
    print(res)

    # TODO:
    # ensemble and HF do not work well together
    # look into workarounds
    # switch back to pure torch implementation?
