import argparse
from pathlib import Path
from typing import Optional

import torch
import transformers
from transformers import AutoTokenizer
from transformers.generation_stopping_criteria import StoppingCriteria

import adatest

class EOSStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_tokens: list[str], tokenizer: transformers.PreTrainedTokenizer) -> None:
        self.eos_tokens = set(eos_tokens)
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # have to convert at point of generation because of tokens like `."`
        last_token = self.tokenizer.convert_ids_to_tokens(input_ids[:,-1].item())
        return any(t in last_token for t in self.eos_tokens)

if __name__ == "__main__":
    # create a HuggingFace sentiment analysis model
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_tree_fpath")
    parser.add_argument("--model", default="gpt2", help="Model to use, e.g., `EleutherAI/gpt-j-6B` or `gpt-2`")
    parser.add_argument("--revision", default=None, help="Model revision")
    parser.add_argument("--pipeline", default="text-generation")
    parser.add_argument("--eos_tokens", nargs="*", default=[])
    parser.add_argument("--openai_model", default="curie")
    parser.add_argument("--openai_api_key", default="~/.openai_api_key", help="Path to OpenAI key (or key itself)")
    parser.add_argument("--seed", type=int, default=11235)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    
    try:
        api_key = Path(args.openai_api_key).read_text().strip()
    except FileNotFoundError:
        api_key = args.openai_api_key

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    eos_stopping_criteria = EOSStoppingCriteria(args.eos_tokens, tokenizer)
    suffix_generator = transformers.pipeline(
        "text-generation",
        model=args.model,
        revision=args.revision,
        return_all_scores=True,
        stopping_criteria=[eos_stopping_criteria], # using StoppingCriteriaList seems unnecessary
        device=0 if torch.cuda.is_available() else -1,
    )
        
    # specify the backend generator used to help you write tests
    prefix_generator = adatest.generators.OpenAI(
        'curie',
        top_p=0.95,
        filter=None,
        api_key=api_key
    )

    # ...or you can use an open source generator
    #neo = transformers.pipeline('text-generation', model="EleutherAI/gpt-neo-125M")
    #prefix_generator = adatest.generators.Transformers(neo.model, neo.tokenizer)

    # create a new test tree
    tests = adatest.TestTree(args.test_tree_fpath)

    # adapt the tests to our model to launch a notebook-based testing interface
    # (wrap with adatest.serve to launch a standalone server)
    adatest.serve(tests.adapt(suffix_generator, prefix_generator, auto_save=True), port=8888)