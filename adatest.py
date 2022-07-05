from pathlib import Path

import transformers
import adatest

OPENAI_API_KEY = Path("secret.txt").read_text().strip()

# create a HuggingFace sentiment analysis model
generator_to_test = transformers.pipeline("text-generation", return_all_scores=True)

# specify the backend generator used to help you write tests
generator = adatest.generators.OpenAI('curie', api_key=OPENAI_API_KEY)

# ...or you can use an open source generator
#neo = transformers.pipeline('text-generation', model="EleutherAI/gpt-neo-125M")
#generator = adatest.generators.Transformers(neo.model, neo.tokenizer)

# create a new test tree
tests = adatest.TestTree("sensitive.csv")

# adapt the tests to our model to launch a notebook-based testing interface
# (wrap with adatest.serve to launch a standalone server)
adatest.serve(tests.adapt(generator_to_test, generator, auto_save=True), port=8081)