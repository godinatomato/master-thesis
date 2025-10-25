from datasets import load_dataset

# fin_dataset = load_dataset("takala/financial_phrasebank", "sentences_allagree", trust_remote_code=True)
# fin_dataset = load_dataset("takala/financial_phrasebank", "sentences_75agree")
# fin_dataset["train"]["sentence"]

# med_dataset = load_dataset("lavita/MedQuad")
# med_dataset["train"]["answer"]

# law_dataset = load_dataset("casehold/casehold")
# print(law_dataset)

# daily_dataset = load_dataset("stanfordnlp/imdb")
# iily_dataset["train"]["text"]
# print(daily_dataset)

# aca_dataset = load_dataset("gfissore/arxiv-abstracts-2021")
# aca_dataset["train"]["abstract"]
# print(aca_dataset)

math_dataset = load_dataset("openai/gsm8k", "main")
math_dataset["train"]["qusetion"]
print(math_dataset)


