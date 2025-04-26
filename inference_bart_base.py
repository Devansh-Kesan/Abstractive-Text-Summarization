from transformers import BartTokenizer, BartForConditionalGeneration

# Load BART-base or BART-large-cnn (better for summarization)
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Prepare input text
# article = """
# Legal case judgement documents are usually very lengthy and unstructured, making it difficult
# for legal professionals to read them and understand the key information. Also, reading such
# lengthy legal documents is time-consuming where the text is around 4500 words on an average
# [1]. In this direction, if some shorter versions are available for these lengthy documents, it
# would be beneficial for lawyers, judges, lawmakers, and ordinary citizens. To deal with this,
# the organizers of the FIRE 2021 Artificial Intelligence for Legal Assistance (AILA) track have
# introduced the shared task known as Legal Document Summarization (Task 2) [2]. This task
# is further divided into two subtasks: (a) Identifying the summary-worthy sentences in legal
# judgements for creating a headnote or a summary. (b) Automatic generation of summaries from
# legal documents.
# In the text summarization literature, it has been identified that there are mainly two types of
# automatic summarization approaches—Abstractive summarization and Extractive summarization. The techniques corresponding to abstractive summarization involves novel text generation
# based summary formation which is dependent on the understanding from the input documents
# """

article = """
The highly anticipated NISAR mission — a joint venture between NASA and ISRO — could finally see its launch in June, marking a landmark partnership between the US and India.

June might witness the launch of the NISAR satellite, a collaborative effort by NASA and ISRO that represents a historic first in US-India space cooperation.

After much anticipation, the NISAR satellite developed jointly by NASA and ISRO is expected to be launched in June, symbolizing a unique US-India space alliance.

A groundbreaking NASA-ISRO mission known as NISAR may lift off in June, reflecting an unprecedented level of collaboration between India and the United States in space exploration.
"""


inputs = tokenizer([article], max_length=1024, return_tensors="pt", truncation=True)

# Generate summary
summary_ids = model.generate(
    inputs["input_ids"],
    num_beams=4,
    max_length=256,
    early_stopping=True,
    no_repeat_ngram_size=3
)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)
