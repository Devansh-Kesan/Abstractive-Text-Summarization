# 🌌 Astral & uv Setup Guide
Installation
Install Astral UV from the terminal:

**Windows**:

```powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"```

**Linux**:

```curl -LsSf https://astral.sh/uv/install.sh | sh```

After cloning the repository, 
**run**:

```uv sync```

make sure you are in Abstractive-Text-Summarization folder while doing **uv sync**


# About the Previous Dataset: Scisumm_Net

We used the Scisumm_Net Dataset to create clean_dataset.csv from the raw dataset, as per the instructions in Dataset_Documentation.txt. The dataset was processed by extracting the abstract from XML files, cleaning citing sentences, and including authority scores from citing_sentences_annotated.json as input, with evaluation on the annotated gold summary.

### Issue with the Dataset
When training a T5 model on this dataset, it produced abstractive summaries. Upon analysis, we found that the input (abstract) and output (gold summary) texts were very similar, leading to suboptimal summarization performance.

**Dataset Download**

You can download the Scisumm_Net dataset in folder format here:
[Scisumm_Net Dataset](https://cs.stanford.edu/~myasu/projects/scisumm_net/)

# Switched to CNN/DailyMail Dataset
Due to the issues with Scisumm_Net, we switched to the CNN/DailyMail Dataset for better summarization performance.

**Running NLP Text Summarization**

Ensure uv is installed on your system using the installation commands provided above.
We have uploaded our trained model to Google Drive. To run the summarization project, you can either train the model from scratch or use our pretrained model.

**Pretrained Model Download**: [Trained model](https://drive.google.com/drive/u/0/folders/1XftoDr4d61XltKnR43yhgy6nn9lIwKNh)

If using the pretrained model, download the output folder and place it in the same directory as pyproject.toml.

Steps

Step 1: Clone the Repository
Clone the project repository to your local machine.

Step 2: (Optional): Train the Model
If you want to train the model from scratch, run:

```uv run python train.py --model_name facebook/bart-base --use_coverage --output_dir ./output```

Requirements:

CUDA-enabled GPU with at least 15 GB of RAM.
Ensure CUDA is installed on your system.

Step 3: Evaluate the Pretrained Model
To evaluate the pretrained model, run:

```uv run python evaluate.py --enhanced_model_path ./output/checkpoint-epoch-5 --bart_base_model_name facebook/bart-base --t5_model_name t5-base --sample_size 100 --human_eval_size 10```

you can see the detailed evaluation of the models in the comparison_results_4.txt file, also we have made the human_evaluation_samples.txt file for the hunam_evaluation, it consists of 10 summaries, where humans have to give ratings between 1-5 for each summary based on the following criteria's :

1. coherenace
2. relevance
3. Overall quality

Inference: Run inference with the following commands:

BART-Base Model:

```uv run inference_bart_base.py```

Enhanced BART-Base Model:

```uv run inference_bart_enhanced.py```

Custom Summaries: To summarize a different article:

Open the respective inference file (inference_bart_base.py or inference_bart_enhanced.py).
Replace the article content with the text you want to summarize.
Run the inference command for the respective model.


# Contact Information

**Devansh Manoj Kesan**
 - GitHub: Devansh-Kesan
 - Email: 142201017mail.iitpkd.ac.in

**Yash Anil Aher**
 - GitHub: yashanilaher
 - Email: 142201035@smail.iitpkd.ac.in

**Rajkumar Naik**
 - GitHub: Rohit4459
 - Email: 142201029@smail.iitpkd.ac.in


# Credits

**Mentor: Swapnil Sir**