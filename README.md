# 🌌 Astral & uv Setup Guide

Download Astral UV from terminal:

**Windows:**
```powershell```
> powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

**Linux**:
```Terminal```
>curl -LsSf https://astral.sh/uv/install.sh | sh

After Clonning Do type Command:
>uv sync

<!-- --- -->

# About Previous Used DataSet: Scisumm_Net Dataset

### We have Uploaded the formed Csv (clean_dataset.csv) and Raw Dataset (Raw Dataset)

## Issue in Dataset
### According to Dataset_Documentation.txt inisde Raw Dataset we formed our Csv file. Where it was  Mention to take Abstract Part of Xml file, Clean Citing Sentences and Authority scores from citing_sentences_annotated.json as Input and to evaluate on Annotated Gold Summary

### So We observed on Training T5 model on this Dataset it was giving abstractive Summary. So we Analysed the Dataset where we found the total input text and output text , more precisely the abstract and Gold Summary were closely same.

### If you want to download Dataset in folder format Link:
>https://cs.stanford.edu/~myasu/projects/scisumm_net/

### So Changed the dataset to CNN/Dailymail Dataset

<!-- --- -->

# Who To Contact:

### Devansh Manoj Kesan:
 - https://github.com/Devansh-Kesan
 - 142201035@smail.iitpkd.ac.in

### Yash Anil Aher
 - https://github.com/yashanilaher
 - 142201035@smail.iitpkd.ac.in

<!-- --- -->

# Credits:

### Swapnil Sir (Mentor)






