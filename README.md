# RAC-Bot for News

## Idea
This app is based on the langchain framework, which provides a wrapper to the Faicebook AI Similarity Search (FAISS) and huggingface'str
transformer library. To provide the model with up-to-date information, the query is used to request matching documents from a vector-store.
The best matching documents are used to provide further context for the model to produce a correct answer to the query.
The vector-storage is built upon [Universal AnglE Embedding](https://arxiv.org/abs/2309.12871) as it was upon the most successful models on the
[MTEB-leaderboard](https://huggingface.co/spaces/mteb/leaderboard).
To produce the output, a powerful model is required, that is both capable of dealing with large context length and is performant enough to produce
high-quality text. Thus, we decided to apply a quantizized version of [Llama 2](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) with seven billion parameters ```[ct2fast-Llama-2-7b-hf](https://huggingface.co/michaelfeil/ct2fast-Llama-2-7b-hf)'''. Quantization is a technique, in which
the precision of the floating point representation is reduced to speed up inference performance and overall model size. Furthermore, this implementation
also utilizes the speed of the C++ language and can be used with hardware accelartion.

## Prerequisites
- 8GB of free disk space
- Optional: Create virtual environment
- Run: `pip install -r requirements.txt`

## Run App
Run `python src/main.py` and ask a question.

## Limitations
- Without appropriate hardware-acceleration, the app performs slow. A GPU is recommended