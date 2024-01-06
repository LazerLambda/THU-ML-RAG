# RAG-Bot for News
![THU](./thubell.png)
## Idea
This app is based on the LangChain framework, which provides a wrapper for the Facebook AI Similarity Search (FAISS) and Hugging Face's Transformer library. To provide the model with up-to-date information, the query is used to request matching documents from a vector store. The best matching documents are used to provide further context for the model to produce a correct answer to the query. The vector storage is built upon [Universal AnglE Embedding](https://arxiv.org/abs/2309.12871) as it was one of the most successful models on the [MTEB-leaderboard](https://huggingface.co/spaces/mteb/leaderboard). To produce the output, a powerful model is required, one that is both capable of dealing with large context lengths and is performant enough to produce high-quality text. Thus, we decided to apply a quantized version of [Llama 2](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) with seven billion parameters [ct2fast−Llama−2−7b−hf](https://huggingface.co/michaelfeil/ct2fast−Llama−2−7b−hf). Quantization is a technique in which the precision of the floating-point representation is reduced to speed up inference performance and reduce overall model size. Furthermore, this implementation also utilizes the speed of the C++ language and can be used with hardware acceleration.

## Prerequisites
- 8GB of free disk space
- 20GB of available RAM
- Optional: Create a virtual environment
- Run: pip install -r requirements.txt

## Run App
Run `python src/main.py`, visit http://127.0.0.1:7860 and ask a question in the online interface.

## Limitations
 - Without appropriate hardware acceleration, the app performs slowly. A GPU is recommended.