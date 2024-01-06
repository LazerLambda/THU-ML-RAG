import pandas as pd
from datasets import load_dataset
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from hf_hub_ctranslate2 import GeneratorCT2fromHfHub
from typing import List



class RAG:
    def __init__(self, data_path: str, model_hf: str="WhereIsAI/UAE-Large-V1", model_name: str = "michaelfeil/ct2fast-Llama-2-7b-hf", vec_store: str = "faiss_doc_idx", k:int=1) -> None:
        dataset = load_dataset('csv', data_files=data_path)
        dataset = dataset['train'].map(lambda e: {'text': str(e['Description']) + '\n\n' + str(e['Keywords']) + '\n\n' + str(e['Body'])}, remove_columns=['ID', 'Title', 'Description', 'Keywords', 'Body', 'Theme', 'Link'])
        embeddings = HuggingFaceEmbeddings(model_name=model_hf)
        self.vectorstore = FAISS.from_texts(dataset['text'][0:5], embeddings)
        self.vectorstore.save_local(vec_store)
        self.retriever = self.vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
        self.model = GeneratorCT2fromHfHub(
            model_name_or_path=model_name,
            compute_type="int8_float16",
            max_length=64,
        )

    def __call__(self, query: str) -> str:
        results: List = self.retriever(query)
        context = "\n".join([e.page_content for e in results])
        query = f"Question: {query}. Context {context}. Answer of the question, given the context: "
        self.model.generate(
            text=[query],
            include_prompt_in_result=False
        )
        return results[0]

if __name__ == '__main__':
    rag = RAG("data/train.csv")
    query= input("Please enter your question: ")
    print(rag(query))