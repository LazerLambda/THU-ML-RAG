"""Retrieval-Augmented Generation (RAG) Application

Machine Learning, 2023 Fall
Tsinghua University
Haidian District, Beijing

Philipp Koch, 2024
"""

import os
import gradio as gr
import logging
from datasets import load_dataset
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from hf_hub_ctranslate2 import GeneratorCT2fromHfHub
from typing import List


class RAG:
    def __init__(
        self,
        data_path: str,
        model_hf: str = "WhereIsAI/UAE-Large-V1",
        model_name: str = "michaelfeil/ct2fast-Llama-2-7b-hf",
        vec_store: str = "faiss_doc_idx",
        k: int = 5,
    ) -> None:
        """Initialize the RAG model.

        :param data_path: The path to the data file.
        :param model_hf: The model name of the embedding model (from huggingface).
        :param model_name: Name of the LLM model for question answering.
        :param vec_store: The path to the vectorstore.
        :param k: The number of results to return.
        """
        embeddings = HuggingFaceEmbeddings(model_name=model_hf)
        if os.path.exists(vec_store):
            vectorstore = FAISS.load_local(vec_store, embeddings)
            logging.info(f"Loaded vectorstore from local: {vec_store}")
        else:
            dataset = load_dataset("csv", data_files=data_path)
            dataset = dataset["train"].map(
                lambda e: {
                    "text": str(e["Description"])
                    + "\n\n"
                    + str(e["Keywords"])
                    + "\n\n"
                    + str(e["Body"])
                },
                remove_columns=[
                    "ID",
                    "Title",
                    "Description",
                    "Keywords",
                    "Body",
                    "Theme",
                    "Link",
                ],
            )
            self.vectorstore = FAISS.from_texts(dataset["text"], embeddings)
            self.vectorstore.save_local(vec_store)
            logging.info(f"Saved vectorstore to local: {vec_store}")

        self.retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )
        self.model = GeneratorCT2fromHfHub(
            model_name_or_path=model_name, device="cpu", compute_type="int8"
        )
        logging.info(f"Loaded model: {model_name}")

    def __call__(self, query: str) -> str:
        """Call the RAG model with a query and return the answer.

        :param query: The query to be answered.
        :returns: The answer to the query given the context.
        """
        logging.info(f"Query: {query}")
        results: List = self.retriever.get_relevant_documents(query)
        logging.info(f"Results: {results}")
        context = "\n".join([e.page_content for e in results])
        query = f"Question: {query}. Context {context}. Answer of the question, given the context: "
        logging.info(f"Formated query: {query}")
        self.model.generate(text=[query], include_prompt_in_result=False)
        return results[0].page_content


if __name__ == "__main__":
    """Run the Application."""
    rag = RAG("data/news.csv")
    interface = gr.Interface(fn=rag, inputs="text", outputs="text")
    interface.launch()
