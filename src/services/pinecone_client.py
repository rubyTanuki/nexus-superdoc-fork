from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.utils.math import (
    cosine_similarity,
)
from pinecone import Pinecone, IndexModel, ServerlessSpec
import os
import hashlib
import numpy as np
import time
from pydantic import BaseModel 
from numpy import ndarray
from typing import Optional
from uuid import uuid4
from langchain_core.documents import Document
import re
from langchain_openai import ChatOpenAI
from lexical.lexical_algs import extract_text_similarity_jaccard
from googledoc.googledoc import GoogleDocsEditor

from pdf_pipeline.etree import EmbedTreeNode
from io import BytesIO
'''
Need this to, create tables automatically
Initalize and Access the tables
And also do cosine similarity search efficently
In order to do the superdoc comparison alg properly

'''

class VectorDBManager(BaseModel):
    """
    The Semantic Controller. Manages Pinecone indices, handles CRUD operations 
    for document headings.
    """
    pc:Pinecone
    vs:Optional[PineconeVectorStore] = None
    index_name:Optional[str] = None
    model_config = {"arbitrary_types_allowed" : True}


    def initVectorStore(self,index_name:str,embedding:OpenAIEmbeddings):
        """Connects to an existing Pinecone index and initializes the LangChain wrapper."""
        if not self.pc.has_index(index_name): 
            raise ValueError(f"Index:{index_name}, does not exist")
        index = self.pc.Index(index_name)
        self.vs = PineconeVectorStore(index=index,embedding=embedding)
        self.index_name = index_name        
    
    
    def createIndex(self,index_name:str):
        """Provisions a new Serverless Pinecone index optimized for cosine similarity."""
        if self.pc.has_index(index_name): 
            raise ValueError(f"Index:{index_name}, already exists")
        self.pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )    
        return self.pc.Index(index_name)
    
    def generate_timestamp_id(self,course_id):
        """Generates a collision-resistant ID using the course context and epoch time."""
        timestamp = int(time.time() * 1000)
        return f"{course_id}_{timestamp}"  
    

    def _chunk_list(self, lst: list, n: int):
        """Helper generator to yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def batch_upsert_vectors(self, vectors: list[dict], course_id: str):
        """
        Safely pushes massive arrays of vectors to Pinecone by chunking them
        into blocks of 1,000 to respect API rate limits.
        """
        if not vectors:
            return
        
        index = self.pc.Index(self.index_name)
        total_batches = 0
        
        print(f"Starting batch upsert of {len(vectors)} vectors in namespace: {course_id}...")
        for chunk in self._chunk_list(vectors, 1000):
            index.upsert(vectors=chunk, namespace=course_id)
            total_batches += 1
            
        print(f"Successfully upserted {len(vectors)} vectors across {total_batches} batches.")

    def batch_delete_vectors(self, vector_ids: list[str], course_id: str):
        """
        Safely deletes massive arrays of vector IDs from Pinecone by chunking them
        into blocks of 1,000.
        """
        if not vector_ids:
            return
            
        index = self.pc.Index(self.index_name)
        total_batches = 0
        
        print(f"Starting batch deletion of {len(vector_ids)} vectors in namespace: {course_id}...")
        for chunk in self._chunk_list(vector_ids, 1000):
            index.delete(ids=chunk, namespace=course_id)
            total_batches += 1
            
        print(f"Successfully deleted {len(vector_ids)} vectors across {total_batches} batches.")





    def create_vectordb_heading(self, heading_text: str, course_id: str, superdoc_id: str) -> None:
        """
        Creates a new heading entry in vector DB with dynamically generated OpenAI embedding.
    
        Args:
            heading_text: The heading text (will be used to generate embedding)
            course_id: Course namespace for vector DB operations
            superdoc_id: Source filter for vector DB queries
    
        Raises:
            Exception: If embedding generation or creation fails
        """
        try:
            # Generate OpenAI embedding for the heading text
            embeddings = OpenAIEmbeddings()
            embedding = embeddings.embed_query(heading_text)
    
            print(f"Generated embedding for heading: '{heading_text}' (dimension: {len(embedding)})")
    
            index = self.pc.Index(self.index_name)
    
            # Create new entry with the dynamically generated embedding
            new_vector = {
                "id": self.generate_timestamp_id(course_id),
                "values": embedding,
                "metadata": {
                    "position": [heading_text],
                    "superdoc": superdoc_id,
                    "heading": heading_text
                }
            }
    
            index.upsert(vectors=[new_vector], namespace=course_id)
            print(f"Successfully created new entry with heading: '{heading_text}'")
    
        except ImportError:
            raise Exception("OpenAIEmbeddings not available. Please install langchain-openai")
        except Exception as e:
            raise Exception(f"Failed to create vector DB heading '{heading_text}': {str(e)}")
    
    def remove_vectordb_heading(self, heading: str, course_id: str, superdoc_id: str) -> int:
        """
        Removes all entries with a specific heading from vector DB.

        Args:
            heading: The heading name to be removed
            course_id: Course namespace for vector DB operations
            superdoc_id: Source filter for vector DB queries

        Returns:
            int: Number of entries deleted

        Raises:
            Exception: If deletion fails
        """
        try:
            index = self.pc.Index(self.index_name)

            # Query for entries with the specified heading
            response = index.query(
                top_k=100,
                filter={"superdoc": superdoc_id, "heading": heading},
                vector=[0] * 1536,  # Default OpenAI embedding dimension
                namespace=course_id,
                include_metadata=True
            )

            matches = response.get("matches", [])
            if matches:
                index.delete(ids=[match.id for match in matches], namespace=course_id)
                print(f"Deleted {len(matches)} entries with heading: '{heading}'")
                return len(matches)
            else:
                print(f"No existing entries found with heading: '{heading}'")
                return 0

        except Exception as e:
            raise Exception(f"Failed to remove vector DB heading '{heading}': {str(e)}")
        
        
    def replace_vectordb_heading_with_text(self, old_heading: str, new_heading_text: str, course_id: str, superdoc_id: str) -> None:
        """
        Replaces a heading in vector DB by deleting the old entry and creating a new one
        with the new heading text and its dynamically generated OpenAI embedding.

        Args:
            old_heading: The heading name to be removed
            new_heading_text: The new heading text (will be used to generate embedding)
            course_id: Course namespace for vector DB operations
            superdoc_id: Source filter for vector DB queries

        Raises:
            Exception: If embedding generation fails
        """
        try:
            # Generate OpenAI embedding for the new heading text

            embeddings = OpenAIEmbeddings()
            new_embedding = embeddings.embed_query(new_heading_text)

            print(f"Generated embedding for new heading: '{new_heading_text}' (dimension: {len(new_embedding)})")

            index = self.pc.Index(self.index_name)

            # Delete old entries
            response = index.query(
                top_k=100,
                filter={"superdoc": superdoc_id, "heading": old_heading}, #{"source": superdoc_id, "heading": old_heading}
                vector=[0] * len(new_embedding),
                namespace=course_id,
                include_metadata=True
            )

            matches = response.get("matches", [])
            if matches:
                index.delete(ids=[match.id for match in matches], namespace=course_id)
                print(f"Deleted {len(matches)} entries with old heading: '{old_heading}'")
            else:
                print(f"No existing entries found with heading: '{old_heading}'")

            # Create new entry with the dynamically generated embedding
            new_vector = {
                "id": self.generate_timestamp_id(course_id),
                "values": new_embedding,
                "metadata": {
                    "position": [new_heading_text],
                    "superdoc": superdoc_id,
                    "heading": new_heading_text
                }
            }

            index.upsert(vectors=[new_vector], namespace=course_id)
            print(f"Successfully created new entry with heading: '{new_heading_text}'")

        except ImportError:
            raise Exception("OpenAIEmbeddings not available. Please install langchain-openai")
        except Exception as e:
            raise Exception(f"Failed to replace vector DB heading: {str(e)}")
    
    
    def _generate_heading_from_sentence(self, sentence: str) -> str:
        """
        Generate a clean, short heading using an OpenAI LLM.
        Falls back to 'Basic' if input is missing or model fails.
        """
        if not sentence or not isinstance(sentence, str):
            return "Basic"      
        try:
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                max_tokens=20  # keep it short
            )       
            prompt = (
                "Create a concise heading (4–7 words maximum) based ONLY on the following "
                "sentence. The heading must:\n"
                "- be title case\n"
                "- remove unnecessary words\n"
                "- not include punctuation\n"
                "- sound like a real document section heading\n\n"
                f"Sentence: \"{sentence}\"\n\n"
                "Heading:"
            )       
            response = llm.invoke(prompt)
            heading = response.content.strip()      
            # safety check
            if not heading:
                return "Basic"      
            return heading      
        except Exception as e:
            print(f"[WARN] LLM heading generation failed: {e}")
            return "Basic"


    def get_all_headings_for_doc(self, course_id: str, superdoc_id: str) -> list[dict]:
        """
        Retrieves all heading entries associated with a specific superdoc_id 
        within a course namespace.
        """
        try:
            index = self.pc.Index(self.index_name)
    
            # We query using a dummy vector and a metadata filter.
            # Since we want ALL headings, we set top_k to a high number (e.g., 1000).
            response = index.query(
                namespace=course_id,
                vector=[0.0] * 1536,  # Dummy vector for filter-based search
                filter={
                    "superdoc": superdoc_id
                },
                top_k=1000, 
                include_metadata=True,
                include_values=True # Set to True if you need the embeddings for similarity math
            )
    
            # Extract just the metadata/values into a clean list
            headings = []
            for match in response.get("matches", []):
                headings.append({
                    "id": match.id,
                    "heading": match.metadata.get("heading"),
                    "position": match.metadata.get("position"),
                    "embedding": match.values
                })
                
            return headings
    
        except Exception as e:
            print(f"Error fetching headings: {e}")
            return []


    def remove_heading_entry(self,heading:str,course_id:str,superdoc_id:str): 
        try:
            index = self.pc.Index(self.index_name)

            # Delete old entries
            response = index.query(
                top_k=100,
                filter={"superdoc": superdoc_id, "heading": heading},
                vector=[0] * 1536,
                namespace=course_id,
                include_metadata=True
            )

            matches = response.get("matches", [])
            if matches:
                index.delete(ids=[match.id for match in matches], namespace=course_id)
                print(f"Deleted {len(matches)} entries with old heading: '{heading}'")
            else:
                print(f"No existing entries found with heading: '{heading}'")
        except Exception as e:
            raise Exception(f"Failed to delete vector DB heading: {str(e)}")                            

    def append_documents(self,e_branches:list[EmbedTreeNode],course_id:str,superdoc_id:str):
        """
        Batch-uploads semantic branches of an EmbedTree. 
        Uses the branch's 'mean_emb' (the centroid of all its children) as the vector.
        """
        if len(e_branches)==0:
            return
        index = self.pc.Index(self.index_name)
        filtered_docs = ""
        print(f"Append Documents branch check")
        #for branch in e_branches:
        #    print(branch)
        
        index.upsert(
            vectors=[{
                "id":self.generate_timestamp_id(course_id), 
                "values": branch.mean_emb,
                "metadata": {"superdoc":superdoc_id,"heading":branch.content}
                    
            } for branch in e_branches],
            namespace=course_id)           
                
if __name__ == "__main__": 
    pass

