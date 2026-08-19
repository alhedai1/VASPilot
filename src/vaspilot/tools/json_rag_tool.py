import json
from typing import Any, Dict, Optional, Type, Set
from pathlib import Path

from crewai.tools.base_tool import BaseTool
from pydantic import BaseModel, Field, model_validator
import chromadb
from chromadb import EmbeddingFunction, Collection, ClientAPI
import uuid


class JsonApproxSearchInput(BaseModel):
    """Input schema for JsonRagTool"""
    query: str = Field(description="text to query the docs")
    top_k: int = Field(default=10, description="number of results to return. The default value is 10.")


class JsonApproxSearch(BaseTool):
    """
    Use RAG technology to search for relevant information from the JSON knowledge base.
    JSON format: {tag_name: {default_value, description, detailed_description, related_tags}}
    """
    
    name: str = "json_approx_search_tool"
    description: str = (
        "Use RAG technology to search for relevant information from the JSON knowledge base."
        "Can search for the most relevant configuration items and return short details."
    )
    args_schema: Type[BaseModel] = JsonApproxSearchInput
    
    # embedding_function field
    embedding_function: EmbeddingFunction = Field(description="Embedding function for ChromaDB")
    source_files: Set[str] = Field(default_factory=set)
    # Declared instance attribute types with default values
    client: Optional[ClientAPI] = Field(default=None)
    collection: Optional[Collection] = Field(default=None)
    chroma_db_path: str = Field(default="")

    @model_validator(mode='after')
    def initialize_components(self) -> 'JsonApproxSearch':
        """Pydantic v2-style initializer"""
        # Initialize the ChromaDB client
        if self.chroma_db_path:
            self.client = chromadb.PersistentClient(path=self.chroma_db_path)
        else:
            self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(
            name="json_knowledge_base",
            metadata={"hnsw:space": "cosine"},
            embedding_function=self.embedding_function
        )
        
        # Track files that have already been added
        if self.collection.count() == 0:
            for files in self.source_files:
                self.add(files)

        return self

    def add(self, json_file_path: str) -> None:
        """
        Add a JSON file to the knowledge base

        Args:
            json_file_path: Path to the JSON file
        """
        json_path = Path(json_file_path)

        # Check whether the file exists
        if not json_path.exists():
            raise FileNotFoundError(f"File does not exist: {json_file_path}")

        # Read the JSON file
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Prepare data for embedding
        documents = []
        metadatas = []
        ids = []

        for tag_name, tag_info in data.items():
            # Build the text to embed
            text_parts = [
                f"Tag name: {tag_name}",
                f"Default value: {tag_info.get('default_value', '')}",
                f"Description: {tag_info.get('description', '')}",
                f"Detailed description: {tag_info.get('detailed_description', '')}",
                f"Related tags: {', '.join(tag_info.get('related_tags', []))}"
            ]
            document_text = "\n".join(text_parts)

            # Prepare metadata
            metadata = {
                "tag_name": tag_name,
                "default_value": tag_info.get('default_value', ''),
                "description": tag_info.get('description', ''),
                "source_file": str(json_path.absolute()),
                "related_tags": json.dumps(tag_info.get('related_tags', [])),
            }
            
            documents.append(document_text)
            metadatas.append(metadata)
            ids.append(f"{tag_name}_{str(uuid.uuid4())[:8]}")

        # Add to the vector database
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Successfully added {len(documents)} tags to the knowledge base: {json_file_path}")

    def _run(self, query: str, top_k: int = 10) -> dict:
        """
        Execute a RAG query

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            Dictionary of query results
        """
        # Check whether the knowledge base is empty
        if self.collection.count() == 0:
            return "The knowledge base is empty. Please add JSON files first using the add() method."

        # Execute the query
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        # Format the results
        if not results['documents'][0]:
            return "No relevant tag information found."

        response = {}
        
        for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0]), 1):
            tag_name = metadata['tag_name']
            description = metadata['description']
            
            response[tag_name] = {
                "description": description,
                "default_value": metadata['default_value'],
                "related_tags": metadata['related_tags'],
                "score": f"{1 - distance:.3f}"
            }
        
        return response

class JsonStrictSearchInput(BaseModel):
    page_name: str = Field(description="the exact page_name to query details.")

class JsonStrictSearch(BaseTool):
    """
    Tool to query detailed descriptions
    """
    name: str = "json_strict_search_tool"
    description: str = "Tool to query detailed descriptions. The detailed description is long, only query the most important tags."
    args_schema: Type[BaseModel] = JsonStrictSearchInput
    source_files: Set[str] = Field(default_factory=set)
    # Declared instance attribute type
    data_dict: Dict[str, Any] = Field(default_factory=dict)

    def add(self, json_file_path: str) -> None:
        """
        Add a JSON file to the knowledge base

        Args:
            json_file_path: Path to the JSON file
        """
        json_path = Path(json_file_path)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for tag_name, tag_info in data.items():
            self.data_dict[tag_name] = tag_info

    @model_validator(mode='after')
    def initialize_components(self) -> 'JsonStrictSearch':
        """Pydantic v2-style initializer"""
        # Track files that have already been added
        for files in self.source_files:
            self.add(files)
            
        return self

    def _run(self, page_name: str) -> str:
        if self.data_dict.get(page_name, None) is not None:
            return self.data_dict[page_name]
        else:
            return f"No detailed description found for page_name: {page_name}"