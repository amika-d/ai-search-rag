import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from typing import List, Dict, Optional
import os


class WeaviateDB:
    def __init__(self, url: str= "http://localhost:8080", api_key: Optional[str] = None, collection_name: str = "SkincareProducts"):
        
        """
        Initialize Weaviate client
        
        Args:
            url: Weaviate instance URL
            api_key: Optional API key for cloud instances
            collection_name: Name of the collection to use
        """

        self.collection_name = collection_name
        
        self.url = url
        
        self.client = weaviate.connect_to_local(
            host="localhost",
            port=8080,
            grpc_port=50051
            
        )
         # Verify connection is ready
        if not self.client.is_ready():
            raise ConnectionError("Weaviate not ready at localhost:8080. Check if server is running.")
        
        print(f"✅ Connected to local Weaviate at {url}")
        print(f"📁 Default collection: {collection_name}")
        
        
    def create_schema(self):
        try:
            self.client.collections.create(
                name=self.collection_name,
                vector_config=Configure.Vectorizer.text2vec_transformers(),
                 properties=[
                    Property(name="product_id", data_type=DataType.TEXT),
                    Property(name="collection_id", data_type=DataType.TEXT),
                    Property(name="name", data_type=DataType.TEXT),
                    Property(name="description", data_type=DataType.TEXT),
                    Property(name="price", data_type=DataType.NUMBER),
                    Property(name="cost_price", data_type=DataType.NUMBER),
                    Property(name="profit_margin", data_type=DataType.NUMBER),
                    Property(name="inventory", data_type=DataType.INT),
                    Property(name="rating", data_type=DataType.NUMBER),
                    Property(name="ingredients", data_type=DataType.TEXT_ARRAY),
                    Property(name="concerns_addressed", data_type=DataType.TEXT_ARRAY),
                    Property(name="texture", data_type=DataType.TEXT),
                    Property(name="image", data_type=DataType.TEXT),
                ]

    def __enter__(self):
        return self
    
    
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.client.close()
        print("🔌 Weaviate connection closed")



