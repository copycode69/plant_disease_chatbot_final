import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyB8fZVq11Pe6kLRJi0q4FhCsTmZz_xHnrM" 
import csv
from io import StringIO
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

PERSIST_DIR = "./backend/chroma_store"

def create_or_load_vectorstore():
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not os.path.exists(PERSIST_DIR):
        loader = CSVLoader(
            file_path="backend/products.csv",
            csv_args={
                "delimiter": ",",
                "fieldnames": ["disease", "product_name", "product_link", "description", "image_url"]
            },
            metadata_columns=["disease", "product_name", "product_link", "description", "image_url"]
        )
        docs = loader.load()
        vectorstore = Chroma.from_documents(docs, embedding, persist_directory=PERSIST_DIR)
        vectorstore.persist()
    else:
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
    return vectorstore

# MODIFIED: Only retrieve 1 most relevant product
retriever = create_or_load_vectorstore().as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 1}  # Changed from 3 to 1
)

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(model="gemini-2.5-pro"), 
    retriever=retriever,
    return_source_documents=True 
)

def get_bot_reply(query):
    result = qa_chain.invoke({"query": query})
    answer = result["result"]
    source_docs = result.get("source_documents", [])
    
    # Initialize response with just the bot's answer
    response = f"<div class='bot-response'>{answer}</div>"
    
    if source_docs:
        try:
            doc = source_docs[0]  # Take only the first document
            
            # Get product details from metadata
            metadata = getattr(doc, 'metadata', {})
            product_name = metadata.get('product_name', '').strip()
            product_link = metadata.get('product_link', '').strip()
            description = metadata.get('description', '').strip()
            image_url = metadata.get('image_url', '').strip()
            
            # Fallback to content parsing if metadata is incomplete
            if not all([product_name, product_link, description]):
                content = doc.page_content.split('\n')
                product_data = {}
                for line in content:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        product_data[key.strip()] = value.strip()
                
                product_name = product_data.get('product_name', '').strip()
                product_link = product_data.get('product_link', '').strip()
                description = product_data.get('description', '').strip()
                image_url = product_data.get('image_url', '').strip()
            
            if product_name and product_name.lower() != 'product_name':
                # Set default image if empty
                if not image_url:
                    image_url = 'https://via.placeholder.com/200x150?text=No+Image'
                
                # Replace the entire response with just the formatted product card
                response = f"""
                <div class="recommended-product">
                    <div class="product-header">
                        <h3>Recommended Treatment</h3>
                    </div>
                    <div class="product-card">
                        <div class="product-image">
                            <img src="{image_url}" 
                                 alt="{product_name}"
                                 onerror="this.src='https://via.placeholder.com/200x150?text=Image+Not+Found'">
                        </div>
                        <div class="product-details">
                            <h4>{product_name}</h4>
                            <p class="description">{description}</p>
                            <a href="{product_link}" 
                               target="_blank" 
                               class="buy-button">
                               View Product
                            </a>
                        </div>
                    </div>
                </div>
                """
                
        except Exception as e:
            print(f"Error processing product: {e}")
    
    return response