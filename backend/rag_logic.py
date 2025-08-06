import os
import csv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Set API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyB8fZVq11Pe6kLRJi0q4FhCsTmZz_xHnrM"

PERSIST_DIR = "./backend/chroma_store"
CSV_PATH = "backend/products.csv"
DEFAULT_IMAGE = 'https://via.placeholder.com/200x150?text=No+Image'

# Custom CSV loader with enhanced content creation
def custom_csv_loader(file_path):
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Create robust content for embeddings including symptoms
            page_content = (
                f"Disease: {row['disease']} | "
                f"Symptoms: {row['disease'].replace(' ', ', ')} | "
                f"Product: {row['product_name']} | "
                f"Description: {row['description']}"
            )
            metadata = {
                "disease": row["disease"],
                "product_name": row["product_name"],
                "product_link": row["product_link"],
                "description": row["description"],
                "image_url": row.get("image_url", "")
            }
            docs.append(Document(page_content=page_content, metadata=metadata))
    return docs

# Vectorstore management with diagnostics
def create_or_load_vectorstore():
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not os.path.exists(PERSIST_DIR):
        os.makedirs(os.path.dirname(PERSIST_DIR), exist_ok=True)
        docs = custom_csv_loader(CSV_PATH)
        vectorstore = Chroma.from_documents(
            docs, 
            embedding, 
            persist_directory=PERSIST_DIR
        )
        vectorstore.persist()
        print(f"Created new vectorstore with {len(docs)} products")
    else:
        vectorstore = Chroma(
            persist_directory=PERSIST_DIR, 
            embedding_function=embedding
        )
        print(f"Loaded existing vectorstore with {vectorstore._collection.count()} products")
    return vectorstore

# Build retriever with optimized thresholds
retriever = create_or_load_vectorstore().as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3, "score_threshold": 0.5}  # Broader search parameters
)

# QA Chain configuration
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.2),
    retriever=retriever,
    return_source_documents=True
)

# Enhanced disease matching
def find_products_for_disease(disease_query):
    disease_lower = disease_query.strip().lower()
    matches = []
    
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Check multiple matching strategies
            row_disease = row["disease"].strip().lower()
            if (disease_lower == row_disease or
                disease_lower in row_disease or
                any(word in row_disease for word in disease_lower.split()) or
                disease_lower in row["description"].lower()):
                matches.append(row)
                
    return matches

# HTML product card generator
def build_product_card(name, link, desc, img):
    if not img or img == "NA":
        img = DEFAULT_IMAGE
    return f"""
    <div class="recommended-product">
        <div class="product-header">
            <h3>Recommended Treatment</h3>
        </div>
        <div class="product-card">
            <div class="product-image">
                <img src="{img}" alt="{name}" onerror="this.src='{DEFAULT_IMAGE}'">
            </div>
            <div class="product-details">
                <h4>{name}</h4>
                <p class="description">{desc}</p>
                <a href="{link}" target="_blank" class="buy-button">View Product</a>
            </div>
        </div>
    </div>
    """

# Bot response handler
def get_bot_reply(query):
    # First try vector-based matching
    try:
        result = qa_chain.invoke({"query": f"Plant disease: {query}"})
        if source_docs := result.get("source_documents", []):
            for doc in source_docs:
                metadata = doc.metadata
                if all(key in metadata for key in ["product_name", "product_link", "description"]):
                    return build_product_card(
                        metadata["product_name"],
                        metadata["product_link"],
                        metadata["description"],
                        metadata.get("image_url", "")
                    )
    except Exception as e:
        print(f"Vector search error: {str(e)}")

    # Then try direct CSV matching
    if products := find_products_for_disease(query):
        product = products[0]  # Get best match
        return build_product_card(
            product["product_name"],
            product["product_link"],
            product["description"],
            product.get("image_url", "")
        )

    # Final fallback
    return "<div class='bot-response'>Sorry, no specific treatment found. Try general fungicide for fungal diseases.</div>"