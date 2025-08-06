import os
import csv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.schema import Document

# Set API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyB8fZVq11Pe6kLRJi0q4FhCsTmZz_xHnrM"

PERSIST_DIR = "./backend/chroma_store"
CSV_PATH = "backend/products.csv"
DEFAULT_IMAGE = 'https://via.placeholder.com/200x150?text=No+Image'

# Fixed CSV loader with unique embeddings for each disease
def custom_csv_loader(file_path):
    docs = []
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Create disease-specific embedding content
            page_content = (
                f"Treatment for {row['disease']}: "
                f"Use {row['product_name']} which is {row['description']}. "
                f"It specifically targets {row['disease']} pathogens."
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

# Vectorstore management
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

# Preload disease-product mapping
def load_disease_map():
    disease_map = {}
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            disease_map[row['disease'].lower()] = row
    return disease_map

DISEASE_MAP = load_disease_map()

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

# Bot response handler - SIMPLIFIED and FIXED
def get_bot_reply(query):
    normalized_query = query.lower().strip()
    
    # 1. First try exact match from preloaded CSV data
    if normalized_query in DISEASE_MAP:
        product = DISEASE_MAP[normalized_query]
        return build_product_card(
            product["product_name"],
            product["product_link"],
            product["description"],
            product.get("image_url", "")
        )
    
    # 2. Try semantic search
    try:
        vectorstore = create_or_load_vectorstore()
        results = vectorstore.similarity_search(query, k=1)
        if results:
            product = results[0].metadata
            return build_product_card(
                product["product_name"],
                product["product_link"],
                product["description"],
                product.get("image_url", "")
            )
    except Exception as e:
        print(f"Semantic search error: {str(e)}")
    
    # 3. Try keyword matching in disease names
    for disease_name in DISEASE_MAP.keys():
        if normalized_query in disease_name:
            product = DISEASE_MAP[disease_name]
            return build_product_card(
                product["product_name"],
                product["product_link"],
                product["description"],
                product.get("image_url", "")
            )
    
    # Final fallback
    return "<div class='bot-response'>Sorry, no specific treatment found. Consult a plant specialist.</div>"