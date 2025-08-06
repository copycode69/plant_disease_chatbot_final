import os
import csv
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Set your API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyB8fZVq11Pe6kLRJi0q4FhCsTmZz_xHnrM"

PERSIST_DIR = "./backend/chroma_store"
CSV_PATH = "backend/products.csv"

# Create or load vectorstore
def create_or_load_vectorstore():
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not os.path.exists(PERSIST_DIR):
        loader = CSVLoader(
            file_path=CSV_PATH,
            csv_args={"delimiter": ","},
            metadata_columns=["disease", "product_name", "product_link", "description", "image_url"]
        )
        docs = loader.load()
        vectorstore = Chroma.from_documents(docs, embedding, persist_directory=PERSIST_DIR)
        vectorstore.persist()
    else:
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
    return vectorstore

# Build retriever
retriever = create_or_load_vectorstore().as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 1, "score_threshold": 0.6}
)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatGoogleGenerativeAI(model="gemini-2.5-pro"), 
    retriever=retriever,
    return_source_documents=True 
)

# Optional fallback: Exact match lookup
def get_product_by_exact_disease(disease):
    with open(CSV_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["disease"].strip().lower() == disease.strip().lower():
                return row
    return None

# Main response function
def get_bot_reply(query):
    result = qa_chain.invoke({"query": query})
    answer = result["result"]
    source_docs = result.get("source_documents", [])

    response = f"<div class='bot-response'>{answer}</div>"

    def build_product_card(name, link, desc, img):
        if not img:
            img = 'https://via.placeholder.com/200x150?text=No+Image'
        return f"""
        <div class="recommended-product">
            <div class="product-header">
                <h3>Recommended Treatment</h3>
            </div>
            <div class="product-card">
                <div class="product-image">
                    <img src="{img}" 
                         alt="{name}"
                         onerror="this.src='https://via.placeholder.com/200x150?text=Image+Not+Found'">
                </div>
                <div class="product-details">
                    <h4>{name}</h4>
                    <p class="description">{desc}</p>
                    <a href="{link}" 
                       target="_blank" 
                       class="buy-button">
                       View Product
                    </a>
                </div>
            </div>
        </div>
        """

    # Handle vectorstore match
    if source_docs:
        try:
            doc = source_docs[0]
            metadata = getattr(doc, 'metadata', {})
            product_name = metadata.get('product_name', '').strip()
            product_link = metadata.get('product_link', '').strip()
            description = metadata.get('description', '').strip()
            image_url = metadata.get('image_url', '').strip()

            if product_name and product_link and description:
                response = build_product_card(product_name, product_link, description, image_url)
            else:
                content = doc.page_content.split('\n')
                data = {}
                for line in content:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        data[key.strip()] = value.strip()
                response = build_product_card(
                    data.get("product_name", ""),
                    data.get("product_link", ""),
                    data.get("description", ""),
                    data.get("image_url", "")
                )
        except Exception as e:
            print(f"Error processing product: {e}")

    # Fallback to exact match
    elif not source_docs:
        fallback = get_product_by_exact_disease(query)
        if fallback:
            response = build_product_card(
                fallback["product_name"],
                fallback["product_link"],
                fallback["description"],
                fallback.get("image_url", "")
            )

    return response
