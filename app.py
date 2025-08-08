import os
import requests
import tempfile
import streamlit as st
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------------------------------
# Constants
# ----------------------------------
PDF_DOWNLOAD_DIR = "downloaded_pdfs"
if not os.path.exists(PDF_DOWNLOAD_DIR):
    os.makedirs(PDF_DOWNLOAD_DIR)

PREDEFINED_PDF_LINKS = {
    "Knowledge": ["https://example.com/knowledge.pdf"],
    "SD": ["https://example.com/sd.pdf"],
    "EUC": [
        "https://example.com/euc_page1.html",
        "https://example.com/euc_page2.html"
    ]
}

# ----------------------------------
# Helper Functions
# ----------------------------------
def load_and_split_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(docs)
    except Exception as e:
        st.error(f"Error processing PDF {file_path}: {e}")
        return []

def load_and_split_webpage(url):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(docs)
    except Exception as e:
        st.error(f"Error processing webpage {url}: {e}")
        return []

def download_pdf(url, output_path):
    try:
        response = requests.get(url, stream=True, timeout=60)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            st.success(f"Downloaded PDF: {output_path}")
            return True
        else:
            st.error(f"Failed to download PDF from {url} (status {response.status_code})")
    except Exception as e:
        st.error(f"Error downloading PDF from {url}: {e}")
    return False

# ----------------------------------
# NEW FUNCTIONS (parallel + linked PDF processing)
# ----------------------------------
def fetch_and_process_linked_pdfs(base_url):
    """
    Detect and download linked PDFs from a webpage, then process them.
    Returns list of document chunks.
    """
    docs = []
    try:
        html = requests.get(base_url, timeout=30).text
        soup = BeautifulSoup(html, "html.parser")
        pdf_links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.lower().endswith(".pdf"):
                from urllib.parse import urljoin
                href = urljoin(base_url, href)
                pdf_links.append(href)

        if pdf_links:
            st.success(f"Found {len(pdf_links)} linked PDF(s) on {base_url}")
            for pdf_url in pdf_links:
                file_name = os.path.basename(urlparse(pdf_url).path)
                output_path = os.path.join(PDF_DOWNLOAD_DIR, file_name)
                if download_pdf(pdf_url, output_path):
                    docs.extend(load_and_split_pdf(output_path))
        else:
            st.info(f"No linked PDFs found in {base_url}")

    except Exception as e:
        st.error(f"Error fetching linked PDFs from {base_url}: {e}")
    return docs

def load_webpage_and_pdfs_parallel(urls):
    """
    Load webpage content AND any linked PDFs in parallel for multiple URLs.
    Returns list of all document chunks.
    """
    all_docs = []

    def process_url(url):
        docs_for_url = []
        try:
            # First, get webpage content
            docs_for_url.extend(load_and_split_webpage(url))
            # Then, get linked PDFs
            docs_for_url.extend(fetch_and_process_linked_pdfs(url))
        except Exception as e:
            st.error(f"Error processing {url}: {e}")
        return docs_for_url

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(process_url, url): url for url in urls}
        for future in as_completed(future_to_url):
            try:
                result_docs = future.result()
                all_docs.extend(result_docs)
            except Exception as e:
                st.error(f"Error in parallel processing: {e}")

    return all_docs

# ----------------------------------
# Main Ingestion Function
# ----------------------------------
def ingest_selected_source(selected_source):
    all_docs = []

    if selected_source in ["Knowledge", "SD"]:
        for pdf_url in PREDEFINED_PDF_LINKS[selected_source]:
            file_name = os.path.basename(urlparse(pdf_url).path)
            output_path = os.path.join(PDF_DOWNLOAD_DIR, file_name)
            if download_pdf(pdf_url, output_path):
                docs = load_and_split_pdf(output_path)
                all_docs.extend(docs)

    elif selected_source == "EUC":
        # ✅ NEW: parallel ingestion for EUC pages + linked PDFs
        docs = load_webpage_and_pdfs_parallel(PREDEFINED_PDF_LINKS["EUC"])
        all_docs.extend(docs)

    else:
        st.error("Invalid source selected.")

    st.success(f"Total documents ingested: {len(all_docs)}")
    return all_docs

# ----------------------------------
# Streamlit UI
# ----------------------------------
st.title("📄 Document Ingestion Tool")

source_option = st.selectbox("Select Data Source", ["Knowledge", "SD", "EUC"])
if st.button("Ingest Selected Source"):
    with st.spinner("Processing..."):
        ingested_docs = ingest_selected_source(source_option)
        st.write(f"✅ Ingested {len(ingested_docs)} chunks.")
