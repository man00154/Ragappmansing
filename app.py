import streamlit as st
import os
import requests
import tempfile
import base64
from urllib.parse import urlparse

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Config ---
PDF_DOWNLOAD_DIR = "downloaded_pdfs"
FAISS_INDEX_DIR = "faiss_index"
os.makedirs(PDF_DOWNLOAD_DIR, exist_ok=True)

# --- Links ---
PREDEFINED_PDF_LINKS = {
    "Dell": [
        "https://i.dell.com/sites/csdocuments/Product_Docs/en/Dell-EMC-PowerEdge-Rack-Servers-Quick-Reference-Guide.pdf",
        "https://www.delltechnologies.com/asset/en-us/products/servers/technical-support/poweredge-r660xs-technical-guide.pdf",
        "https://i.dell.com/sites/csdocuments/shared-content_data-sheets_documents/en/aa/poweredge_r740_r740xd_technical_guide.pdf",
        "https://dl.dell.com/topicspdf/openmanage-server-administrator-v95_users-guide_en-us.pdf",
        "https://dl.dell.com/manuals/common/dellemc-server-config-profile-refguide.pdf",
    ],
    "IBM": [
        "https://www.redbooks.ibm.com/redbooks/pdfs/sg248513.pdf",
        "https://www.ibm.com/docs/SSLVMB_28.0.0/pdf/IBM_SPSS_Statistics_Server_Administrator_Guide.pdf",
        "https://public.dhe.ibm.com/software/webserver/appserv/library/v60/ihs_60.pdf",
        "https://www.ibm.com/docs/en/storage-protect/8.1.25?topic=pdf-files",
    ],
    "Cisco": [
        "https://www.cisco.com/c/dam/global/shared/assets/pdf/cisco_enterprise_campus_infrastructure_design_guide.pdf",
        "https://www.cisco.com/c/dam/en_us/about/ciscoitatwork/downloads/ciscoitatwork/pdf/Cisco_IT_Wireless_LAN_Design_Guide.pdf",
        "https://www.cisco.com/c/dam/en_us/about/ciscoitatwork/downloads/ciscoitatwork/pdf/Cisco_IT_IP_Addressing_Best_Practices.pdf",
        "https://www.cisco.com/c/en/us/td/docs/net_mgmt/network_registrar/7-2/user/guide/cnr72book.pdf",
    ],
    "Juniper": [
        "https://www.juniper.net/documentation/us/en/software/junos/junos-overview/junos-overview.pdf",
        "https://archive.org/download/junos-srxsme/JunOS%20SRX%20Documentation%20Set/network-management.pdf",
        "https://csrc.nist.gov/CSRC/media/projects/cryptographic-module-validation-program/documents/security-policies/140sp3779.pdf",
    ],
    "Fortinet (FortiGate)": [
        "https://fortinetweb.s3.amazonaws.com/docs.fortinet.com/v2/attachments/b94274f8-1a11-11e9-9685-f8bc1258b856/FortiOS-5.6-Firewall.pdf",
        "https://docs.fortinet.com/document/fortiweb/6.0.7/administration-guide-pdf",
        "https://www.andovercg.com/datasheets/fortigate-fortinet-200.pdf",
        "https://www.commoncriteriaportal.org/files/epfiles/Fortinet%20FortiGate_EAL4_ST_V1.5.pdf",
    ],
    "EUC": [
        "https://www.dell.com/en-us/lp/dt/end-user-computing",
        "https://www.nutanix.com/solutions/end-user-computing",
        "https://eucscore.com/docs/tools.html",
        "https://apparity.com/euc-resources/spreadsheet-euc-documents/",
    ],
}

# --- Helper Functions ---
def download_pdf(url, output_path):
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success(f"Downloaded: {os.path.basename(output_path)}")
        return True
    except Exception as e:
        st.error(f"Error downloading {url}: {e}")
        return False

def load_and_split_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(documents)
        st.success(f"Processed {len(texts)} chunks from {os.path.basename(file_path)}")
        return texts
    except Exception as e:
        st.error(f"Error processing PDF {os.path.basename(file_path)}: {e}")
        return []

def load_and_split_webpage(url):
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(documents)
        st.success(f"Processed {len(texts)} chunks from webpage: {url}")
        return texts
    except Exception as e:
        st.error(f"Error processing webpage {url}: {e}")
        return []

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama3-70b-8192",
        temperature=0
    )

def get_rag_chain(vector_store, llm):
    if vector_store is None:
        st.error("Vector store not initialized.")
        return None
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

def initialize_vector_store(documents, embeddings):
    if documents:
        if 'vector_store' in st.session_state and st.session_state.vector_store is not None:
            st.session_state.vector_store.add_documents(documents)
            st.info("Added new documents to existing FAISS index.")
        else:
            st.session_state.vector_store = FAISS.from_documents(documents, embeddings)
            st.info("Created new FAISS index.")
        st.session_state.vector_store.save_local(FAISS_INDEX_DIR)
        st.success("Vector store updated and saved locally!")
    else:
        st.warning("No documents to add.")

def display_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700px" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not display PDF: {e}")

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="RAG App with Groq")
st.title("📄 MANISH SINGH - RAG Application with Document & Web Chat (Groq, FAISS)")

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_display_path" not in st.session_state:
    st.session_state.pdf_display_path = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Initialize models
try:
    embeddings = get_embeddings()
    llm = get_llm()
except Exception as e:
    st.error(f"Failed to initialize models. Error: {e}")
    st.stop()

# Load existing FAISS
if os.path.exists(FAISS_INDEX_DIR):
    try:
        st.session_state.vector_store = FAISS.load_local(
            FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        st.success("Loaded existing FAISS index.")
    except Exception as e:
        st.warning(f"Could not load existing FAISS index. Starting fresh. Error: {e}")

# Sidebar
with st.sidebar:
    st.header("Upload & Ingest")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("Process Uploaded PDFs"):
        all_new_docs = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name
            new_docs = load_and_split_pdf(temp_file_path)
            all_new_docs.extend(new_docs)
            if not st.session_state.pdf_display_path:
                st.session_state.pdf_display_path = temp_file_path
        if all_new_docs:
            initialize_vector_store(all_new_docs, embeddings)
            st.session_state.rag_chain = get_rag_chain(st.session_state.vector_store, llm)
            st.session_state.messages = []
            st.rerun()

    st.subheader("Predefined Link Ingestion")  
    selected_company = st.selectbox("Select a source", [""] + list(PREDEFINED_PDF_LINKS.keys()))  
    if st.button("Ingest Selected Source"):  
        if selected_company:  
            all_docs = []  
            for url in PREDEFINED_PDF_LINKS[selected_company]:  
                if url.lower().endswith(".pdf"):  
                    file_name = os.path.basename(urlparse(url).path)  
                    output_path = os.path.join(PDF_DOWNLOAD_DIR, file_name)  
                    if download_pdf(url, output_path):  
                        docs = load_and_split_pdf(output_path)  
                        all_docs.extend(docs)  
                        if not st.session_state.pdf_display_path:  
                            st.session_state.pdf_display_path = output_path  
                else:  
                    docs = load_and_split_webpage(url)  
                    all_docs.extend(docs)  
            if all_docs:  
                initialize_vector_store(all_docs, embeddings)  
                st.session_state.rag_chain = get_rag_chain(st.session_state.vector_store, llm)  
                st.session_state.messages = []  
                st.rerun()

# Ensure RAG chain is ready
if st.session_state.vector_store and not st.session_state.rag_chain:
    st.session_state.rag_chain = get_rag_chain(st.session_state.vector_store, llm)

# Layout
col1, col2 = st.columns([0.6, 0.4])
with col1:
    st.subheader("📑 PDF Viewer")
    if st.session_state.pdf_display_path:
        display_pdf(st.session_state.pdf_display_path)
    else:
        st.info("Upload or ingest a PDF to view it here.")

with col2:
    st.subheader("💬 Chat with Data")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask something..."):  
        st.session_state.messages.append({"role": "user", "content": prompt})  
        with st.chat_message("assistant"):  
            if st.session_state.rag_chain is None:  
                reply = "Please upload or ingest documents first."  
                st.markdown(reply)  
                st.session_state.messages.append({"role": "assistant", "content": reply})  
            else:  
                with st.spinner("Thinking..."):  
                    try:  
                        chat_history_formatted = [  
                            (m['role'], m['content']) for m in st.session_state.messages if m['role'] != 'assistant'  
                        ]  
                        response = st.session_state.rag_chain.invoke({  
                            "question": prompt,  
                            "chat_history": chat_history_formatted  
                        })  
                        ai_response = response["answer"]  
                        st.markdown(ai_response)  
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})  
                    except Exception as e:  
                        st.error(f"Error during retrieval: {e}")  
                        st.session_state.messages.append({  
                            "role": "assistant",  
                            "content": "I'm sorry, I encountered an error."  
                        })        "https://public.dhe.ibm.com/software/webserver/appserv/library/v60/ihs_60.pdf",
        "https://www.ibm.com/docs/en/storage-protect/8.1.25?topic=pdf-files",
    ],
    "Cisco": [
        "https://www.cisco.com/c/dam/global/shared/assets/pdf/cisco_enterprise_campus_infrastructure_design_guide.pdf",
        "https://www.cisco.com/c/dam/en_us/about/ciscoitatwork/downloads/ciscoitatwork/pdf/Cisco_IT_Wireless_LAN_Design_Guide.pdf",
        "https://www.cisco.com/c/dam/en_us/about/ciscoitatwork/downloads/ciscoitatwork/pdf/Cisco_IT_IP_Addressing_Best_Practices.pdf",
        "https://www.cisco.com/c/en/us/td/docs/net_mgmt/network_registrar/7-2/user/guide/cnr72book.pdf",
    ],
    "Juniper": [
        "https://www.juniper.net/documentation/us/en/software/junos/junos-overview/junos-overview.pdf",
        "https://archive.org/download/junos-srxsme/JunOS%20SRX%20Documentation%20Set/network-management.pdf",
        "https://csrc.nist.gov/CSRC/media/projects/cryptographic-module-validation-program/documents/security-policies/140sp3779.pdf",
    ],
    "Fortinet (FortiGate)": [
        "https://fortinetweb.s3.amazonaws.com/docs.fortinet.com/v2/attachments/b94274f8-1a11-11e9-9685-f8bc1258b856/FortiOS-5.6-Firewall.pdf",
        "https://docs.fortinet.com/document/fortiweb/6.0.7/administration-guide-pdf",
        "https://www.andovercg.com/datasheets/fortigate-fortinet-200.pdf",
        "https://www.commoncriteriaportal.org/files/epfiles/Fortinet%20FortiGate_EAL4_ST_V1.5.pdf",
    ],
    "EUC": [
        "https://www.dell.com/en-us/lp/dt/end-user-computing",
        "https://www.nutanix.com/solutions/end-user-computing",
        "https://eucscore.com/docs/tools.html",
        "https://apparity.com/euc-resources/spreadsheet-euc-documents/",
    ],
}

# --- Helper Functions ---

def download_pdf(url, output_path):
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success(f"Downloaded: {os.path.basename(output_path)}")
        return True
    except Exception as e:
        st.error(f"Error downloading {url}: {e}")
        return False

def load_and_split_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(documents)
        st.success(f"Processed {len(texts)} chunks from {os.path.basename(file_path)}")
        return texts
    except Exception as e:
        st.error(f"Error processing PDF {os.path.basename(file_path)}: {e}")
        return []

def load_and_split_webpage(url):
    try:
        loader = WebBaseLoader(url)
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = splitter.split_documents(documents)
        st.success(f"Processed {len(texts)} chunks from webpage: {url}")
        return texts
    except Exception as e:
        st.error(f"Error processing webpage {url}: {e}")
        return []

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama3-70b-8192",
        temperature=0
    )

def get_rag_chain(vector_store, llm):
    if vector_store is None:
        st.error("Vector store not initialized.")
        return None
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True
    )

def initialize_vector_store(documents, embeddings):
    if documents:
        if 'vector_store' in st.session_state and st.session_state.vector_store is not None:
            st.session_state.vector_store.add_documents(documents)
            st.info("Added new documents to existing FAISS index.")
        else:
            st.session_state.vector_store = FAISS.from_documents(documents, embeddings)
            st.info("Created new FAISS index.")
        st.session_state.vector_store.save_local(FAISS_INDEX_DIR)
        st.success("Vector store updated and saved locally!")
    else:
        st.warning("No documents to add.")

def display_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="700px" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not display PDF: {e}")

# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="RAG App with Groq")
st.title("📄 MANISH SINGH - RAG Application with Document & Web Chat (Groq, FAISS)")

# Session state init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "pdf_display_path" not in st.session_state:
    st.session_state.pdf_display_path = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

# Initialize models
try:
    embeddings = get_embeddings()
    llm = get_llm()
except Exception as e:
    st.error(f"Failed to initialize models. Error: {e}")
    st.stop()

# Load existing FAISS
if os.path.exists(FAISS_INDEX_DIR):
    try:
        st.session_state.vector_store = FAISS.load_local(
            FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        st.success("Loaded existing FAISS index.")
    except Exception as e:
        st.warning(f"Could not load existing FAISS index. Starting fresh. Error: {e}")

# Sidebar
with st.sidebar:
    st.header("Upload & Ingest")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files and st.button("Process Uploaded PDFs"):
        all_new_docs = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name
            new_docs = load_and_split_pdf(temp_file_path)
            all_new_docs.extend(new_docs)
            if not st.session_state.pdf_display_path:
                st.session_state.pdf_display_path = temp_file_path
        if all_new_docs:
            initialize_vector_store(all_new_docs, embeddings)
            st.session_state.rag_chain = get_rag_chain(st.session_state.vector_store, llm)
            st.session_state.messages = []
            st.rerun()

    st.subheader("Predefined Link Ingestion")
    selected_company = st.selectbox("Select a source", [""] + list(PREDEFINED_PDF_LINKS.keys()))
    if st.button("Ingest Selected Source"):
        if selected_company:
            all_docs = []
            for url in PREDEFINED_PDF_LINKS[selected_company]:
                if url.lower().endswith(".pdf"):
                    file_name = os.path.basename(urlparse(url).path)
                    output_path = os.path.join(PDF_DOWNLOAD_DIR, file_name)
                    if download_pdf(url, output_path):
                        docs = load_and_split_pdf(output_path)
                        all_docs.extend(docs)
                        if not st.session_state.pdf_display_path:
                            st.session_state.pdf_display_path = output_path
                else:
                    docs = load_and_split_webpage(url)
                    all_docs.extend(docs)
            if all_docs:
                initialize_vector_store(all_docs, embeddings)
                st.session_state.rag_chain = get_rag_chain(st.session_state.vector_store, llm)
                st.session_state.messages = []
                st.rerun()

# Ensure RAG chain is ready
if st.session_state.vector_store and not st.session_state.rag_chain:
    st.session_state.rag_chain = get_rag_chain(st.session_state.vector_store, llm)

# Layout
col1, col2 = st.columns([0.6, 0.4])
with col1:
    st.subheader("📑 PDF Viewer")
    if st.session_state.pdf_display_path:
        display_pdf(st.session_state.pdf_display_path)
    else:
        st.info("Upload or ingest a PDF to view it here.")

with col2:
    st.subheader("💬 Chat with Data")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask something..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            if st.session_state.rag_chain is None:
                reply = "Please upload or ingest documents first."
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            else:
                with st.spinner("Thinking..."):
                    try:
                        chat_history_formatted = [
                            (m['role'], m['content']) for m in st.session_state.messages if m['role'] != 'assistant'
                        ]
                        response = st.session_state.rag_chain.invoke({
                            "question": prompt,
                            "chat_history": chat_history_formatted
                        })
                        ai_response = response["answer"]
                        st.markdown(ai_response)
                        st.session_state.messages.append({"role": "assistant", "content": ai_response})
                    except Exception as e:
                        st.error(f"Error during retrieval: {e}")
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "I'm sorry, I encountered an error."
                        })    except Exception as e:
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
