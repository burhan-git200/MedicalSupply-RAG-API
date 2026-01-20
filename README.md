### **System Name: RAG-Based Medical Prescription Auditor**

**Core Purpose:**
This is an automated **AI verification engine** designed to cross-reference patient prescriptions (often scanned Faxes) against complex medical insurance policies. Its goal is to automate the "Prior Authorization" or "Claims Auditing" process to ensure a requested device (like a CGM) or diagnosis (ICD Code) is covered by the payer.

---

### **1. Architecture & Workflow**
The system operates as a **Retrieval-Augmented Generation (RAG)** pipeline.

1.  **Ingestion (Policy Side):**
    *   **`policy_manager.py`**: Accepts raw Policy PDFs (e.g., Medicare guidelines).
    *   It chunks the text, creates embeddings using **Ollama (`nomic-embed-text`)**, and stores them in a **FAISS Vector Database**.
    *   It also maintains a "Sparse" index (BM25) for keyword-based retrieval.

2.  **Processing (Prescription Side):**
    *   **`api_processor.py`**: This is the core logic engine.
    *   **Step 1: OCR:** Converts scanned PDF images into text using `pytesseract`.
    *   **Step 2: Hybrid Extraction (The "Smart" Part):**
        *   Uses **Regex** to mathematically find every specific ICD-10 code (e.g., `E10.65`, `Z96.41`).
        *   Uses **LLM (Stage 1)** to identify complex medical equipment names (e.g., "Freestyle Libre 3").
    *   **Step 3: Retrieval:** Takes the extracted codes/items and searches the FAISS database for the specific rules regarding those items.
    *   **Step 4: Verification (Stage 2):** Sends the Patient Note + The Specific Policy Rules to the LLM. The LLM acts as an auditor to verify coverage and extract "Evidence Quotes".

3.  **Delivery (API):**
    *   **`main.py`**: A **FastAPI** application that orchestrates the flow.
    *   It manages concurrency (`run_in_threadpool`) to prevent the heavy AI tasks from blocking the server.

---

### **2. Key Logic Features**
*   **Two-Stage Reasoning:** Instead of blindly asking "Is this approved?", the system first identifies *what* is being asked for, and *then* verifies it. This reduces hallucinations.
*   **Force-Directed Prompting:** The prompts are engineered to stop the LLM from being "lazy." It explicitly counts items (e.g., "You have 3 codes") and forces the LLM to output a JSON result for *every* item.
*   **Hybrid Search:** Uses an **Ensemble Retriever** (Vector Search + Keyword Search) to ensure it finds policy rules even if the wording is slightly different.

---

### **3. Infrastructure (DevOps)**
*   **Dockerized Microservices:**
    *   **Nginx:** Reverse proxy handling SSL/Termination.
    *   **API:** Python container running the business logic.
    *   **Ollama:** A dedicated container running the AI models on GPU.
*   **Hardware Optimization:**
    *   Configured for **NVIDIA GPUs** (specifically referencing an RTX 5090).
    *   Includes `shm_size: 32gb` to handle large context windows without crashing.

### **Summary**
We have built a **specialized, local-hosted AI Auditor**. It is not a generic chatbot; it is a structured extraction and verification pipeline capable of reading messy medical faxes and performing logical checks against a rulebook (the Policy).

# How To Run

Here are the exact commands to deploy and initialize your system on a new server or location.

**Prerequisites:**
Ensure the `Policies` folder (containing your PDF) is in the root directory next to `docker-compose.yml`, exactly as shown in your screenshot.

**For Building and Starting the System:**
```bash
docker-compose up -d --build
```
*(This starts the API, Nginx, and Ollama containers in the background)*

**For Creating the Initial Database (Seeding from create_db.py):**
```bash
docker exec prescription_api python create_db.py
```
*(This runs the python script inside the running container to process the PDF and generate the vector store files)*

**For Loading the New Database into API Memory:**
```bash
docker restart prescription_api
```
*(Since the API loads the database only on startup, you must restart it once after creating the DB so it sees the new files)*

**For Checking Real-time Logs:**
```bash
docker logs -f prescription_api
```
*(Use this to watch the analysis happening or debug errors)*

**Ollama Commands:**
```bash
docker exec ollama_server ollama list
```


**For Stopping the System:**
```bash
docker-compose down
```