import re
import os
import pickle
import json
import logging
import icd10

from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.output_parsers import JsonOutputParser

from pdf2image import convert_from_path
import pytesseract

from create_db import MockUploadFile

DB_PATH = "vectorstores/db_faiss"
SPLIT_DOCS_PATH = os.path.join(DB_PATH, "split_docs.pkl")
EMBEDDING_MODEL = "nomic-embed-text"

LLM_MODEL = "phi4-reasoning:14b" 
# LLM_MODEL = "llama3.3:70b" 
# LLM_MODEL = "gpt-oss:20b"


ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

LLM_TEMPERATURE = 0
RETRIEVER_K = 5
ENSEMBLE_WEIGHTS = [0.5, 0.5]
EQUIPMENT_KEYWORDS = ["CGM", "CONTINUOUS GLUCOSE MONITOR", "INSULIN PUMP", "LIBRE", "DEXCOM", "BLOOD GLUCOSE METER"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class PrescriptionAnalyzer:
    def __init__(self):
        self.db_path = DB_PATH
        self._load_components()
    
    def _load_components(self):
        if not os.path.exists(self.db_path):
            return
        
        logger.info("Loading analysis components...")
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=ollama_base_url)
        self.db = FAISS.load_local(self.db_path, self.embeddings, allow_dangerous_deserialization=True)
        self.vector_retriever = self.db.as_retriever(search_kwargs={"k": RETRIEVER_K})
        
        with open(SPLIT_DOCS_PATH, 'rb') as f:
            split_docs = pickle.load(f)
        
        self.bm25_retriever = BM25Retriever.from_documents(split_docs)
        self.bm25_retriever.k = RETRIEVER_K
        
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.vector_retriever, self.bm25_retriever],
            weights=ENSEMBLE_WEIGHTS
        )
        
        self.llm = OllamaLLM(
            model=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            base_url=ollama_base_url,
            format='json',
        )
        
        self.json_parser = JsonOutputParser()
        logger.info("✅ Analysis components loaded successfully.")

    def reload_components(self):
        logger.info("Reloading components...")
        self._load_components()

    def _extract_icd_codes(self, text: str):
        """Regex to find ICD-10 codes like E10.65"""
        potential_codes = re.findall(r'\b[A-Z][0-9][A-Z0-9.]{1,6}\b', text.upper())
        valid_codes = [code for code in potential_codes if icd10.find(code)]
        return sorted(list(set(valid_codes)))

    def _extract_equipment_keywords(self, text: str):
        """Fallback keyword search"""
        found_keywords = set()
        for keyword in EQUIPMENT_KEYWORDS:
            if keyword in text.upper():
                if "CGM" in keyword or "LIBRE" in keyword or "DEXCOM" in keyword or "MONITOR" in keyword:
                    found_keywords.add("Continuous Glucose Monitor (CGM)")
                elif "PUMP" in keyword:
                    found_keywords.add("Insulin Pump")
                elif "METER" in keyword:
                    found_keywords.add("Blood Glucose Meter")
        return list(found_keywords)

    
    def analyze_prescription(self, pdf_path):
        try:
            full_text = self._extract_pdf_text(pdf_path)
            regex_icds = self._extract_icd_codes(full_text)
            patient_data = self._extract_patient_entities(full_text)
            
            llm_icds = patient_data.get('icd_codes', [])
            llm_equip = patient_data.get('equipment', [])

            combined_icds = sorted(list(set(regex_icds + llm_icds)))
            combined_equip = llm_equip

            logger.info(f"🏥 MERGED DATA: ICDs={combined_icds}, Equip={combined_equip}")

            if not combined_icds and not combined_equip:
                return {"success": False, "error": "No valid ICD codes or Equipment found."}

            policy_context = self._retrieve_policy_context(combined_icds, combined_equip)
            
            full_patient_data = {
                "icd_codes": combined_icds,
                "equipment": combined_equip
            }
            
            final_result = self._verify_coverage(full_text, full_patient_data, policy_context)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Analysis Error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _extract_pdf_text(self, pdf_path):
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            text = "\n".join([doc.page_content for doc in documents])
            if text.strip(): return text
        except:
            pass
        pages = convert_from_path(pdf_path)
        return "\n".join([pytesseract.image_to_string(page) for page in pages])

    def _extract_patient_entities(self, text: str):
        """
        Stage 1: Pure Extraction. No Policy logic yet.
        """
        prompt = f"""
        Extract medical data from this raw OCR text.
        
        TEXT:
        {text[:4000]}  # Limit text to avoid context overflow if needed

        TASK:
        Identify:
        1. All **ICD-10 Codes** mentioned (e.g., E10.65, Z96.41).
        2. All **Medical Equipment/Device Names** requested (e.g., "Freestyle Libre 3", "Dexcom", "Insulin Pump").
           - Do NOT look for HCPCS codes (like A9276) unless explicitly written in the text.
           - Extract the commercial name.

        OUTPUT JSON:
        {{
            "icd_codes": ["Code1", "Code2"],
            "equipment": ["Device Name 1", "Device Name 2"]
        }}
        """
        try:
            response = self.llm.invoke(prompt)
            clean_json = response[response.find('{'):response.rfind('}')+1]
            return json.loads(clean_json)
        except Exception as e:
            logger.error(f"Entity Extraction Failed: {e}")
            return {"icd_codes": [], "equipment": []}

    def _retrieve_policy_context(self, icd_codes: list, equipment: list):
        queries = []
        for code in icd_codes:
            queries.append(f"Coverage requirements for diagnosis code {code}")
        
        for item in equipment:
            queries.append(f"HCPCS code and policy criteria for {item}")

        if not queries: return ""
        
        search_results = self.ensemble_retriever.batch(queries)
        context_str = ""
        for i, query in enumerate(queries):
            context_str += f"--- POLICY SECTION FOR: '{query}' ---\n"
            unique_docs = {doc.page_content for doc in search_results[i]}
            context_str += "\n".join(unique_docs)
            context_str += "\n\n"
        return context_str
    
    def _verify_coverage(self, full_text, patient_data, policy_context):
        """Stage 2: Verification with Dynamic Count Enforcement"""
        
        icds = patient_data.get('icd_codes', [])
        equips = patient_data.get('equipment', [])
        
        # 1. Calculate counts to force the LLM
        icd_count = len(icds)
        equip_count = len(equips)
        total_items = icd_count + equip_count

        # 2. Build explicit checklists
        icd_checklist = "\n".join([f"{i+1}. ICD: {code}" for i, code in enumerate(icds)])
        equip_checklist = "\n".join([f"{i+1}. EQUIP: {item}" for i, item in enumerate(equips)])

        prompt = f"""
        **ROLE**: Medical Auditor Bot.
        
        **TASK**: Verify coverage for the following **{total_items} SPECIFIC ITEMS**.
        
        **INPUT CHECKLIST (You must process ALL {total_items} items)**:
        {icd_checklist}
        {equip_checklist}

        ______________________
        **POLICY RULES**:
        {policy_context}

        ______________________
        **PATIENT RECORD**:
        {full_text}

        ______________________
        **STRICT OUTPUT RULES**:
        1. **QUANTITY CHECK**:
           - You were given **{icd_count}** ICD codes and **{equip_count}** Equipment items.
           - Your output JSON `analysis_results` array **MUST** contain exactly **{total_items}** objects.
           - **DO NOT STOP** after the first valid match. Process the secondary codes too (e.g., Z96.41).

        2. **VERIFICATION LOGIC**:
           - **For ICDs**: Check if the code exists in the `PATIENT RECORD`. If it is present, `match_found` is true.
           - **For Equipment**: Map the requested name to the HCPCS code in `POLICY RULES`. 

        3. **FORMAT**:
           - Provide the result as a standard JSON object.

        **JSON TEMPLATE**:
        {{
            "success": true,
            "analysis_results": [
                {{
                    "type": "ICD_CODE",
                    "code": "E10.65",
                    "description": "...",
                    "match_found": true, 
                    "justification": "...",
                    "evidence_quote": "..." 
                }},
                {{
                    "type": "ICD_CODE",
                    "code": "Z96.41", 
                    ...
                }}
                ... (Repeat for ALL {total_items} items)
            ]
        }}
        """
        
        try:
            logger.info(f"Invoking LLM to verify {total_items} items ({icd_count} ICDs, {equip_count} Equip)...")
            response_str = self.llm.invoke(prompt)
            
            # Robust JSON Extraction
            first_brace = response_str.find('{')
            last_brace = response_str.rfind('}')
            
            if first_brace == -1 or last_brace == -1:
                raise ValueError("LLM failed to return JSON")
                
            clean_json = response_str[first_brace : last_brace + 1]
            result = json.loads(clean_json)
            
            if "analysis_results" not in result:
                result = {"analysis_results": []}
            
            result["success"] = True
            
            # Log the count mismatch if it happens
            result_count = len(result["analysis_results"])
            if result_count != total_items:
                logger.warning(f"⚠️ LLM Lazy Warning: Expected {total_items} results, got {result_count}.")

            return result

        except Exception as e:
            logger.error(f"Verification Error: {e}")
            return {
                "success": False, 
                "error": str(e),
                "analysis_results": []
            }

    # def _verify_coverage(self, full_text, patient_data, policy_context):
    #     """
    #     Stage 2: The Audit. Map Patient Items -> Policy Rules.
    #     """
        
    #     icds = patient_data.get('icd_codes', [])
    #     equips = patient_data.get('equipment', [])

    #     prompt = f"""
    #     **ROLE**: Medical Insurance Auditor.
        
    #     **OBJECTIVE**: 
    #     Determine if the Patient's Request is supported by the Policy Rules.

    #     **1. PATIENT DATA (From Prescription)**:
    #     - Diagnoses: {icds}
    #     - Requested Equipment: {equips}
        
    #     **2. POLICY RULES (From Database)**:
    #     {policy_context}

    #     **3. FULL PATIENT NOTE (For evidence quotes)**:
    #     {full_text}

    #     **INSTRUCTIONS**:
    #     For each item in the "PATIENT DATA":
    #     1. **Map to Policy**: Find the corresponding HCPCS code or Rule in the Policy text.
    #        - *Example:* If patient wants "Libre 3", and Policy says "Libre 3 maps to A9276", then A9276 is the Policy Code.
    #     2. **Verify**: Does the patient's diagnosis (from PATIENT DATA) meet the criteria in the POLICY RULES?
    #     3. **Quote**: Provide an EXACT quote from the **FULL PATIENT NOTE** that proves the patient has this condition/request.

    #     **JSON OUTPUT**:
    #     {{
    #         "success": true,
    #         "analysis_results": [
    #             {{
    #                 "type": "ICD_CODE",
    #                 "code": "E10.65",
    #                 "description": "Description found in policy",
    #                 "match_found": true, 
    #                 "justification": "Policy explicitly covers E10.65 for CGM",
    #                 "evidence_quote": "Patient diagnosed with E10.65" 
    #             }},
    #             {{
    #                 "type": "HCPCS_EQUIPMENT",
    #                 "requested_item": "Freestyle Libre 3",
    #                 "policy_code": "A9276 (Found in Policy)",
    #                 "match_found": true,
    #                 "justification": "Libre 3 (A9276) is covered because patient has Type 1 Diabetes (E10.65)",
    #                 "evidence_quote": "Prescription for freestyle libre 3"
    #             }}
    #         ]
    #     }}
    #     """
        
    #     try:
    #         response_str = self.llm.invoke(prompt)
    #         clean_json = response_str[response_str.find('{'):response_str.rfind('}')+1]
    #         result = json.loads(clean_json)
            
    #         if "analysis_results" not in result:
    #             result = {"analysis_results": []}
    #         result["success"] = True
    #         return result

    #     except Exception as e:
    #         logger.error(f"Verification Error: {e}")
    #         return {"success": False, "error": str(e)}

def test_analyzer(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: File {pdf_path} not found")
        return
    
    print("🏥 PRESCRIPTION ANALYSIS")
    print("=" * 50)
    
    analyzer = PrescriptionAnalyzer()
    result = analyzer.analyze_prescription(pdf_path)
    
    print("\n--- RAW JSON RESPONSE ---")
    print(json.dumps(result, indent=2))
    print("-------------------------\n")
    
    if result.get('success'):
        print("✅ Analysis completed successfully!")
        print("\n📋 DETAILED RESULTS:")
        
        for item in result.get('analysis_results', []):
            print("-" * 30)
            print(f"Type: {item.get('type')}")
            if item.get('type') == 'ICD_CODE':
                print(f"Code: {item.get('code')}")
                print(f"Description: {item.get('description')}")
            else:
                print(f"Item: {item.get('requested_item')}")
                print(f"Policy Code: {item.get('policy_code')}")

            print(f"Match Found: {item.get('match_found')}")
            print(f"Justification: {item.get('justification')}")
            print(f"Evidence Quote: '{item.get('evidence_quote')}'")
        print("-" * 30)
    else:
        print(f"❌ Error: {result.get('error')}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python api_processor.py <path_to_prescription_pdf>")
        print("Example: python api_processor.py 'Prescriptions/Fax_Referral.pdf'")
    else:
        test_analyzer(sys.argv[1])