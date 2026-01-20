import os
from policy_manager import PolicyManager

class MockUploadFile:
    def __init__(self, filename, file_path):
        self.filename = filename
        self.file = open(file_path, 'rb')
    
    def close(self):
        if not self.file.closed:
            self.file.close()

policy_manager = PolicyManager()

pdf_path = r"Policies/Article - Glucose Monitor - Policy Article (A52464) (1).pdf"
try:
    upload_file = MockUploadFile(os.path.basename(pdf_path), pdf_path)

    result = policy_manager.update_vector_store(upload_file)
    print(result)
except Exception as e:
    print(f"Error: {e}")
    e.add_note("Error occurred while updating vector store")
finally:
    upload_file.close()