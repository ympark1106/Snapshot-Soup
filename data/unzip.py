import zipfile
import os 
# Function to extract ZIP file
def extract_zip(zip_path, extract_to):
    if not os.path.exists(extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    print(f"Extracted ZIP file to {extract_to}")
    
if __name__ == '__main__':
    # Extract the Diabetic Retinopathy Detection dataset
    zip_path = '/home/youmin/workspace/VFMs-Adapters-Ensemble/diabetic-retinopathy-detection.zip'
    extract_to = '/home/youmin/workspace/VFMs-Adapters-Ensemble/diabetic-retinopathy-detection'
    extract_zip(zip_path, extract_to)
    