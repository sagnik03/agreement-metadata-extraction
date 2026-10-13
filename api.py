import os
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File

from metadata_extractor import MetadataExtractor


app = FastAPI(title='Document Metadata Extractor')
extractor = MetadataExtractor()


@app.post('/extract')
async def extract_metadata(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1] or '.bin'
    with NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await file.read())
        temp_path = temp_file.name

    try:
        result = extractor.extract_file(temp_path)
        return {'file_name': file.filename, 'metadata': result}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.get('/')
async def root():
    return {'status': 'ok', 'message': 'Use POST /extract with a .docx or .png file'}
