# Receipt OCR API

A FastAPI-based OCR service for extracting and validating information from receipt images and PDFs.

## Features

- Extract text from images (JPEG, PNG) and PDFs using Tesseract OCR
- Identify monetary amounts and dates from receipts
- Validate extracted data against expected values
- Batch processing for multiple files
- RESTful API with comprehensive error handling

## Installation

### Option 1: Using Docker (Recommended)

```bash
# Clone the repository
git clone <your-repo>
cd receipt-ocr

# Build and run with Docker Compose
docker-compose up --build