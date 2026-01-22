import os
import tempfile
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pytesseract
from PIL import Image
import cv2
import numpy as np
import fitz  # PyMuPDF for PDFs
from pdf2image import convert_from_path
import dateutil.parser
import logging

from app.config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReceiptOCRProcessor:
    def __init__(self, tesseract_path: str = None):
        """
        Initialize OCR processor.
        
        Args:
            tesseract_path: Path to tesseract executable (if not in PATH)
        """
        if tesseract_path and os.path.exists(tesseract_path):
            # pytesseract.pytesseract.tesseract_cmd = tesseract_path
            try:
                pytesseract.pytesseract.tesseract_cmd = Config.get_tesseract_path()
            except:
                # Fallback to PATH
                pytesseract.pytesseract.tesseract_cmd = 'tesseract'
        
        # Common receipt patterns
        self.amount_patterns = [
            r'(?:total|amount|amt|sum|balance)[\s:]*[\$£€¥₹₦]?\s*(\d+[,\d]*\.?\d*)',
            r'[\$£€¥₹₦]\s*(\d+[,\d]*\.?\d*)',
            r'(\d+[,\d]*\.?\d\d)\s*(?:usd|ngn|gbp|eur|inr)',
        ]
        
        self.date_patterns = [
            r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b',
            r'\b(\d{2,4}[/\-]\d{1,2}[/\-]\d{1,2})\b',
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2},? \d{4}\b',
            r'\b\d{1,2} (?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{4}\b',
        ]
        
        self.currency_symbols = ['$', '£', '€', '¥', '₹', '₦', 'USD', 'NGN', 'GBP', 'EUR', 'INR']
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply adaptive thresholding
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Remove noise
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            
            # Enhance contrast
            processed = cv2.convertScaleAbs(processed, alpha=1.5, beta=0)
            
            return processed
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image
    
    def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from image using Tesseract OCR.
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            # Preprocess image
            processed_img = self.preprocess_image(img)
            
            # Convert to PIL Image for pytesseract
            pil_img = Image.fromarray(processed_img)
            
            # Extract text with multiple configurations
            config = '--oem 3 --psm 6'
            text = pytesseract.image_to_string(pil_img, config=config)
            
            # Try with different PSM if no text found
            if not text.strip():
                config = '--oem 3 --psm 11'
                text = pytesseract.image_to_string(pil_img, config=config)
            
            logger.info(f"Extracted text from {image_path}")
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from image {image_path}: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.
        """
        try:
            text = ""
            
            # Method 1: Try PyMuPDF first (faster for text-based PDFs)
            try:
                doc = fitz.open(pdf_path)
                for page in doc:
                    text += page.get_text()
                doc.close()
                
                if text.strip():
                    logger.info(f"Extracted text from PDF using PyMuPDF: {pdf_path}")
                    return text.strip()
            except:
                pass
            
            # Method 2: Convert PDF to images and use OCR
            images = convert_from_path(pdf_path)
            for i, image in enumerate(images):
                # Save temp image
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_path = temp_file.name
                    image.save(temp_path, 'PNG')
                
                # Extract text from image
                page_text = self.extract_text_from_image(temp_path)
                text += f"\n--- Page {i+1} ---\n{page_text}\n"
                
                # Clean up
                os.unlink(temp_path)
            
            logger.info(f"Extracted text from PDF using OCR: {pdf_path}")
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""
    
    def extract_amounts(self, text: str) -> List[float]:
        """
        Extract monetary amounts from text.
        """
        amounts = []
        text_lower = text.lower()
        
        for pattern in self.amount_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                try:
                    # Clean the amount string
                    clean_amount = match.replace(',', '')
                    amount = float(clean_amount)
                    amounts.append(amount)
                except ValueError:
                    continue
        
        # Also look for standalone numbers that could be amounts
        standalone_numbers = re.findall(r'\b(\d+[,\d]*\.?\d\d)\b', text)
        for num in standalone_numbers:
            try:
                clean_num = num.replace(',', '')
                amount = float(clean_num)
                # Only consider reasonable amounts (not too small or too large for receipts)
                if 1 <= amount <= 1000000:  # Adjust range as needed
                    amounts.append(amount)
            except ValueError:
                continue
        
        # Remove duplicates and sort
        unique_amounts = sorted(list(set(amounts)))
        return unique_amounts
    
    def extract_dates(self, text: str) -> List[datetime]:
        """
        Extract dates from text.
        """
        dates = []
        
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    # Try to parse with dateutil (handles multiple formats)
                    date_obj = dateutil.parser.parse(match, fuzzy=True)
                    dates.append(date_obj)
                except:
                    continue
        
        # Try to find common date formats
        common_patterns = [
            r'\b(\d{4}-\d{2}-\d{2})\b',  # YYYY-MM-DD
            r'\b(\d{2}/\d{2}/\d{4})\b',  # MM/DD/YYYY
            r'\b(\d{2}-\d{2}-\d{4})\b',  # DD-MM-YYYY
        ]
        
        for pattern in common_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    date_obj = datetime.strptime(match, '%Y-%m-%d')
                    dates.append(date_obj)
                except:
                    try:
                        date_obj = datetime.strptime(match, '%m/%d/%Y')
                        dates.append(date_obj)
                    except:
                        try:
                            date_obj = datetime.strptime(match, '%d-%m-%Y')
                            dates.append(date_obj)
                        except:
                            continue
        
        # Remove duplicates
        unique_dates = []
        seen = set()
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            if date_str not in seen:
                seen.add(date_str)
                unique_dates.append(date)
        
        return sorted(unique_dates)
    
    def validate_against_expected(
        self, 
        extracted_amounts: List[float], 
        extracted_dates: List[datetime], 
        expected_amount: float, 
        expected_date: str
    ) -> Dict[str, Any]:
        """
        Validate extracted data against expected values.
        """
        try:
            expected_date_obj = datetime.strptime(expected_date, '%Y-%m-%d')
            
            # Find closest amount match
            amount_matches = False
            matched_amount = None
            amount_difference = float('inf')
            
            for amount in extracted_amounts:
                diff = abs(amount - expected_amount)
                # Allow 1% tolerance or ₦10, whichever is smaller
                tolerance = min(expected_amount * 0.01, 10)
                if diff <= tolerance:
                    if diff < amount_difference:
                        amount_difference = diff
                        matched_amount = amount
                        amount_matches = True
            
            # Find closest date match
            date_matches = False
            matched_date = None
            date_difference = float('inf')
            
            for date in extracted_dates:
                diff = abs((date - expected_date_obj).days)
                # Allow ±1 day tolerance
                if diff <= 1:
                    if diff < date_difference:
                        date_difference = diff
                        matched_date = date
                        date_matches = True
            
            return {
                "is_valid": amount_matches and date_matches,
                "amount_matches": amount_matches,
                "date_matches": date_matches,
                "extracted_amounts": extracted_amounts,
                "extracted_dates": [d.strftime('%Y-%m-%d') for d in extracted_dates],
                "matched_amount": matched_amount,
                "matched_date": matched_date.strftime('%Y-%m-%d') if matched_date else None,
                "amount_difference": amount_difference if not amount_matches else 0,
                "date_difference": date_difference if not date_matches else 0,
                "confidence": self.calculate_confidence(extracted_amounts, extracted_dates)
            }
            
        except Exception as e:
            logger.error(f"Error in validation: {e}")
            return {
                "is_valid": False,
                "amount_matches": False,
                "date_matches": False,
                "error": str(e)
            }
    
    def calculate_confidence(self, amounts: List[float], dates: List[datetime]) -> float:
        """
        Calculate confidence score based on extracted data.
        """
        confidence = 0.0
        
        # Higher confidence if we found amounts
        if amounts:
            confidence += 0.4
        
        # Higher confidence if we found dates
        if dates:
            confidence += 0.4
        
        # Higher confidence if we found multiple amounts (more data)
        if len(amounts) > 1:
            confidence += 0.1
        
        # Higher confidence if we found multiple dates
        if len(dates) > 1:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def process_file(
        self, 
        file_path: str, 
        expected_amount: float = None, 
        expected_date: str = None
    ) -> Dict[str, Any]:
        """
        Process a single file and extract information.
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Determine file type
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Extract text based on file type
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                extracted_text = self.extract_text_from_image(file_path)
            elif file_ext == '.pdf':
                extracted_text = self.extract_text_from_pdf(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # Extract amounts and dates
            extracted_amounts = self.extract_amounts(extracted_text)
            extracted_dates = self.extract_dates(extracted_text)
            
            result = {
                "file_path": file_path,
                "file_type": file_ext[1:],  # Remove dot
                "text_extracted": bool(extracted_text.strip()),
                "extracted_text_sample": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
                "extracted_amounts": extracted_amounts,
                "extracted_dates": [d.strftime('%Y-%m-%d') for d in extracted_dates],
                "processing_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Validate against expected values if provided
            if expected_amount is not None and expected_date is not None:
                validation_result = self.validate_against_expected(
                    extracted_amounts, extracted_dates, expected_amount, expected_date
                )
                result.update(validation_result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {
                "file_path": file_path,
                "error": str(e),
                "success": False
            }


# Singleton instance
ocr_processor = ReceiptOCRProcessor()