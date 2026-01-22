import os
import re
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import pytesseract
from PIL import Image
import cv2
import numpy as np
import fitz
from pdf2image import convert_from_path
from dateutil.parser import parse
import tempfile

logger = logging.getLogger(__name__)

class ReceiptOCRProcessor:
    def __init__(self, tesseract_path: str = None):
        if tesseract_path and os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Nigerian-specific receipt patterns
        self.amount_patterns = [
            # Nigerian currency patterns
            r'(?:₦|NGN|Naira)[\s:]*[\s]*([\d,]+\.?\d*)',
            r'(?:amount|amt|total|sum|balance|transfer|sent|received)[\s:]*[\s₦NGN]*([\d,]+\.?\d*)',
            r'([\d,]+\.?\d{2})\s*(?:₦|NGN|naira)',
            # Transaction amount patterns (common in Nigerian receipts)
            r'(?:transaction|txn|trans)[\s:]*[\s]*([\d,]+\.?\d*)',
            r'#?([\d,]+\.?\d{2})\b',
        ]
        
        # Nigerian date patterns (DD/MM/YYYY, DD-MM-YYYY, etc.)
        self.date_patterns = [
            r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b',
            r'\b(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b',
            r'\b(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4})\b',
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
            # Nigerian format with time
            r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?\b',
        ]
        
        # Transaction ID patterns
        self.transaction_patterns = [
            r'transaction\s*(?:no|number|id|#)[\s:]*[\s]*([a-zA-Z0-9\-]+)',
            r'txn\s*(?:no|number|id|#)[\s:]*[\s]*([a-zA-Z0-9\-]+)',
            r'trans\s*(?:no|number|id|#)[\s:]*[\s]*([a-zA-Z0-9\-]+)',
            r'ref\s*(?:no|number|id|#)[\s:]*[\s]*([a-zA-Z0-9\-]+)',
            r'([A-Z0-9]{10,20})\b',  # Long alphanumeric strings
        ]

    def preprocess_image_for_nigerian_receipts(self, image: np.ndarray) -> np.ndarray:
        """
        Special preprocessing for Nigerian receipts (often have colors, watermarks)
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply bilateral filter to preserve edges while removing noise
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Apply adaptive thresholding
            processed = cv2.adaptiveThreshold(
                filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Apply morphological operations to clean up text
            kernel = np.ones((1, 1), np.uint8)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            
            # Apply dilation to make text thicker
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.dilate(processed, kernel, iterations=1)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {e}")
            return image

    def extract_text_with_optimizations(self, image_path: str) -> str:
        """
        Extract text with optimizations for Nigerian receipts
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            # Preprocess for Nigerian receipts
            processed_img = self.preprocess_image_for_nigerian_receipts(img)
            
            # Convert to PIL Image
            pil_img = Image.fromarray(processed_img)
            
            # Try multiple OCR configurations
            configs = [
                '--oem 3 --psm 6',  # Assume uniform block of text
                '--oem 3 --psm 11',  # Sparse text
                '--oem 3 --psm 4',   # Single column of text
                '--oem 3 --psm 3',   # Fully automatic page segmentation
            ]
            
            all_text = ""
            for config in configs:
                text = pytesseract.image_to_string(pil_img, config=config)
                all_text += text + "\n---\n"
            
            # Clean up the text
            cleaned_text = self.clean_extracted_text(all_text)
            
            logger.info(f"Extracted {len(cleaned_text)} characters from {image_path}")
            return cleaned_text
            
        except Exception as e:
            logger.error(f"Error extracting text from image {image_path}: {e}")
            return ""

    def clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common OCR errors in Nigerian receipts
        replacements = {
            '0Pay': 'OPay',
            '0pay': 'OPay',
            'QPay': 'OPay',
            'Naira': 'NGN',
            'Naira ': 'NGN ',
            'naira': 'NGN',
            'transction': 'transaction',
            'transacation': 'transaction',
            'sucessful': 'successful',
            'reciept': 'receipt',
            'recieved': 'received',
        }
        
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
        
        return text.strip()

    def extract_amounts_smart(self, text: str) -> List[float]:
        """
        Smart amount extraction for Nigerian receipts
        """
        amounts = []
        text_lower = text.lower()
        
        # First, look for amount with NGN/₦ symbol
        ngn_patterns = [
            r'₦\s*([\d,]+\.?\d*)',
            r'ngn\s*([\d,]+\.?\d*)',
            r'([\d,]+\.?\d{2})\s*(?:₦|ngn)',
        ]
        
        for pattern in ngn_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    clean_amount = match.replace(',', '')
                    amount = float(clean_amount)
                    amounts.append(amount)
                except ValueError:
                    continue
        
        # Look for common amount patterns in receipts
        amount_keywords = [
            'amount', 'total', 'sent', 'received', 'transfer',
            'balance', 'sum', 'value', 'payment'
        ]
        
        for keyword in amount_keywords:
            pattern = rf'{keyword}[\s:]*[\s₦ngn]*([\d,]+\.?\d*)'
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            for match in matches:
                try:
                    clean_amount = match.replace(',', '')
                    amount = float(clean_amount)
                    # Filter out unlikely amounts (phone numbers, dates, etc.)
                    if 100 <= amount <= 10000000:  # ₦100 to ₦10M
                        amounts.append(amount)
                except ValueError:
                    continue
        
        # Extract standalone numbers that look like amounts
        number_pattern = r'\b(\d{3,}[,\d]*\.?\d{2})\b'
        matches = re.findall(number_pattern, text)
        for match in matches:
            try:
                clean_num = match.replace(',', '')
                amount = float(clean_num)
                # Filter: must be reasonable for a transaction
                if 100 <= amount <= 10000000 and amount not in amounts:
                    amounts.append(amount)
            except ValueError:
                continue
        
        # Remove duplicates and sort
        unique_amounts = sorted(list(set(amounts)))
        
        # If we found multiple amounts, prioritize larger ones (transfers are usually significant)
        if len(unique_amounts) > 1:
            # Filter out amounts that look like phone numbers or IDs
            filtered_amounts = []
            for amount in unique_amounts:
                amount_str = str(int(amount)) if amount.is_integer() else str(amount)
                # Skip if it looks like a phone number (11 digits starting with 0)
                if len(amount_str) == 11 and amount_str.startswith('0'):
                    continue
                # Skip if it looks like a date (2026, 2024, etc.)
                if 2020 <= amount <= 2030:
                    continue
                filtered_amounts.append(amount)
            return filtered_amounts
        
        return unique_amounts

    def extract_dates_smart(self, text: str) -> List[datetime]:
        """
        Smart date extraction for Nigerian receipts
        """
        dates = []
        
        # Look for date patterns with time (common in receipts)
        datetime_patterns = [
            r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?)\b',
            r'\b(\d{4}[/\-]\d{1,2}[/\-]\d{1,2}\s+\d{1,2}:\d{2}(?::\d{2})?)\b',
        ]
        
        for pattern in datetime_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    date_obj = parse(match, fuzzy=True)
                    dates.append(date_obj)
                except:
                    continue
        
        # Look for date without time
        date_patterns = [
            r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{4})\b',
            r'\b(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b',
            r'\b(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4})\b',
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    date_obj = parse(match, fuzzy=True)
                    dates.append(date_obj)
                except:
                    continue
        
        # Look for "Jan 20th 2026" type patterns
        ordinal_pattern = r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?\s*,?\s*\d{4}\b'
        matches = re.findall(ordinal_pattern, text, re.IGNORECASE)
        for match in matches:
            try:
                # Remove ordinal suffixes
                cleaned = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', match)
                date_obj = parse(cleaned, fuzzy=True)
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

    def extract_transaction_info(self, text: str) -> Dict[str, Any]:
        """
        Extract transaction-specific information
        """
        result = {
            "transaction_id": None,
            "sender": None,
            "recipient": None,
            "bank_or_service": None,
        }
        
        # Extract transaction ID
        for pattern in self.transaction_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                result["transaction_id"] = matches[0]
                break
        
        # Extract sender/recipient info
        sender_patterns = [
            r'sender[\s:]*[\s]*(.+)',
            r'from[\s:]*[\s]*(.+)',
        ]
        
        recipient_patterns = [
            r'recipient[\s:]*[\s]*(.+)',
            r'to[\s:]*[\s]*(.+)',
            r'beneficiary[\s:]*[\s]*(.+)',
        ]
        
        for pattern in sender_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["sender"] = match.group(1).strip()[:100]
                break
        
        for pattern in recipient_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["recipient"] = match.group(1).strip()[:100]
                break
        
        # Detect bank/service
        services = ['opay', 'palmpay', 'paystack', 'flutterwave', 'moniepoint', 'gtbank', 'zenith', 'uba', 'firstbank']
        for service in services:
            if service in text.lower():
                result["bank_or_service"] = service.upper()
                break
        
        return result

    def process_file(
        self, 
        file_path: str, 
        expected_amount: float = None, 
        expected_date: str = None
    ) -> Dict[str, Any]:
        """
        Process a receipt file with improved Nigerian receipt parsing
        """
        try:
            start_time = datetime.now()
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # Extract text
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                extracted_text = self.extract_text_with_optimizations(file_path)
            elif file_ext == '.pdf':
                extracted_text = self.extract_text_from_pdf(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # Extract information
            extracted_amounts = self.extract_amounts_smart(extracted_text)
            extracted_dates = self.extract_dates_smart(extracted_text)
            transaction_info = self.extract_transaction_info(extracted_text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "file_path": file_path,
                "file_type": file_ext[1:],
                "text_extracted": bool(extracted_text.strip()),
                "extracted_text_sample": extracted_text[:500] + "..." if len(extracted_text) > 500 else extracted_text,
                "extracted_amounts": extracted_amounts,
                "extracted_dates": [d.strftime('%Y-%m-%d %H:%M:%S') for d in extracted_dates],
                "transaction_info": transaction_info,
                "processing_time_seconds": round(processing_time, 2),
                "processing_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "ocr_confidence": self.calculate_confidence(extracted_text, extracted_amounts, extracted_dates)
            }
            
            # Validate if expected values provided
            if expected_amount is not None and expected_date is not None:
                try:
                    expected_date_obj = datetime.strptime(expected_date, '%Y-%m-%d')
                    
                    # Find best amount match
                    best_amount_match = None
                    amount_difference = float('inf')
                    
                    for amount in extracted_amounts:
                        diff = abs(amount - expected_amount)
                        # 1% tolerance or ₦10
                        tolerance = min(expected_amount * 0.01, 10)
                        if diff <= tolerance and diff < amount_difference:
                            amount_difference = diff
                            best_amount_match = amount
                    
                    # Find best date match
                    best_date_match = None
                    date_difference = float('inf')
                    
                    for date in extracted_dates:
                        diff = abs((date - expected_date_obj).days)
                        if diff <= 1 and diff < date_difference:  # ±1 day tolerance
                            date_difference = diff
                            best_date_match = date
                    
                    validation_result = {
                        "is_valid": best_amount_match is not None and best_date_match is not None,
                        "amount_matches": best_amount_match is not None,
                        "date_matches": best_date_match is not None,
                        "matched_amount": best_amount_match,
                        "matched_date": best_date_match.strftime('%Y-%m-%d') if best_date_match else None,
                        "expected_amount": expected_amount,
                        "expected_date": expected_date,
                        "amount_difference": amount_difference if best_amount_match is None else 0,
                        "date_difference": date_difference if best_date_match is None else 0,
                        "validation_score": self.calculate_validation_score(
                            best_amount_match is not None,
                            best_date_match is not None,
                            len(extracted_amounts),
                            len(extracted_dates)
                        )
                    }
                    
                    result.update(validation_result)
                    
                except Exception as e:
                    logger.error(f"Validation error: {e}")
                    result["validation_error"] = str(e)
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {
                "file_path": file_path,
                "error": str(e),
                "success": False,
                "processing_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF with fallback to OCR"""
        try:
            text = ""
            
            # Try PyMuPDF first (fast for text-based PDFs)
            try:
                doc = fitz.open(pdf_path)
                for page in doc:
                    text += page.get_text()
                doc.close()
                
                if text.strip():
                    return text.strip()
            except:
                pass
            
            # Fallback to OCR for scanned PDFs
            images = convert_from_path(pdf_path)
            for i, image in enumerate(images):
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                    temp_path = temp_file.name
                    image.save(temp_path, 'PNG')
                
                page_text = self.extract_text_with_optimizations(temp_path)
                text += f"\n--- Page {i+1} ---\n{page_text}\n"
                
                os.unlink(temp_path)
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""

    def calculate_confidence(self, text: str, amounts: List[float], dates: List[datetime]) -> float:
        """Calculate OCR confidence score"""
        confidence = 0.0
        
        if text.strip():
            confidence += 0.3
        
        if amounts:
            confidence += 0.4
        
        if dates:
            confidence += 0.3
        
        # Bonus for having transaction info
        if any(keyword in text.lower() for keyword in ['transaction', 'receipt', 'successful', 'transfer']):
            confidence += 0.1
        
        return min(confidence, 1.0)

    def calculate_validation_score(self, amount_match: bool, date_match: bool, 
                                  num_amounts: int, num_dates: int) -> float:
        """Calculate validation score"""
        score = 0.0
        
        if amount_match:
            score += 0.5
        if date_match:
            score += 0.5
        
        # Bonus for multiple matches
        if num_amounts > 1:
            score += 0.1
        if num_dates > 1:
            score += 0.1
        
        return min(score, 1.0)


# Global instance
ocr_processor = ReceiptOCRProcessor()