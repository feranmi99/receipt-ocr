import os
import re
import logging
import time
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

    def _get_image_characteristics(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image to determine optimal processing strategy"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate brightness and contrast
            mean_brightness = np.mean(gray)
            std_brightness = np.std(gray) # Contrast
            
            # Estimate noise using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Estimate text density (percentage of edges)
            edges = cv2.Canny(gray, 100, 200)
            text_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            
            # Check if it's likely a digital screenshot (low noise, high sharpness, uniform areas)
            # Screenshots usually have very high sharpness and distinct edges
            is_digital = bool(laplacian_var > 400 and text_density < 0.05)
            
            return {
                "mean_brightness": float(mean_brightness),
                "contrast": float(std_brightness),
                "sharpness": float(laplacian_var),
                "text_density": float(text_density),
                "is_likely_digital": is_digital,
                "is_low_light": bool(mean_brightness < 80),
                "height": int(image.shape[0]),
                "width": int(image.shape[1])
            }
        except Exception as e:
            logger.error(f"Error analyzing image: {e}")
            return {"is_likely_digital": False}

    def _calculate_ocr_confidence(self, text: str) -> float:
        """Determine if OCR result is reliable based on key fields presence"""
        if not text or len(text.strip()) < 10:
            return 0.0
        
        score = 0.0
        text_lower = text.lower()
        
        # Check for Nigerian Financial Keywords (Weight: 0.4)
        keywords = ['transaction', 'successful', 'amount', 'ngn', '₦', 'sender', 'recipient', 'beneficiary', 'ref', 'terminal', 'merchant', 'approved']
        matches = sum(1 for kw in keywords if kw in text_lower)
        score += min((matches / (len(keywords) * 0.4)) * 0.4, 0.4)
        
        # Check for Transaction ID/Ref patterns (Weight: 0.3)
        if any(re.search(p, text, re.IGNORECASE) for p in self.transaction_patterns):
            score += 0.3
            
        # Check for Amount patterns (Weight: 0.2)
        # Look for ₦ or NGN followed by digits
        if re.search(r'(?:₦|NGN|naira)\s*[\d,]+', text, re.IGNORECASE):
            score += 0.2
        elif re.search(r'[\d,]+\.\d{2}', text):
            score += 0.1
            
        # Structural check: does it look like a receipt? (Weight: 0.1)
        if len(text.splitlines()) > 5:
            score += 0.1
            
        return min(score, 1.0)

    def preprocess_image_progressive(self, image: np.ndarray, level: int = 1) -> np.ndarray:
        """
        Progressive preprocessing levels optimized for speed:
        Level 1: Grayscale + Otsu threshold (Fastest, best for screenshots)
        Level 2: Level 1 + Gaussian Blur (Good for moderately noisy images)
        Level 3: Level 2 + Bilateral Filter (Slowest, for noisy/poor photos)
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            if level == 1:
                # Grayscale + Otsu threshold
                _, processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                return processed

            if level == 2:
                # Gaussian Blur + Adaptive Threshold
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                processed = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
                return processed

            if level == 3:
                # Bilateral Filter + Adaptive Threshold + Morphology
                # Only use if confidence is low after Level 1 & 2
                filtered = cv2.bilateralFilter(gray, 9, 75, 75)
                processed = cv2.adaptiveThreshold(
                    filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 11, 2
                )
                
                kernel = np.ones((1, 1), np.uint8)
                processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
                return processed
            
            return gray
            
        except Exception as e:
            logger.error(f"Error in progressive preprocessing: {e}")
            return image

    def extract_text_with_optimizations(self, image_path: str) -> str:
        """
        Optimized text extraction with early exit, progressive preprocessing,
        caching, and timeout protection.
        """
        start_time = time.time()
        timeout = 15.0 # 15 seconds max logic
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            characteristics = self._get_image_characteristics(img)
            logger.info(f"Image characteristics: {characteristics}")
            
            final_text = ""
            best_confidence = 0.0
            
            # Adaptive PSM selection: PSM 6 is usually best for receipts
            # We focus on PSM 6 first, only try others if needed
            psms = [6, 3, 11]
            
            # Preprocessed images cache to avoid re-calculating
            preprocessed_cache = {}
            
            # Progressive levels
            # Level 1: Fast/Screen, Level 2: Moderate, Level 3: Slow/Noisy
            levels = [1, 2, 3]
            
            # Start with digital assumption if likely
            if not characteristics.get("is_likely_digital") and characteristics.get("contrast", 0) < 40:
                levels = [2, 1, 3] # Start with moderate if low contrast

            for level in levels:
                # Check timeout
                if time.time() - start_time > timeout:
                    logger.warning(f"Timeout reached. Returning best result with confidence {best_confidence}")
                    break
                    
                # Get or create preprocessed image
                if level not in preprocessed_cache:
                    preprocessed_cache[level] = self.preprocess_image_progressive(img, level=level)
                
                processed_img = preprocessed_cache[level]
                pil_img = Image.fromarray(processed_img)
                
                # 1. First try PSM 6 (fastest and most reliable for receipts)
                config_6 = f'--oem 3 --psm 6'
                text_6 = pytesseract.image_to_string(pil_img, config=config_6)
                conf_6 = self._calculate_ocr_confidence(text_6)
                
                if conf_6 > best_confidence:
                    best_confidence = conf_6
                    final_text = text_6
                
                # Early exit if we have found high-confidence results
                if best_confidence >= 0.8:
                    logger.info(f"Early exit at Level {level}, PSM 6. Confidence: {best_confidence:.2f}")
                    return self.clean_extracted_text(final_text)
                
                # 2. Try other PSMs only if confidence is still low and it's Level 1 or 2
                if best_confidence < 0.5:
                    for psm in [3, 11]:
                        if time.time() - start_time > timeout:
                            break
                            
                        config = f'--oem 3 --psm {psm}'
                        text = pytesseract.image_to_string(pil_img, config=config)
                        conf = self._calculate_ocr_confidence(text)
                        
                        if conf > best_confidence:
                            best_confidence = conf
                            final_text = text
                        
                        if best_confidence >= 0.8:
                            logger.info(f"Early exit at Level {level}, PSM {psm}. Confidence: {best_confidence:.2f}")
                            return self.clean_extracted_text(final_text)

            return self.clean_extracted_text(final_text)
            
        except Exception as e:
            logger.error(f"Error extracting text from image {image_path}: {e}")
            return ""

    def clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        # Replace multiple spaces/newlines with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Specific cleanup for Nigerian receipt artifacts
        replacements = {
            '0Pay': 'OPay', '0pay': 'OPay', 'QPay': 'OPay',
            'Naira': 'NGN', 'naira': 'NGN', '₦': 'NGN', # Normalize currency
            'transction': 'transaction', 'transacation': 'transaction',
            'sucessful': 'successful', 'reciept': 'receipt',
            'recieved': 'received', 'beneficary': 'beneficiary'
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
            "status": "UNKNOWN",
        }
        
        text_lower = text.lower()
        
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
        
        # Detect status
        if any(kw in text_lower for kw in ['successful', 'success', 'completed', 'paid']):
            result["status"] = "SUCCESSFUL"
        elif any(kw in text_lower for kw in ['failed', 'declined', 'rejected']):
            result["status"] = "FAILED"
        elif any(kw in text_lower for kw in ['pending', 'processing']):
            result["status"] = "PENDING"
        
        # Detect bank/service
        services = ['opay', 'palmpay', 'paystack', 'flutterwave', 'moniepoint', 'gtbank', 'zenith', 'uba', 'firstbank', 'kuda', 'vfd', 'stanbic']
        for service in services:
            if service in text_lower:
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
            
            # Analyze image characteristics if it's an image
            metadata = {}
            if file_ext in ['.jpg', '.jpeg', '.png']:
                img = cv2.imread(file_path)
                if img is not None:
                    metadata = self._get_image_characteristics(img)

            result = {
                "file_metadata": {
                    "path": file_path,
                    "type": file_ext[1:],
                    "characteristics": metadata
                },
                "extraction_summary": {
                    "text_found": bool(extracted_text.strip()),
                    "amounts_count": len(extracted_amounts),
                    "dates_count": len(extracted_dates),
                    "confidence_score": self.calculate_confidence(extracted_text, extracted_amounts, extracted_dates)
                },
                "data": {
                    "amounts": extracted_amounts,
                    "dates": [d.strftime('%Y-%m-%d %H:%M:%S') for d in extracted_dates],
                    "transaction_info": transaction_info,
                    "raw_text_preview": extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text
                },
                "performance": {
                    "processing_time_seconds": round(processing_time, 2),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            # Additional validation logic
            if expected_amount is not None and expected_date is not None:
                try:
                    expected_date_obj = datetime.strptime(expected_date, '%Y-%m-%d')
                    
                    # Match Amount
                    best_amount_match = None
                    min_diff = float('inf')
                    for amount in extracted_amounts:
                        diff = abs(amount - expected_amount)
                        if diff < min_diff:
                            min_diff = diff
                            best_amount_match = amount
                    
                    # Match Date
                    best_date_match = None
                    min_date_diff = float('inf')
                    for date in extracted_dates:
                        days_diff = abs((date.date() - expected_date_obj.date()).days)
                        if days_diff < min_date_diff:
                            min_date_diff = days_diff
                            best_date_match = date

                    # Detailed Validation Response
                    result["validation"] = {
                        "is_valid": min_diff <= (expected_amount * 0.02) and min_date_diff <= 1,
                        "checks": {
                            "amount_match": {
                                "status": min_diff <= (expected_amount * 0.02),
                                "expected": expected_amount,
                                "found": best_amount_match,
                                "difference": round(min_diff, 2)
                            },
                            "date_match": {
                                "status": min_date_diff <= 1,
                                "expected": expected_date,
                                "found": best_date_match.strftime('%Y-%m-%d') if best_date_match else None,
                                "days_difference": min_date_diff
                            }
                        },
                        "score": self.calculate_validation_score(
                            min_diff <= (expected_amount * 0.02),
                            min_date_diff <= 1,
                            len(extracted_amounts),
                            len(extracted_dates)
                        )
                    }
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