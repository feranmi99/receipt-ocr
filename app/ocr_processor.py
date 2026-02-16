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
            r'(?:₦|NGN|Naira|N)[\s:]*[\s]*([\d,]+\.?\d*)',
            r'(?:amount|amt|total|sum|balance|transfer|sent|received)[\s:]*[\s₦NGN]*([\d,]+\.?\d*)',
            r'([\d,]+\.?\d{2})\s*(?:₦|NGN|naira)',
            r'(?:transaction|txn|trans)[\s:]*[\s]*([\d,]+\.?\d*)',
            r'#?([\d,]+\.?\d{2})\b',
        ]
        
        self.date_patterns = [
            r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b',
            r'\b(\d{4}[/\-]\d{1,2}[/\-]\d{1,2})\b',
            r'\b(\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4})\b',
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b',
            r'\b\d{1,2}[/\-]\d{1,2}[/\-]\d{4}\s+\d{1,2}:\d{2}(?::\d{2})?\b',
        ]
        
        self.transaction_patterns = [
            r'transaction\s*(?:no|number|id|#)[\s:]*[\s]*([a-zA-Z0-9\-]+)',
            r'txn\s*(?:no|number|id|#)[\s:]*[\s]*([a-zA-Z0-9\-]+)',
            r'ref\s*(?:no|number|id|#)[\s:]*[\s]*([a-zA-Z0-9\-]+)',
            r'\b([A-Z0-9]{10,24})\b',  # Long alphanumeric strings
        ]

        self.nigerian_banks = [
            'OPAY', 'PALMPAY', 'PAYSTACK', 'FLUTTERWAVE', 'MONIEPOINT', 
            'GTBANK', 'GUARANTY TRUST', 'ZENITH', 'UBA', 'UNITED BANK FOR AFRICA',
            'FIRSTBANK', 'FIRST BANK', 'KUDA', 'STANBIC', 'VFD', 'ACCESS BANK',
            'FIDELITY', 'WEMA', 'HERITAGE', 'UNION BANK', 'STERLING', 'POLARIS',
            'KEYSTONE', 'GLOBUS', 'TITAN', 'PROVIDUS', 'CARBON', 'FAIRMONEY',
            'PIGGYVEST', 'PARALLEX', 'LOTUS', 'LOTUS BANK', 'TAJBANK', 'TAJ BANK',
            'SUDIPAY', 'REGINA MFB', 'TEAMAPT', 'SPARKLE'
        ]

        self.status_keywords = {
            'SUCCESSFUL': ['successful', 'completed', 'paid', 'success', 'approved', 'done'],
            'FAILED': ['failed', 'declined', 'rejected', 'failed', 'unsuccessful', 'cancelled'],
            'PENDING': ['pending', 'processing', 'in-progress', 'submitted']
        }

    def _resize_image(self, image: np.ndarray, max_width: int = 1200) -> np.ndarray:
        """Resize image to a max width while maintaining aspect ratio"""
        height, width = image.shape[:2]
        if width > max_width:
            scaling_factor = max_width / float(width)
            new_size = (max_width, int(height * scaling_factor))
            return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return image

    def _crop_to_text(self, image: np.ndarray) -> np.ndarray:
        """Crop image to the region containing text by removing white/empty margins"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Binary threshold to find text regions
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            
            # Find all non-zero points
            coords = cv2.findNonZero(thresh)
            if coords is not None:
                x, y, w, h = cv2.boundingRect(coords)
                # Add a small padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                return image[y:y+h, x:x+w]
        except Exception as e:
            logger.error(f"Error cropping image: {e}")
        return image

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
            
            # Check if it's likely a digital screenshot
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
        if re.search(r'(?:₦|NGN|naira|n)\s*[\d,]+', text, re.IGNORECASE):
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

    def _ocr_with_confidence_filter(self, pil_img: Image, config: str) -> str:
        """Run OCR and ignore words with confidence < 40"""
        data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)
        words = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) >= 40:
                words.append(data['text'][i])
        return " ".join(words)

    def extract_text_with_optimizations(self, image_path: str) -> str:
        """
        Optimized text extraction with early exit, progressive preprocessing,
        and strict stage-based strategy.
        """
        start_time = time.time()
        timeout = 15.0
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Cannot read image: {image_path}")
            
            # --- PREPARATION ---
            # 1. Resize before anything else
            img = self._resize_image(img, max_width=1200)
            
            # 2. Crop to text region
            img = self._crop_to_text(img)
            
            characteristics = self._get_image_characteristics(img)
            logger.info(f"Image characteristics: {characteristics}")
            
            is_photo = not characteristics.get("is_likely_digital", True)
            
            best_text = ""
            best_confidence = 0.0

            # --- STAGE 1: FAST PATH ---
            logger.info("Executing STAGE 1: FAST PATH")
            processed_1 = self.preprocess_image_progressive(img, level=1)
            text_1 = self._ocr_with_confidence_filter(Image.fromarray(processed_1), config='--oem 3 --psm 6')
            conf_1 = self._calculate_ocr_confidence(text_1)
            
            if conf_1 >= 0.6:
                logger.info(f"Early exit at Stage 1. Confidence: {conf_1:.2f}")
                return self.clean_extracted_text(text_1)
            
            best_text, best_confidence = text_1, conf_1

            # --- STAGE 2: MODERATE PATH ---
            logger.info("Executing STAGE 2: MODERATE PATH")
            processed_2 = self.preprocess_image_progressive(img, level=2)
            # Try PSM 6 first
            text_2 = self._ocr_with_confidence_filter(Image.fromarray(processed_2), config='--oem 3 --psm 6')
            conf_2 = self._calculate_ocr_confidence(text_2)
            
            if conf_2 > best_confidence:
                best_text, best_confidence = text_2, conf_2
            
            if best_confidence >= 0.6:
                logger.info(f"Early exit at Stage 2 (PSM 6). Confidence: {best_confidence:.2f}")
                return self.clean_extracted_text(best_text)

            # Fallback to PSM 3 (once only)
            text_2_psm3 = self._ocr_with_confidence_filter(Image.fromarray(processed_2), config='--oem 3 --psm 3')
            conf_2_psm3 = self._calculate_ocr_confidence(text_2_psm3)
            
            if conf_2_psm3 > best_confidence:
                best_text, best_confidence = text_2_psm3, conf_2_psm3

            if best_confidence >= 0.6:
                logger.info(f"Early exit at Stage 2 (PSM 3). Confidence: {best_confidence:.2f}")
                return self.clean_extracted_text(best_text)

            # --- STAGE 3: SLOW PATH (LAST RESORT) ---
            # Only if photo receipt or very low confidence
            if is_photo or best_confidence < 0.4:
                logger.info("Executing STAGE 3: SLOW PATH")
                processed_3 = self.preprocess_image_progressive(img, level=3)
                text_3 = self._ocr_with_confidence_filter(Image.fromarray(processed_3), config='--oem 3 --psm 6')
                conf_3 = self._calculate_ocr_confidence(text_3)
                
                if conf_3 > best_confidence:
                    best_text, best_confidence = text_3, conf_3
            
            return self.clean_extracted_text(best_text)
            
        except Exception as e:
            logger.error(f"Error extracting text from image {image_path}: {e}")
            return ""

    def clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text for Nigerian context"""
        if not text:
            return ""
            
        # Replace multiple spaces/newlines with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Specific cleanup for Nigerian receipt artifacts
        replacements = {
            '0Pay': 'OPay', '0pay': 'OPay', 'QPay': 'OPay',
            'Naira': 'NGN', 'naira': 'NGN', '₦': 'NGN', 
            '#': 'NGN', # Often # is misread for ₦
            'transction': 'transaction', 'transacation': 'transaction',
            'sucessful': 'successful', 'reciept': 'receipt',
            'recieved': 'received', 'beneficary': 'beneficiary',
            'amt': 'amount', 'txn': 'transaction', 'ref': 'reference'
        }
        
        for wrong, correct in replacements.items():
            text = text.replace(wrong, correct)
            
        # Handle 'N' if it's likely a currency symbol (followed by digit)
        text = re.sub(r'\bN\s?(\d)', r'NGN \1', text)
        
        return text.strip()

    def extract_amounts_smart(self, text: str) -> List[float]:
        """
        Smart amount extraction for Nigerian receipts.
        Valid range: ₦100 – ₦10,000,000.
        Ignores phone numbers, dates, and ref numbers.
        """
        amounts = []
        text_lower = text.lower()
        
        # 1. Matches with currency markers
        currency_patterns = [
            r'(?:₦|NGN|naira|n)\s*([\d,]+\.?\d*)',
            r'([\d,]+\.?\d{2})\s*(?:₦|ngn|naira)',
        ]
        
        for p in currency_patterns:
            for match in re.findall(p, text_lower):
                try:
                    val = float(match.replace(',', ''))
                    if 100 <= val <= 10000000:
                        amounts.append(val)
                except: continue

        # 2. Key-word based matches
        amount_keywords = ['amount', 'total', 'sent', 'received', 'transfer', 'balance', 'sum', 'value', 'paid']
        for kw in amount_keywords:
            p = rf'{kw}[\s:]*[\s₦ngn]*([\d,]+\.?\d*)'
            for match in re.findall(p, text_lower):
                try:
                    val = float(match.replace(',', ''))
                    if 100 <= val <= 10000000:
                        amounts.append(val)
                except: continue

        # 3. Floating points that look like currency (e.g. 5,000.00)
        for match in re.findall(r'\b(\d{1,3}(?:,\d{3})*\.\d{2})\b', text):
            try:
                val = float(match.replace(',', ''))
                if 100 <= val <= 10000000:
                    amounts.append(val)
            except: continue

        # Filter out common false positives (like 11-digit phone numbers starting with 0)
        unique_amounts = []
        for a in set(amounts):
            a_str = f"{int(a)}"
            if len(a_str) == 11 and a_str.startswith('0'):
                continue
            # Skip if it's likely a year (2020-2030)
            if 2020 <= a <= 2030:
                continue
            unique_amounts.append(a)
            
        return sorted(unique_amounts)

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
        Extract transaction-specific information with Nigerian intelligence
        """
        result = {
            "transaction_id": None,
            "sender": None,
            "recipient": None,
            "bank_or_service": None,
            "status": "UNKNOWN",
        }
        
        text_lower = text.lower()
        
        # 1. Transaction ID / Reference
        for pattern in self.transaction_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["transaction_id"] = match.group(1).strip()
                break
        
        # 2. Sender / Recipient
        sender_p = [
            r'sender(?:[\s-]name)?[\s:]*[\s]*(.+)', 
            r'from[\s:]*[\s]*(.+)', 
            r'account[\s-]?holder[\s:]*[\s]*(.+)', 
            r'source[\s:]*[\s]*(.+)',
            r'sender[\s:]*([A-Z\s]{3,})'
        ]
        recipient_p = [
            r'recipient(?:[\s-]name)?[\s:]*[\s]*(.+)', 
            r'to[\s:]*[\s]*(.+)', 
            r'beneficiary(?:[\s-]name)?[\s:]*[\s]*(.+)', 
            r'destination[\s:]*[\s]*(.+)',
            r'beneficiary[\s:]*([A-Z\s]{3,})'
        ]
        
        for p in sender_p:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                # Clean up if it captures too much (e.g. includes 'Amount')
                val = re.split(r'amount|date|bank|ref|txn|time|success', val, flags=re.IGNORECASE)[0]
                result["sender"] = val.strip()[:60]
                break
        for p in recipient_p:
            m = re.search(p, text, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                val = re.split(r'amount|date|bank|ref|txn|time|success', val, flags=re.IGNORECASE)[0]
                result["recipient"] = val.strip()[:60]
                break
        
        # 3. Status Detection
        for status, keywords in self.status_keywords.items():
            if any(kw in text_lower for kw in keywords):
                result["status"] = status
                break
        
        # 4. Bank / Service Detection
        for bank in self.nigerian_banks:
            if bank.lower() in text_lower:
                result["bank_or_service"] = bank
                break
        
        return result

    def process_file(
        self, 
        file_path: str, 
        expected_amount: float = None, 
        expected_date: str = None
    ) -> Dict[str, Any]:
        """
        Process a receipt file and return a structured object as per requirements.
        """
        start_time = time.time()
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # --- TEXT EXTRACTION ---
            if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                extracted_text = self.extract_text_with_optimizations(file_path)
            elif file_ext == '.pdf':
                extracted_text = self.extract_text_from_pdf(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            # --- DATA EXTRACTION ---
            amounts = self.extract_amounts_smart(extracted_text)
            dates = self.extract_dates_smart(extracted_text)
            info = self.extract_transaction_info(extracted_text)
            confidence = self.calculate_confidence(extracted_text, amounts, dates)
            
            # --- METADATA ---
            metadata = {"path": file_path, "type": file_ext[1:]}
            if file_ext in ['.jpg', '.jpeg', '.png']:
                img = cv2.imread(file_path)
                if img is not None:
                    metadata["characteristics"] = self._get_image_characteristics(img)

            # --- VALIDATION ---
            validation = {"is_valid": False, "matched_amount": None, "matched_date": None, "score": 0.0}
            if expected_amount and expected_date:
                try:
                    exp_date = datetime.strptime(expected_date, '%Y-%m-%d').date()
                    
                    # Match Amount (±2%)
                    for a in amounts:
                        if abs(a - expected_amount) <= (expected_amount * 0.02):
                            validation["matched_amount"] = a
                            break
                    
                    # Match Date (±1 day)
                    for d in dates:
                        if abs((d.date() - exp_date).days) <= 1:
                            validation["matched_date"] = d.strftime('%Y-%m-%d')
                            break
                    
                    validation["is_valid"] = validation["matched_amount"] is not None and validation["matched_date"] is not None
                    validation["score"] = self.calculate_validation_score(
                        validation["matched_amount"] is not None,
                        validation["matched_date"] is not None,
                        len(amounts), len(dates)
                    )
                except Exception as e:
                    logger.error(f"Validation error: {e}")

            processing_time = time.time() - start_time
            
            return {
                "file_metadata": metadata,
                "ocr_confidence": round(confidence, 2),
                "extracted_data": {
                    "amount": validation["matched_amount"] if validation["matched_amount"] else (amounts[0] if amounts else None),
                    "date": validation["matched_date"] if validation["matched_date"] else (dates[0].strftime('%Y-%m-%d') if dates else None),
                    "transaction_reference": info["transaction_id"],
                    "sender": info["sender"],
                    "recipient": info["recipient"],
                    "bank": info["bank_or_service"],
                    "status": info["status"],
                    "all_amounts_found": amounts,
                    "all_dates_found": [d.strftime('%Y-%m-%d') for d in dates]
                },
                "validation_result": validation if expected_amount else None,
                "performance": {
                    "processing_time_seconds": round(processing_time, 2),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Critical error processing file {file_path}: {e}")
            return {
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