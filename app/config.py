import os
import platform

class Config:
    """Configuration for OCR API"""
    
    # Environment settings
    UPLOAD_FOLDER = "uploads"
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max file size
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
    
    @staticmethod
    def get_tesseract_path():
        """Get Tesseract path based on platform"""
        system = platform.system().lower()
        
        if system == "windows":
            # Windows paths
            paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
            ]
            for path in paths:
                if os.path.exists(path):
                    return path
            return "tesseract"  # Fallback to PATH
        else:
            # Linux/Unix/Mac - assumes tesseract is in PATH
            return "tesseract"
    
    @staticmethod
    def get_poppler_path():
        """Get Poppler path for PDF processing"""
        system = platform.system().lower()
        
        if system == "windows":
            return r"C:\poppler\Library\bin" if os.path.exists(r"C:\poppler\Library\bin") else None
        else:
            # For Linux deployment
            return "/usr/bin"  # Common location
    
    @staticmethod
    def setup_environment():
        """Setup environment variables"""
        poppler_path = Config.get_poppler_path()
        if poppler_path and os.path.exists(poppler_path):
            os.environ['poppler_path'] = poppler_path