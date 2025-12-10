import os
import cv2
import pytesseract
import re
import json
import numpy as np
from PIL import Image, ImageEnhance
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Configure Tesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'<path_to_tesseract>'

class ImagePreprocessor:
    """Handles image preprocessing for better OCR results."""
    
    @staticmethod
    def preprocess_image(image_path: str) -> np.ndarray:
        """
        Preprocess the image to enhance text recognition.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image as numpy array
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to get binary image
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove noise
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Apply dilation to connect text components
        kernel = np.ones((2, 2), np.uint8)
        img_dilation = cv2.dilate(opening, kernel, iterations=1)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(img_dilation)
        
        return enhanced
    
    @staticmethod
    def deskew(image: np.ndarray) -> np.ndarray:
        """
        Deskew the image to correct any tilt.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Deskewed image
        """
        coords = np.column_stack(np.where(image > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
            
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated


class PIIDetector:
    """Detects and handles PII in extracted text."""
    
    # Regular expressions for common PII patterns
    PII_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b',
        'ssn': r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b',
        'credit_card': r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b',
        'date_of_birth': r'\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d{2}\b',
    }
    
    @classmethod
    def detect_pii(cls, text: str) -> Dict[str, List[str]]:
        """
        Detect PII in the given text.
        
        Args:
            text: Text to analyze for PII
            
        Returns:
            Dictionary mapping PII types to lists of detected values
        """
        pii_found = {}
        
        for pii_type, pattern in cls.PII_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                pii_found[pii_type] = list(set(matches))  # Remove duplicates
                
        return pii_found
    
    @staticmethod
    def redact_text(text: str, pii_matches: Dict[str, List[str]]) -> str:
        """
        Redact PII in the text.
        
        Args:
            text: Original text
            pii_matches: Dictionary of PII types and their matches
            
        Returns:
            Text with PII redacted
        """
        redacted_text = text
        
        for pii_type, matches in pii_matches.items():
            for match in matches:
                redacted_text = redacted_text.replace(match, f'[REDACTED_{pii_type.upper()}]')
                
        return redacted_text


class OCRProcessor:
    """Handles OCR processing of images."""
    
    @staticmethod
    def extract_text(image: np.ndarray) -> str:
        """
        Extract text from image using Tesseract OCR.
        
        Args:
            image: Preprocessed image as numpy array
            
        Returns:
            Extracted text
        """
        # Use Tesseract to extract text
        custom_config = r'--oem 3 --psm 6'  # PSM 6: Assume a single uniform block of text
        text = pytesseract.image_to_string(image, config=custom_config)
        
        return text
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize the extracted text.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace and normalize newlines
        cleaned = ' '.join(text.split())
        # Remove non-printable characters
        cleaned = ''.join(char for char in cleaned if char.isprintable() or char.isspace())
        return cleaned


def process_document(image_path: str, output_dir: str) -> Dict:
    """
    Process a single document through the OCR and PII detection pipeline.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save output files
        
    Returns:
        Dictionary containing processing results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize results dictionary
    result = {
        'filename': os.path.basename(image_path),
        'preprocessing': {},
        'ocr': {},
        'pii_detection': {}
    }
    
    try:
        # Step 1: Preprocess the image
        preprocessor = ImagePreprocessor()
        preprocessed_img = preprocessor.preprocess_image(image_path)
        
        # Step 2: Deskew the image
        deskewed_img = preprocessor.deskew(preprocessed_img)
        
        # Save preprocessed image
        output_image_path = os.path.join(output_dir, f'preprocessed_{os.path.basename(image_path)}')
        cv2.imwrite(output_image_path, deskewed_img)
        result['preprocessing']['output_image'] = output_image_path
        
        # Step 3: Perform OCR
        ocr_processor = OCRProcessor()
        raw_text = ocr_processor.extract_text(deskewed_img)
        cleaned_text = ocr_processor.clean_text(raw_text)
        
        result['ocr'] = {
            'raw_text': raw_text,
            'cleaned_text': cleaned_text,
            'character_count': len(cleaned_text),
            'word_count': len(cleaned_text.split())
        }
        
        # Step 4: Detect and handle PII
        pii_detector = PIIDetector()
        pii_matches = pii_detector.detect_pii(cleaned_text)
        redacted_text = pii_detector.redact_text(cleaned_text, pii_matches)
        
        result['pii_detection'] = {
            'pii_found': pii_matches,
            'redacted_text': redacted_text,
            'has_pii': len(pii_matches) > 0
        }
        
        # Save results to JSON
        output_json_path = os.path.join(output_dir, f'results_{Path(image_path).stem}.json')
        with open(output_json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Save redacted text to file
        output_txt_path = os.path.join(output_dir, f'redacted_{Path(image_path).stem}.txt')
        with open(output_txt_path, 'w') as f:
            f.write(redacted_text)
        
        result['output_files'] = {
            'preprocessed_image': output_image_path,
            'results_json': output_json_path,
            'redacted_text': output_txt_path
        }
        
        return result
        
    except Exception as e:
        error_msg = f"Error processing {image_path}: {str(e)}"
        print(error_msg)
        result['error'] = error_msg
        return result


def process_directory(input_dir: str, output_dir: str) -> List[Dict]:
    """
    Process all images in the input directory.
    
    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save output files
        
    Returns:
        List of processing results for each image
    """
    # Get all jpg/jpeg files in the input directory
    image_extensions = ('.jpg', '.jpeg', '.png')
    image_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.lower().endswith(image_extensions)
    ]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return []
    
    # Process each image
    results = []
    for image_file in image_files:
        print(f"\nProcessing {os.path.basename(image_file)}...")
        result = process_document(image_file, output_dir)
        results.append(result)
        
        # Print summary
        print(f"  - Extracted {result['ocr']['word_count']} words")
        if result['pii_detection']['has_pii']:
            print(f"  - Found PII: {', '.join(result['pii_detection']['pii_found'].keys())}")
    
    return results


if __name__ == "__main__":
    # Set input and output directories
    INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'handwritten_test')
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    
    print(f"Starting OCR + PII Extraction Pipeline")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Process all images in the input directory
    results = process_directory(INPUT_DIR, OUTPUT_DIR)
    
    print("\nProcessing complete!")
    print(f"Processed {len(results)} documents.")
    
    # Print summary of PII found
    total_pii = sum(1 for r in results if r.get('pii_detection', {}).get('has_pii', False))
    if total_pii > 0:
        print(f"Found PII in {total_pii} out of {len(results)} documents.")
