"""
Document AI Module for fishstick

Comprehensive document understanding and processing with:
- OCR (Optical Character Recognition): CRNN, TransformerOCR, EasyOCR, Tesseract
- Document Layout Analysis: LayoutLM, LayoutParser, Table/Form detection
- Document Classification: Type classification, quality checking
- Information Extraction: NER, key-value pairs, receipt/invoice parsing
- Document Enhancement: Dewarping, denoising, binarization, super-resolution
- Text Detection: EAST, CRAFT, DBNet
- Utilities: Dataset, visualizer, PDF processing, OCR post-processing
"""

from typing import Tuple, List, Optional, Union, Callable, Dict, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum, auto
from abc import ABC, abstractmethod
import warnings
import re
from collections import defaultdict

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import cv2


# ============================================================================
# Type Definitions
# ============================================================================

class DocumentType(Enum):
    """Document type categories."""
    INVOICE = auto()
    RECEIPT = auto()
    CONTRACT = auto()
    RESUME = auto()
    FORM = auto()
    REPORT = auto()
    LETTER = auto()
    ID_CARD = auto()
    PASSPORT = auto()
    BANK_STATEMENT = auto()
    MEDICAL_RECORD = auto()
    UNKNOWN = auto()


@dataclass
class TextBox:
    """Represents a text region with bounding box."""
    text: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float = 1.0
    angle: float = 0.0
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of bounding box."""
        return ((self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2)
    
    @property
    def width(self) -> int:
        """Get width of bounding box."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        """Get height of bounding box."""
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> int:
        """Get area of bounding box."""
        return self.width * self.height


@dataclass
class DocumentRegion:
    """Represents a document region (table, form field, etc.)."""
    region_type: str
    bbox: Tuple[int, int, int, int]
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    content: Optional[Any] = None


@dataclass
class KeyValuePair:
    """Key-value pair extracted from document."""
    key: str
    value: str
    key_bbox: Optional[Tuple[int, int, int, int]] = None
    value_bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: float = 1.0


@dataclass
class ParsedDocument:
    """Complete parsed document with all extracted information."""
    text_boxes: List[TextBox]
    regions: List[DocumentRegion]
    key_values: List[KeyValuePair]
    document_type: DocumentType
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# OCR (Optical Character Recognition)
# ============================================================================

class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for text recognition.
    
    Combines CNN feature extraction with bidirectional LSTM for
    sequence modeling, enabling end-to-end text recognition.
    
    Args:
        num_classes: Number of character classes
        input_channels: Number of input channels (default: 1 for grayscale)
        hidden_size: LSTM hidden dimension
        num_layers: Number of LSTM layers
    
    Example:
        >>> model = CRNN(num_classes=37)  # 26 letters + 10 digits + blank
        >>> x = torch.randn(1, 1, 32, 100)  # batch, channels, height, width
        >>> output = model(x)  # [T, batch, num_classes]
    """
    
    def __init__(
        self,
        num_classes: int,
        input_channels: int = 1,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        
        # CNN feature extraction (VGG-like backbone)
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # /2
            
            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # /4
            
            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), padding=(0, 1)),  # /8, w/2
            
            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), (2, 1), padding=(0, 1)),  # /16, w/4
            
            # Block 5
            nn.Conv2d(512, 512, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Calculate feature dimension
        self.feature_dim = 512
        
        # Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=False
        )
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, channels, height, width]
        
        Returns:
            Output tensor [time, batch, num_classes]
        """
        # CNN feature extraction
        conv = self.cnn(x)  # [batch, 512, H', W']
        
        # Reshape for RNN: [W', batch, 512 * H']
        batch, channels, height, width = conv.size()
        conv = conv.permute(3, 0, 2, 1)  # [W', batch, H', 512]
        conv = conv.reshape(width, batch, -1)  # [W', batch, features]
        
        # LSTM sequence modeling
        rnn_output, _ = self.rnn(conv)  # [W', batch, hidden*2]
        
        # Output projection
        output = self.fc(rnn_output)  # [W', batch, num_classes]
        
        return output
    
    def decode(self, output: Tensor, alphabet: str, blank_idx: int = 0) -> List[str]:
        """
        Decode CTC output to text.
        
        Args:
            output: Model output [time, batch, num_classes]
            alphabet: String of characters
            blank_idx: Index of blank label
        
        Returns:
            List of decoded strings
        """
        batch_size = output.size(1)
        predictions = output.argmax(dim=2)  # [time, batch]
        
        texts = []
        for b in range(batch_size):
            seq = predictions[:, b].cpu().numpy()
            
            # CTC decoding: merge repeats and remove blanks
            decoded = []
            prev_idx = -1
            for idx in seq:
                if idx != blank_idx and idx != prev_idx:
                    if 0 <= idx - 1 < len(alphabet):
                        decoded.append(alphabet[idx - 1])
                prev_idx = idx
            
            texts.append(''.join(decoded))
        
        return texts


class TransformerOCR(nn.Module):
    """
    Vision Transformer-based OCR model.
    
    Uses Vision Transformer for feature extraction and
    transformer decoder for sequence generation.
    
    Args:
        img_size: Input image size
        patch_size: Patch size for ViT
        num_classes: Number of character classes
        embed_dim: Embedding dimension
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        num_heads: Number of attention heads
    
    Example:
        >>> model = TransformerOCR(num_classes=37, img_size=(32, 100))
        >>> x = torch.randn(1, 1, 32, 100)
        >>> output = model(x, max_length=25)
    """
    
    def __init__(
        self,
        num_classes: int,
        img_size: Tuple[int, int] = (32, 100),
        patch_size: int = 4,
        embed_dim: int = 256,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_seq_len = max_seq_len
        self.patch_size = patch_size
        
        # Calculate number of patches
        self.img_h, self.img_w = img_size
        self.num_patches_h = self.img_h // patch_size
        self.num_patches_w = self.img_w // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            1, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Positional encoding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Token embedding for decoder
        self.token_embed = nn.Embedding(num_classes, embed_dim)
        self.token_pos_embed = nn.Parameter(
            torch.zeros(1, max_seq_len, embed_dim)
        )
        nn.init.trunc_normal_(self.token_pos_embed, std=0.02)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, num_classes)
        
        # Special tokens
        self.start_token = num_classes - 2
        self.end_token = num_classes - 1
        
    def forward(
        self,
        x: Tensor,
        targets: Optional[Tensor] = None,
        max_length: int = 25
    ) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [batch, 1, height, width]
            targets: Target sequences for teacher forcing [batch, seq_len]
            max_length: Maximum decoding length
        
        Returns:
            Output logits [batch, seq_len, num_classes]
        """
        batch_size = x.size(0)
        
        # Encode image
        memory = self.encode(x)  # [batch, num_patches, embed_dim]
        
        # Decode sequence
        if self.training and targets is not None:
            # Teacher forcing
            output = self.decode_teacher_forcing(memory, targets)
        else:
            # Autoregressive decoding
            output = self.decode_autoregressive(memory, max_length)
        
        return output
    
    def encode(self, x: Tensor) -> Tensor:
        """Encode image to feature sequence."""
        # Patch embedding
        x = self.patch_embed(x)  # [batch, embed_dim, h', w']
        x = x.flatten(2).transpose(1, 2)  # [batch, num_patches, embed_dim]
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Transformer encoding
        x = self.encoder(x)
        
        return x
    
    def decode_teacher_forcing(self, memory: Tensor, targets: Tensor) -> Tensor:
        """Decode with teacher forcing."""
        batch_size, seq_len = targets.size()
        
        # Embed target tokens
        tgt = self.token_embed(targets)  # [batch, seq_len, embed_dim]
        tgt = tgt + self.token_pos_embed[:, :seq_len, :]
        
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(targets.device)
        
        # Decode
        output = self.decoder(tgt, memory, tgt_mask=mask)
        output = self.output_proj(output)
        
        return output
    
    def decode_autoregressive(self, memory: Tensor, max_length: int) -> Tensor:
        """Autoregressive decoding."""
        batch_size = memory.size(0)
        device = memory.device
        
        # Start with start token
        tokens = torch.full((batch_size, 1), self.start_token, dtype=torch.long, device=device)
        outputs = []
        
        for _ in range(max_length):
            # Embed current tokens
            tgt = self.token_embed(tokens)
            tgt = tgt + self.token_pos_embed[:, :tokens.size(1), :]
            
            # Causal mask
            mask = nn.Transformer.generate_square_subsequent_mask(tokens.size(1)).to(device)
            
            # Decode
            output = self.decoder(tgt, memory, tgt_mask=mask)
            output = self.output_proj(output[:, -1:, :])  # Take last token
            outputs.append(output)
            
            # Get next token
            next_token = output.argmax(dim=-1)
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Check for end token
            if (next_token == self.end_token).all():
                break
        
        return torch.cat(outputs, dim=1)


class EasyOCRWrapper:
    """
    Wrapper for EasyOCR library.
    
    Provides a simple interface to EasyOCR with fallback behavior.
    
    Args:
        lang_list: List of language codes (e.g., ['en', 'ch_sim'])
        gpu: Whether to use GPU
        model_storage_directory: Directory to store models
    
    Example:
        >>> ocr = EasyOCRWrapper(lang_list=['en'])
        >>> result = ocr.readtext('image.jpg')
    """
    
    def __init__(
        self,
        lang_list: List[str] = None,
        gpu: bool = True,
        model_storage_directory: Optional[str] = None,
        download_enabled: bool = True
    ):
        self.lang_list = lang_list or ['en']
        self.gpu = gpu and torch.cuda.is_available()
        self.model_storage_directory = model_storage_directory
        self.download_enabled = download_enabled
        self.reader = None
        self._initialized = False
        
    def _init_reader(self):
        """Lazy initialization of EasyOCR reader."""
        if self._initialized:
            return
        
        try:
            import easyocr
            self.reader = easyocr.Reader(
                self.lang_list,
                gpu=self.gpu,
                model_storage_directory=self.model_storage_directory,
                download_enabled=self.download_enabled
            )
            self._initialized = True
        except ImportError:
            raise ImportError(
                "EasyOCR not installed. Install with: pip install easyocr"
            )
    
    def readtext(
        self,
        img: Union[str, np.ndarray, Image.Image],
        detail: int = 1,
        paragraph: bool = False,
        contrast_ths: float = 0.1,
        text_threshold: float = 0.7,
        low_text: float = 0.4,
        link_threshold: float = 0.4,
        canvas_size: int = 2560,
        mag_ratio: float = 1.0,
        slope_ths: float = 0.1,
        ycenter_ths: float = 0.5,
        height_ths: float = 0.5,
        width_ths: float = 0.5,
        add_margin: float = 0.1,
        reformat: bool = True
    ) -> Union[List[Tuple], List[TextBox]]:
        """
        Read text from image.
        
        Args:
            img: Input image (path, numpy array, or PIL Image)
            detail: 0 for simple list, 1 for detailed results
            paragraph: Whether to combine text into paragraphs
            text_threshold: Text detection threshold
            link_threshold: Link detection threshold
            canvas_size: Maximum image size
            mag_ratio: Image magnification ratio
            reformat: Whether to return TextBox objects
        
        Returns:
            List of text detections
        """
        self._init_reader()
        
        # Convert PIL to numpy if needed
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        result = self.reader.readtext(
            img,
            detail=detail,
            paragraph=paragraph,
            contrast_ths=contrast_ths,
            text_threshold=text_threshold,
            low_text=low_text,
            link_threshold=link_threshold,
            canvas_size=canvas_size,
            mag_ratio=mag_ratio,
            slope_ths=slope_ths,
            ycenter_ths=ycenter_ths,
            height_ths=height_ths,
            width_ths=width_ths,
            add_margin=add_margin
        )
        
        if reformat:
            return self._reformat_results(result)
        return result
    
    def _reformat_results(self, results: List) -> List[TextBox]:
        """Reformat EasyOCR results to TextBox objects."""
        text_boxes = []
        for item in results:
            if len(item) == 3:
                bbox, text, conf = item
                # Convert bbox to (x1, y1, x2, y2)
                points = np.array(bbox)
                x1, y1 = points.min(axis=0)
                x2, y2 = points.max(axis=0)
                text_boxes.append(TextBox(
                    text=text,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    confidence=float(conf)
                ))
        return text_boxes


class TesseractWrapper:
    """
    Wrapper for Tesseract OCR.
    
    Provides a Pythonic interface to Tesseract with preprocessing options.
    
    Args:
        lang: Language code (e.g., 'eng', 'eng+fra')
        oem: OCR Engine Mode (0-3)
        psm: Page Segmentation Mode (0-13)
        config: Additional Tesseract configuration string
    
    Example:
        >>> ocr = TesseractWrapper(lang='eng')
        >>> text = ocr.image_to_string('image.jpg')
        >>> data = ocr.image_to_data('image.jpg')
    """
    
    # OCR Engine Modes
    OEM_TESSERACT_ONLY = 0
    OEM_LSTM_ONLY = 1
    OEM_TESSERACT_LSTM_COMBINED = 2
    OEM_DEFAULT = 3
    
    # Page Segmentation Modes
    PSM_OSD_ONLY = 0
    PSM_AUTO_OSD = 1
    PSM_AUTO_ONLY = 2
    PSM_AUTO = 3
    PSM_SINGLE_COLUMN = 4
    PSM_SINGLE_BLOCK_VERT_TEXT = 5
    PSM_SINGLE_BLOCK = 6
    PSM_SINGLE_LINE = 7
    PSM_SINGLE_WORD = 8
    PSM_CIRCLE_WORD = 9
    PSM_SINGLE_CHAR = 10
    PSM_SPARSE_TEXT = 11
    PSM_SPARSE_TEXT_OSD = 12
    PSM_RAW_LINE = 13
    
    def __init__(
        self,
        lang: str = 'eng',
        oem: int = 3,
        psm: int = 3,
        config: str = ''
    ):
        self.lang = lang
        self.oem = oem
        self.psm = psm
        self.config = config
        self._check_tesseract()
    
    def _check_tesseract(self):
        """Check if pytesseract is installed."""
        try:
            import pytesseract
            self.pytesseract = pytesseract
        except ImportError:
            raise ImportError(
                "pytesseract not installed. Install with: pip install pytesseract"
            )
    
    def image_to_string(
        self,
        image: Union[str, np.ndarray, Image.Image],
        preprocess: Optional[str] = None
    ) -> str:
        """
        Convert image to string.
        
        Args:
            image: Input image
            preprocess: Preprocessing method ('thresh', 'blur', 'adaptive')
        
        Returns:
            Extracted text string
        """
        img = self._load_image(image)
        
        if preprocess:
            img = self._preprocess(img, preprocess)
        
        config = f'--oem {self.oem} --psm {self.psm} {self.config}'
        return self.pytesseract.image_to_string(img, lang=self.lang, config=config)
    
    def image_to_data(
        self,
        image: Union[str, np.ndarray, Image.Image],
        preprocess: Optional[str] = None,
        output_type: str = 'dict'
    ) -> Union[Dict, List[TextBox]]:
        """
        Get detailed OCR data including bounding boxes.
        
        Args:
            image: Input image
            preprocess: Preprocessing method
            output_type: 'dict' for raw data, 'textboxes' for TextBox objects
        
        Returns:
            OCR data with bounding boxes
        """
        img = self._load_image(image)
        
        if preprocess:
            img = self._preprocess(img, preprocess)
        
        config = f'--oem {self.oem} --psm {self.psm} {self.config}'
        data = self.pytesseract.image_to_data(
            img, lang=self.lang, config=config, output_type=self.pytesseract.Output.DICT
        )
        
        if output_type == 'textboxes':
            return self._data_to_textboxes(data)
        return data
    
    def _load_image(self, image: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """Load image to PIL format."""
        if isinstance(image, str):
            return Image.open(image)
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image)
        return image
    
    def _preprocess(self, image: Image.Image, method: str) -> Image.Image:
        """Apply preprocessing to image."""
        img = np.array(image)
        
        if method == 'thresh':
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif method == 'blur':
            img = cv2.medianBlur(img, 5)
        elif method == 'adaptive':
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(img)
    
    def _data_to_textboxes(self, data: Dict) -> List[TextBox]:
        """Convert Tesseract data to TextBox objects."""
        text_boxes = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            if int(data['conf'][i]) > 0:  # Filter low confidence
                text_boxes.append(TextBox(
                    text=data['text'][i],
                    bbox=(
                        data['left'][i],
                        data['top'][i],
                        data['left'][i] + data['width'][i],
                        data['top'][i] + data['height'][i]
                    ),
                    confidence=data['conf'][i] / 100.0
                ))
        
        return text_boxes


class TextRecognizer:
    """
    Unified OCR interface supporting multiple backends.
    
    Provides a common interface for different OCR engines with
    automatic backend selection and result standardization.
    
    Args:
        backend: OCR backend ('crnn', 'transformer', 'easyocr', 'tesseract')
        **kwargs: Backend-specific arguments
    
    Example:
        >>> recognizer = TextRecognizer(backend='easyocr', lang_list=['en'])
        >>> result = recognizer.recognize('image.jpg')
    """
    
    SUPPORTED_BACKENDS = ['crnn', 'transformer', 'easyocr', 'tesseract', 'auto']
    
    def __init__(self, backend: str = 'auto', **kwargs):
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"Unsupported backend: {backend}")
        
        self.backend = backend
        self.kwargs = kwargs
        self._engine = None
        
        if backend != 'auto':
            self._init_engine()
    
    def _init_engine(self):
        """Initialize the OCR engine."""
        if self.backend == 'crnn':
            self._engine = CRNN(**self.kwargs)
        elif self.backend == 'transformer':
            self._engine = TransformerOCR(**self.kwargs)
        elif self.backend == 'easyocr':
            self._engine = EasyOCRWrapper(**self.kwargs)
        elif self.backend == 'tesseract':
            self._engine = TesseractWrapper(**self.kwargs)
    
    def recognize(
        self,
        image: Union[str, np.ndarray, Image.Image, Tensor],
        return_boxes: bool = True,
        **kwargs
    ) -> Union[str, List[TextBox]]:
        """
        Recognize text in image.
        
        Args:
            image: Input image
            return_boxes: Whether to return bounding boxes
            **kwargs: Additional arguments for specific backend
        
        Returns:
            Recognized text or list of TextBox objects
        """
        if self.backend == 'auto':
            # Try available backends
            for backend in ['easyocr', 'tesseract', 'crnn']:
                try:
                    self.backend = backend
                    self._init_engine()
                    return self.recognize(image, return_boxes=return_boxes, **kwargs)
                except ImportError:
                    continue
            raise RuntimeError("No OCR backend available")
        
        if self.backend in ['easyocr', 'tesseract']:
            if return_boxes:
                if self.backend == 'easyocr':
                    return self._engine.readtext(image, **kwargs)
                else:
                    return self._engine.image_to_data(image, output_type='textboxes', **kwargs)
            else:
                if self.backend == 'easyocr':
                    boxes = self._engine.readtext(image, **kwargs)
                    return ' '.join([box.text for box in boxes])
                else:
                    return self._engine.image_to_string(image, **kwargs)
        
        elif self.backend in ['crnn', 'transformer']:
            # For neural models
            if isinstance(image, str):
                image = Image.open(image).convert('L')
            if isinstance(image, Image.Image):
                image = np.array(image)
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float() / 255.0
            
            self._engine.eval()
            with torch.no_grad():
                output = self._engine(image)
            
            # Decode output
            if self.backend == 'crnn':
                alphabet = kwargs.get('alphabet', '0123456789abcdefghijklmnopqrstuvwxyz')
                texts = self._engine.decode(output, alphabet)
                return texts[0] if texts else ""
            else:
                # Transformer - get predictions
                predictions = output.argmax(dim=-1)
                return self._decode_transformer_predictions(predictions)
    
    def _decode_transformer_predictions(self, predictions: Tensor) -> str:
        """Decode transformer predictions."""
        # Simplified decoding - would need alphabet mapping in practice
        return ""


# ============================================================================
# Document Layout Analysis
# ============================================================================

class LayoutLM(nn.Module):
    """
    Layout-aware language model for document understanding.
    
    Combines text embeddings with layout (bounding box) embeddings
    to understand document structure.
    
    Args:
        vocab_size: Vocabulary size
        hidden_size: Hidden dimension
        num_hidden_layers: Number of transformer layers
        num_attention_heads: Number of attention heads
        intermediate_size: FFN intermediate dimension
        max_position_embeddings: Maximum sequence length
        max_2d_position_embeddings: Maximum 2D position (layout)
    
    Example:
        >>> model = LayoutLM(vocab_size=30522, hidden_size=768)
        >>> words = ['Hello', 'World']
        >>> boxes = [[0, 0, 100, 20], [100, 0, 200, 20]]
        >>> output = model(words, boxes)
    """
    
    def __init__(
        self,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        max_2d_position_embeddings: int = 1024,
        layer_norm_eps: float = 1e-12
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        
        # Text embeddings
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        
        # 2D position embeddings (for bounding boxes)
        self.x_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)
        self.y_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)
        self.h_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)
        self.w_position_embeddings = nn.Embedding(max_2d_position_embeddings, hidden_size)
        
        # Layer norm and dropout
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=hidden_dropout_prob,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_hidden_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        bbox: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        token_type_ids: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None
    ) -> Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs [batch, seq_len]
            bbox: Bounding boxes [batch, seq_len, 4] (x1, y1, x2, y2)
            attention_mask: Attention mask [batch, seq_len]
            token_type_ids: Token type IDs [batch, seq_len]
            position_ids: Position IDs [batch, seq_len]
        
        Returns:
            Hidden states [batch, seq_len, hidden_size]
        """
        batch_size, seq_length = input_ids.size()
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Text embeddings
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # Layout embeddings
        if bbox is not None:
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
            
            h_position_embeddings = self.h_position_embeddings(bbox[:, :, 3] - bbox[:, :, 1])
            w_position_embeddings = self.w_position_embeddings(bbox[:, :, 2] - bbox[:, :, 0])
            
            layout_embeddings = (
                left_position_embeddings + upper_position_embeddings +
                right_position_embeddings + lower_position_embeddings +
                h_position_embeddings + w_position_embeddings
            )
        else:
            layout_embeddings = 0
        
        # Combine embeddings
        embeddings = (
            words_embeddings + position_embeddings +
            token_type_embeddings + layout_embeddings
        )
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # Transformer encoding
        encoder_outputs = self.encoder(embeddings)
        
        return encoder_outputs


class LayoutParser(nn.Module):
    """
    Detect and parse document regions.
    
    Uses object detection approach to identify different regions
    in documents (paragraphs, headers, tables, figures, etc.).
    
    Args:
        num_classes: Number of region types
        backbone: Feature extraction backbone
        pretrained: Whether to use pretrained weights
    
    Example:
        >>> parser = LayoutParser(num_classes=6)
        >>> image = torch.randn(1, 3, 800, 600)
        >>> regions = parser.detect(image)
    """
    
    # Default region types
    REGION_TYPES = [
        'background',
        'text',
        'title',
        'list',
        'table',
        'figure',
        'header',
        'footer',
        'form_field',
        'signature'
    ]
    
    def __init__(
        self,
        num_classes: int = 10,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        min_size: int = 800,
        max_size: int = 1333,
        box_score_thresh: float = 0.5,
        box_nms_thresh: float = 0.5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.min_size = min_size
        self.max_size = max_size
        self.box_score_thresh = box_score_thresh
        self.box_nms_thresh = box_nms_thresh
        
        # Build detection model (using torchvision's Mask R-CNN)
        from torchvision.models.detection import maskrcnn_resnet50_fpn
        
        self.model = maskrcnn_resnet50_fpn(
            pretrained=pretrained,
            num_classes=num_classes + 1,  # +1 for background
            min_size=min_size,
            max_size=max_size,
            box_score_thresh=box_score_thresh,
            box_nms_thresh=box_nms_thresh
        )
    
    def forward(self, images: List[Tensor], targets: Optional[List[Dict]] = None):
        """
        Forward pass.
        
        Args:
            images: List of image tensors
            targets: List of target dictionaries (for training)
        
        Returns:
            Losses (training) or predictions (inference)
        """
        return self.model(images, targets)
    
    def detect(self, image: Union[Tensor, np.ndarray, Image.Image]) -> List[DocumentRegion]:
        """
        Detect regions in image.
        
        Args:
            image: Input image
        
        Returns:
            List of detected regions
        """
        self.eval()
        
        # Convert to tensor
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        with torch.no_grad():
            predictions = self.model(image)
        
        # Convert to DocumentRegion objects
        regions = []
        pred = predictions[0]
        
        for i in range(len(pred['boxes'])):
            box = pred['boxes'][i].cpu().numpy().astype(int)
            label = int(pred['labels'][i])
            score = float(pred['scores'][i])
            
            if label < len(self.REGION_TYPES):
                region_type = self.REGION_TYPES[label]
            else:
                region_type = f'class_{label}'
            
            regions.append(DocumentRegion(
                region_type=region_type,
                bbox=tuple(box),
                confidence=score
            ))
        
        return regions


class TableDetector(nn.Module):
    """
    Detect and extract tables from documents.
    
    Specialized model for table detection and structure recognition.
    
    Args:
        detect_tables: Whether to detect table regions
        detect_structure: Whether to detect table structure (rows/columns)
        pretrained: Whether to use pretrained weights
    
    Example:
        >>> detector = TableDetector()
        >>> image = torch.randn(1, 3, 800, 600)
        >>> tables = detector.detect(image)
    """
    
    def __init__(
        self,
        detect_tables: bool = True,
        detect_structure: bool = True,
        pretrained: bool = True
    ):
        super().__init__()
        self.detect_tables = detect_tables
        self.detect_structure = detect_structure
        
        # Table detection model
        if detect_tables:
            from torchvision.models.detection import fasterrcnn_resnet50_fpn
            self.detection_model = fasterrcnn_resnet50_fpn(
                pretrained=pretrained,
                num_classes=2  # background + table
            )
        
        # Table structure recognition
        if detect_structure:
            # Simple CNN for cell detection
            self.structure_model = nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=2, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2),
                
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(256, 3, 1)  # row, column, cell detections
            )
    
    def detect(self, image: Union[Tensor, np.ndarray, Image.Image]) -> List[Dict]:
        """
        Detect tables in image.
        
        Args:
            image: Input image
        
        Returns:
            List of table dictionaries with 'bbox' and optional 'cells'
        """
        self.eval()
        
        # Convert to tensor
        if isinstance(image, Image.Image):
            image = np.array(image)
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        tables = []
        
        with torch.no_grad():
            if self.detect_tables:
                predictions = self.detection_model(image)
                pred = predictions[0]
                
                for i in range(len(pred['boxes'])):
                    box = pred['boxes'][i].cpu().numpy().astype(int)
                    score = float(pred['scores'][i])
                    
                    table_info = {
                        'bbox': tuple(box),
                        'confidence': score
                    }
                    
                    if self.detect_structure:
                        # Crop table region
                        table_img = image[0, :, box[1]:box[3], box[0]:box[2]]
                        table_img = table_img.unsqueeze(0)
                        
                        # Detect structure
                        structure = self.structure_model(table_img)
                        cells = self._extract_cells(structure, box)
                        table_info['cells'] = cells
                    
                    tables.append(table_info)
        
        return tables
    
    def _extract_cells(
        self,
        structure: Tensor,
        table_bbox: Tuple[int, int, int, int]
    ) -> List[Dict]:
        """Extract cell bounding boxes from structure predictions."""
        # Simplified cell extraction
        structure_np = structure.squeeze(0).cpu().numpy()
        
        # Threshold for row/column lines
        cell_map = structure_np[2] > 0.5
        
        cells = []
        # Find connected components for cells
        cell_map_uint8 = (cell_map * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            cell_map_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Map back to original coordinates
            cells.append({
                'bbox': (
                    table_bbox[0] + x,
                    table_bbox[1] + y,
                    table_bbox[0] + x + w,
                    table_bbox[1] + y + h
                )
            })
        
        return cells


class FormUnderstanding:
    """
    Parse form fields and extract key-value pairs.
    
    Combines layout analysis with OCR to understand form structure
    and extract field values.
    
    Args:
        layout_parser: Layout parser model
        ocr_engine: OCR engine for text recognition
    
    Example:
        >>> form_parser = FormUnderstanding()
        >>> fields = form_parser.parse('form.jpg')
    """
    
    def __init__(
        self,
        layout_parser: Optional[LayoutParser] = None,
        ocr_engine: Optional[TextRecognizer] = None
    ):
        self.layout_parser = layout_parser or LayoutParser()
        self.ocr_engine = ocr_engine or TextRecognizer(backend='easyocr')
    
    def parse(
        self,
        image: Union[str, np.ndarray, Image.Image],
        form_type: Optional[str] = None
    ) -> List[KeyValuePair]:
        """
        Parse form and extract key-value pairs.
        
        Args:
            image: Input form image
            form_type: Type of form for specialized parsing
        
        Returns:
            List of extracted key-value pairs
        """
        # Load image
        if isinstance(image, str):
            img = Image.open(image)
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image
        
        # Detect form fields
        regions = self.layout_parser.detect(img)
        form_fields = [r for r in regions if r.region_type == 'form_field']
        
        # Recognize text in each field
        key_values = []
        for field in form_fields:
            # Crop field region
            field_img = img.crop(field.bbox)
            
            # Run OCR
            text = self.ocr_engine.recognize(field_img, return_boxes=False)
            
            # Try to split into key and value
            kv = self._extract_key_value(text, form_type)
            if kv:
                key_values.append(kv)
        
        return key_values
    
    def _extract_key_value(
        self,
        text: str,
        form_type: Optional[str] = None
    ) -> Optional[KeyValuePair]:
        """Extract key-value pair from text."""
        # Common patterns
        patterns = [
            r'(.+?)[:\\s]+(.+)',  # Key: Value or Key   Value
            r'(.+?)=(.+)',       # Key=Value
        ]
        
        for pattern in patterns:
            match = re.match(pattern, text.strip())
            if match:
                return KeyValuePair(
                    key=match.group(1).strip(),
                    value=match.group(2).strip(),
                    confidence=1.0
                )
        
        # If no pattern matches, treat entire text as value
        return KeyValuePair(key='', value=text.strip(), confidence=0.5)
