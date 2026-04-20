#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    print("Testing imports...")
    from baseline.ocr_utils import _is_garbage_text, _is_watermark_text
    print("✅ OCR utils imported")
    
    from baseline.table_utils import rows_to_markdown, _is_garbage_cell
    print("✅ Table utils imported")
    
    # Test garbage detection
    assert _is_garbage_text("random xxx yyy") == True
    print("✅ Garbage text detection works")
    
    # Test watermark detection
    assert _is_watermark_text("стр. 1") == True
    print("✅ Watermark detection works")
    
    # Test garbage cell
    assert _is_garbage_cell("") == False
    print("✅ Garbage cell detection works")
    
    # Test markdown
    result = rows_to_markdown([["Name", "Value"], ["Test", "123"]])
    assert "Name" in result
    print("✅ Markdown generation works")
    
    print("\n✅ All tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
