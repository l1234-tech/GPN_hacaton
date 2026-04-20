#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    print("Testing table utils...")
    # Import only table utils to avoid PaddleOCR slowness
    from baseline.table_utils import rows_to_markdown, _is_garbage_cell
    print("✅ Table utils imported")
    
    # Test garbage cell
    assert _is_garbage_cell("") == False
    assert _is_garbage_cell("abc") == False
    assert _is_garbage_cell("  " * 100) == True  # Very long spaces
    print("✅ Garbage cell detection works")
    
    # Test markdown
    result = rows_to_markdown([["Name", "Value"], ["Test", "123"]])
    assert "Name" in result
    assert "Value" in result
    print("✅ Markdown generation works")
    print(f"Markdown output:\n{result}\n")
    
    # Test with garbage table
    result2 = rows_to_markdown([["x" * 100, "y" * 100]])  # Too long
    print(f"Garbage table result: '{result2}' (should be empty)")
    
    print("\n✅ All table tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
