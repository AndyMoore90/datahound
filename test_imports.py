#!/usr/bin/env python3
"""Import smoke test for Streamlit app files."""

import sys
from pathlib import Path
import importlib.util
import traceback

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def test_file_imports(file_path: Path) -> tuple[bool, str]:
    """Test if a Python file can be imported without errors."""
    try:
        spec = importlib.util.spec_from_file_location("test_module", file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["test_module"] = module
            spec.loader.exec_module(module)
            return True, "OK"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"
    return False, "Unknown error"

def main():
    test_files = [
        "apps/Home.py",
        "apps/pages/1_Data_Pipeline.py",
        "apps/pages/2_Events.py",
        "apps/pages/3_Pipeline_Monitor.py",
        "apps/pages/4_Permits.py",
        "apps/pages/5_Admin.py",
        "apps/components/event_dashboards.py",
        "apps/components/event_detection_ui.py",
        "apps/components/extraction_ui.py",
        "apps/components/scheduler_ui.py",
        "apps/components/ui_components.py",
        "apps/components/upsert_ui.py",
    ]
    
    print("=" * 80)
    print("STREAMLIT APP IMPORT SMOKE TEST")
    print("=" * 80)
    
    results = []
    for file_path_str in test_files:
        file_path = ROOT / file_path_str
        if not file_path.exists():
            results.append((file_path_str, False, "File not found"))
            continue
        
        success, message = test_file_imports(file_path)
        results.append((file_path_str, success, message))
        
        # Clear the test module
        if "test_module" in sys.modules:
            del sys.modules["test_module"]
    
    # Print results
    passed = 0
    failed = 0
    
    for file_path, success, message in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"\n{status}: {file_path}")
        if not success:
            print(f"  Error: {message}")
            failed += 1
        else:
            passed += 1
    
    print("\n" + "=" * 80)
    print(f"Results: {passed} passed, {failed} failed out of {len(results)} files")
    print("=" * 80)
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
