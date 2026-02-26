#!/usr/bin/env python3
"""Test imports with simulated module failures."""

import sys
from pathlib import Path
import importlib.util
import importlib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def test_with_missing_module(file_path: Path, blocked_modules: list[str]) -> tuple[bool, str]:
    """Test file imports with certain modules blocked."""
    original_import = __builtins__.__import__
    
    def mock_import(name, *args, **kwargs):
        if any(blocked in name for blocked in blocked_modules):
            raise ImportError(f"Mocked import failure for {name}")
        return original_import(name, *args, **kwargs)
    
    try:
        __builtins__.__import__ = mock_import
        
        spec = importlib.util.spec_from_file_location("test_module", file_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["test_module"] = module
            spec.loader.exec_module(module)
            return True, "OK"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"
    finally:
        __builtins__.__import__ = original_import
        if "test_module" in sys.modules:
            del sys.modules["test_module"]
    
    return False, "Unknown error"

def main():
    print("=" * 80)
    print("TESTING WITH SIMULATED MODULE FAILURES")
    print("=" * 80)
    
    test_cases = [
        ("apps/pages/1_Data_Pipeline.py", ["datahound.profiles"], "profiles module missing"),
        ("apps/Home.py", ["plotly"], "plotly module missing"),
        ("apps/components/event_dashboards.py", ["plotly"], "plotly module missing"),
    ]
    
    for file_path_str, blocked_modules, scenario in test_cases:
        file_path = ROOT / file_path_str
        if not file_path.exists():
            print(f"\n✗ SKIP: {file_path_str} (file not found)")
            continue
        
        print(f"\n{'='*80}")
        print(f"Testing: {file_path_str}")
        print(f"Scenario: {scenario}")
        print(f"Blocked: {blocked_modules}")
        print(f"{'='*80}")
        
        success, message = test_with_missing_module(file_path, blocked_modules)
        
        if success:
            print(f"✓ PASS: Handles missing modules gracefully")
        else:
            print(f"✗ FAIL: {message}")

if __name__ == "__main__":
    main()
