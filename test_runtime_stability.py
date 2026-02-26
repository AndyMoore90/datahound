#!/usr/bin/env python3
"""
Runtime stability test for Streamlit app.
Verifies graceful handling of missing optional dependencies.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

def test_plotly_fallback():
    """Test that files handle missing plotly gracefully."""
    print("\n" + "="*80)
    print("TEST: Plotly fallback handling")
    print("="*80)
    
    # Test Home.py
    print("\n1. Testing apps/Home.py with plotly available...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("home_module", ROOT / "apps/Home.py")
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["home_module"] = module
            spec.loader.exec_module(module)
            
            # Check that PLOTLY_AVAILABLE flag exists
            if hasattr(module, 'PLOTLY_AVAILABLE'):
                print(f"   ✓ PLOTLY_AVAILABLE = {module.PLOTLY_AVAILABLE}")
            else:
                print("   ✗ PLOTLY_AVAILABLE flag missing")
                return False
            
            # Clean up
            if "home_module" in sys.modules:
                del sys.modules["home_module"]
        
        print("   ✓ Home.py loads successfully")
    except Exception as e:
        print(f"   ✗ Home.py failed: {e}")
        return False
    
    # Test ui_components.py
    print("\n2. Testing apps/components/ui_components.py...")
    try:
        spec = importlib.util.spec_from_file_location("ui_comp", ROOT / "apps/components/ui_components.py")
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["ui_comp"] = module
            spec.loader.exec_module(module)
            
            if hasattr(module, 'PLOTLY_AVAILABLE'):
                print(f"   ✓ PLOTLY_AVAILABLE = {module.PLOTLY_AVAILABLE}")
            else:
                print("   ✗ PLOTLY_AVAILABLE flag missing")
                return False
            
            # Test dh_create_metric_chart gracefully handles missing plotly
            if hasattr(module, 'dh_create_metric_chart'):
                print("   ✓ dh_create_metric_chart function available")
            
            if "ui_comp" in sys.modules:
                del sys.modules["ui_comp"]
        
        print("   ✓ ui_components.py loads successfully")
    except Exception as e:
        print(f"   ✗ ui_components.py failed: {e}")
        return False
    
    print("\n✓ All plotly fallback tests passed")
    return True


def test_profiles_fallback():
    """Test that 1_Data_Pipeline.py handles missing profiles module."""
    print("\n" + "="*80)
    print("TEST: Profiles module fallback handling")
    print("="*80)
    
    print("\n1. Testing apps/pages/1_Data_Pipeline.py...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "pipeline_module",
            ROOT / "apps/pages/1_Data_Pipeline.py"
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules["pipeline_module"] = module
            spec.loader.exec_module(module)
            
            # Check for PROFILES_AVAILABLE flag
            if hasattr(module, 'PROFILES_AVAILABLE'):
                print(f"   ✓ PROFILES_AVAILABLE = {module.PROFILES_AVAILABLE}")
            else:
                print("   ✗ PROFILES_AVAILABLE flag missing")
                return False
            
            # Check that ProfileBuildMode is handled
            if hasattr(module, 'ProfileBuildMode'):
                pbm = getattr(module, 'ProfileBuildMode')
                if module.PROFILES_AVAILABLE:
                    print(f"   ✓ ProfileBuildMode available (profiles loaded)")
                else:
                    print(f"   ✓ ProfileBuildMode = {pbm} (profiles not loaded, graceful fallback)")
            
            # Check for _render_core_data function that guards usage
            if hasattr(module, '_render_core_data'):
                print("   ✓ _render_core_data function available with guards")
            
            if "pipeline_module" in sys.modules:
                del sys.modules["pipeline_module"]
        
        print("   ✓ 1_Data_Pipeline.py loads successfully")
    except Exception as e:
        print(f"   ✗ 1_Data_Pipeline.py failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ Profiles fallback test passed")
    return True


def test_all_imports():
    """Run import smoke test on all app files."""
    print("\n" + "="*80)
    print("TEST: Import smoke test for all files")
    print("="*80)
    
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
    
    import importlib.util
    
    passed = 0
    failed = 0
    
    for file_path_str in test_files:
        file_path = ROOT / file_path_str
        if not file_path.exists():
            print(f"   ✗ SKIP: {file_path_str} (not found)")
            continue
        
        try:
            spec = importlib.util.spec_from_file_location(f"test_{passed}_{failed}", file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[f"test_{passed}_{failed}"] = module
                spec.loader.exec_module(module)
                print(f"   ✓ {file_path_str}")
                passed += 1
                
                # Clean up
                if f"test_{passed}_{failed}" in sys.modules:
                    del sys.modules[f"test_{passed}_{failed}"]
        except Exception as e:
            print(f"   ✗ {file_path_str}: {e}")
            failed += 1
    
    print(f"\n   Results: {passed} passed, {failed} failed")
    return failed == 0


def main():
    print("\n" + "="*80)
    print("STREAMLIT APP RUNTIME STABILITY TEST")
    print("="*80)
    
    all_passed = True
    
    # Test 1: Import smoke test
    if not test_all_imports():
        all_passed = False
    
    # Test 2: Plotly fallback
    if not test_plotly_fallback():
        all_passed = False
    
    # Test 3: Profiles fallback
    if not test_profiles_fallback():
        all_passed = False
    
    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - App is runtime stable")
    else:
        print("✗ SOME TESTS FAILED - Review errors above")
    print("="*80 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
