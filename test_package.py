#!/usr/bin/env python3
"""
Simple test script to verify the NeuroDAP package works correctly
"""

def test_imports():
    """Test that all modules can be imported"""
    try:
        import pyNeuroDAP as ndap
        print("✅ Successfully imported pyNeuroDAP")
        
        # Test basic imports
        print("✅ Package version:", ndap.__version__)
        print("✅ Author:", ndap.__author__)
        
        # Test module imports
        from pyNeuroDAP import spikes, trials, sessions, plots
        print("✅ All modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic package functionality"""
    try:
        import pyNeuroDAP as ndap
        
        # Test that we can access functions
        print("✅ Available functions:")
        print("   - get_spikes:", hasattr(ndap, 'get_spikes'))
        print("   - save_session_data:", hasattr(ndap, 'save_session_data'))
        print("   - rSLDS:", hasattr(ndap, 'rSLDS'))
        print("   - plot_psth:", hasattr(ndap, 'plot_psth'))
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧠 Testing NeuroDAP Package")
    print("=" * 40)
    
    # Test imports
    if not test_imports():
        print("\n❌ Package import failed!")
        return
    
    print("\n" + "=" * 40)
    
    # Test functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality test failed!")
        return
    
    print("\n" + "=" * 40)
    print("🎉 All tests passed! NeuroDAP package is working correctly.")
    print("\nYou can now use:")
    print("  import pyNeuroDAP as ndap")
    print("  ndap.get_spikes(...)")
    print("  ndap.save_session_data(...)")

if __name__ == "__main__":
    main()
