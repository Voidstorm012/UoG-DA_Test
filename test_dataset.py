"""
Test script to ensure we can load the UWB dataset
"""
import os
import sys

# Add UWB dataset directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'data', 'code'))

def test_dataset_loading():
    try:
        import uwb_dataset
        print("Successfully imported uwb_dataset module")
        
        # Change to the data directory for correct relative paths
        original_dir = os.getcwd()
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        os.chdir(data_dir)
        
        # Import the dataset
        print("Loading dataset...")
        try:
            data = uwb_dataset.import_from_files()
            print(f"Dataset loaded successfully! Shape: {len(data)} samples")
            
            # Print first few rows
            print("\nFirst sample:")
            print(f"NLOS: {data[0][0]}")
            print(f"Range: {data[0][1]}")
            print(f"FP_IDX: {data[0][2]}")
            
            return True
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
        finally:
            # Restore original directory
            os.chdir(original_dir)
            
    except ImportError as e:
        print(f"Error importing uwb_dataset module: {e}")
        return False

if __name__ == "__main__":
    print("Testing UWB dataset loading...")
    success = test_dataset_loading()
    print(f"\nTest {'passed' if success else 'failed'}")