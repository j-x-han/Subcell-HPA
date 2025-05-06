"""Console script for subcell_hpa."""
import argparse
import subcell_hpa

def main():
    """Console script for subcell_hpa."""
    parser = argparse.ArgumentParser(description="Console script for subcell_hpa.")
    
    # You can add arguments here as needed
    # parser.add_argument('some_argument', type=str, help='Description of the argument')
    
    args = parser.parse_args()

    # Main logic of the script
    print("See argparse documentation at https://docs.python.org/3/library/argparse.html")
    

if __name__ == "__main__":
    main()
