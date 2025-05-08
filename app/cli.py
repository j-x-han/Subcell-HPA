"""Console script for subcell_hpa."""
import argparse
import subcell_hpa

def main():
    """Console script for subcell_hpa."""
    parser = argparse.ArgumentParser(description="Console script for subcell_hpa.", add_help=True)
    
    # You can add arguments here as needed
    # parser.add_argument('some_argument', type=str, help='Description of the argument')
    parser.add_argument('--antibodies-file', type=str, default='config/test_antibodies.txt', help='Path to the file containing antibodies of interest')
    parser.add_argument('--metadata-file', type=str, default='config/metadata.csv', help='Path to the metadata file containing cell information')
    parser.add_argument('--model-config', type=str, default='config/model_config.yaml', help='Path to the model configuration file')
    parser.add_argument('--offset', type=int, default=0, help='Offset for the starting index of the cells to process')
    parser.add_argument('--batch-size', type=int, default=1, help='Number of cells to process')
    parser.add_argument('--s3-bucket', type=str, default='czi-subcell-public', help='S3 bucket name where the images are stored')
    parser.add_argument('--cell-crops-dir', type=str, default='cell_crops', help='Directory to save the cell crops')
    parser.add_argument('--weights-dir', type=str, default='weights', help='Directory to save the model weights')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory to save the output files')
    
    args = parser.parse_args()

    # Main logic of the script
    subcell_hpa.run(
        antibodies_file=args.antibodies_file,
        metadata_file=args.metadata_file,
        model_config=args.model_config,
        offset=args.offset,
        batch_size=args.batch_size,
        s3_bucket=args.s3_bucket,
        cell_crops_dir=args.cell_crops_dir,
        weights_dir=args.weights_dir,
        output_dir=args.output_dir
    )
    

if __name__ == "__main__":
    main()
