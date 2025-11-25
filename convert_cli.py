import argparse
import sys
import logging
import os
import traceback

from readers import Idd3dReader, BaseReader
from writers import NuScenesWriter, BaseWriter

SUPPORTED_READERS: dict[str, type[BaseReader]] = {
    "idd3d": Idd3dReader,
    # "argoverse2": ArgoverseReader 
}

SUPPORTED_WRITERS: dict[str, type[BaseWriter]] = {
    "nuscenes": NuScenesWriter
}

def main():
    parser = argparse.ArgumentParser(description="Convert autonomous driving datasets.")
    parser.add_argument("--reader", required=True, choices=SUPPORTED_READERS.keys(),
                        help="The input dataset format.")
    parser.add_argument("--writer", required=True, choices=SUPPORTED_WRITERS.keys(),
                        help="The output dataset format.")
    parser.add_argument("--input", required=True,
                        help="Path to the PARENT directory containing all sequence folders (e.g., /path/to/all_idd3d_sequences).")
    parser.add_argument("--output", required=True,
                        help="Path to the single destination (output) directory.")
    
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout) 
        ]
    )
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info(f"Starting Conversion Pipeline")
    log.info("=" * 70)
    log.info(f"Reader:     {args.reader}")
    log.info(f"Writer:     {args.writer}")
    log.info(f"Input Dir:  {args.input}")
    log.info(f"Output Dir: {args.output}")
    log.info("-" * 70)

    if not os.path.isdir(args.input):
        log.error(f"Input path is not a valid directory: {args.input}")
        sys.exit(1)

    try:
        log.info(f"Initializing reader: {args.reader}...")
        ReaderClass = SUPPORTED_READERS[args.reader]
        reader = ReaderClass()

        log.info(f"Initializing writer: {args.writer}...")
        WriterClass = SUPPORTED_WRITERS[args.writer]
        writer = WriterClass()

        sequence_folders_to_process = []
        for item_name in sorted(os.listdir(args.input)):
            seq_path = os.path.join(args.input, item_name)
            if os.path.isdir(seq_path) and os.path.exists(os.path.join(seq_path, 'annot_data.json')):
                sequence_folders_to_process.append(seq_path)
            else:
                log.warning(f"Skipping '{item_name}': Not a valid sequence folder (missing annot_data.json or not a directory).")

        if not sequence_folders_to_process:
            log.error(f"No valid sequence folders found in: {args.input}")
            sys.exit(1)
        
        log.info(f"Found {len(sequence_folders_to_process)} sequences to process.")

        for i, seq_path in enumerate(sequence_folders_to_process):
            log.info("=" * 70)
            log.info(f"Processing sequence {i+1}/{len(sequence_folders_to_process)}: {os.path.basename(seq_path)}")
            log.info("=" * 70)

            log.info(f"Reading from source path: {seq_path}")
            intermediate_data = reader.read(seq_path)
            log.info("Successfully read and parsed source data.")

            log.info(f"Writing to output path: {args.output}")
            writer.write(intermediate_data, args.output)
        
        log.info("=" * 70)
        log.info("All sequences processed successfully!")
        log.info("=" * 70)
        
    except Exception as e:
        log.error("=" * 70)
        log.error("--- A FATAL ERROR OCCURRED ---")
        log.error(f"Error: {e}")
        log.error("=" * 70)
        log.error(traceback.format_exc()) 
        log.error("Conversion FAILED.")
        sys.exit(1) 

if __name__ == "__main__":
    main()
