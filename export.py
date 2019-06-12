#! /usr/bin/env python3
import argparse

from helixerprep.export.exporter import ExportController


def main(args):
    controller = ExportController(args.db_path_in, args.out_dir)
    controller.export(chunk_size=args.chunk_size,
                      shuffle_in_file=args.shuffle_in_file,
                      zip_files=args.zip)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    io = parser.add_argument_group("Data input and output")
    io.add_argument('--db_path_in', type=str, required=True,
                    help=('Path to the Helixer SQLite input database.'))
    io.add_argument('--out_dir', type=str, required=True, help='Output dir for encoded data files.')

    data = parser.add_argument_group("Data generation parameters")
    data.add_argument('--chunk_size', type=int, default=2000000,
                      help='Size of the chunks each genomic sequence gets cut into.')
    data.add_argument('--shuffle_in_file', action='store_true',
                      help=('Whether to shuffle the sequences inside each output file. '
                            '(Does no further shuffling'))
    data.add_argument('--zip', action='store_true', help='Whether to zip the file chunks')
    args = parser.parse_args()
    main(args)
