from argparse import ArgumentParser


def model_opts():
    """
    configuration for training and evaluation
    :param parser: parser
    :return: parsed config
    """
    parser = ArgumentParser(description="train.py")
    #parser.add_argument('--data', default='../data/elec/', type=str,
    #                    help="data path")
    parser.add_argument('--config', default='default.yaml', type=str,
                        help="config file")
    parser.add_argument('--gpu', default="0 1 2", type=str,
                        help="Use CUDA on the device.")
    parser.add_argument('--restore', default='', type=str,
                        help="restore checkpoint")
    parser.add_argument('--mode', default='train', type=str,
                        help="Mode selection")
    parser.add_argument('--expname', default='', type=str,
                        help="expeirment name")
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--beam-size', default=1, type=int)
    parser.add_argument('--save-individual', action='store_true', default=False,
                        help="save individual checkpoint")
    parser.add_argument('--pretrain', default='', type=str,
                        help="load pretrain encoder")
    parser.add_argument('--metrics', default='bleu', type=str,
                        help="evaluation metrics")

    # for prediction
    parser.add_argument('--prediciton_file', type=str,
                        help="Path to store predicted candidates during evaluation or prediction")
    parser.add_argument('--test-src', default='', type=str,
                        help="test source file")
    parser.add_argument('--test-tgt', default='', type=str,
                        help="test target file")
    parser.add_argument('--src-trun', type=int, default=512,
                        help="Truncate source sequence length")
    parser.add_argument('--tgt-trun', type=int, default=0,
                        help="Truncate target sequence length")
    parser.add_argument('--max_sent_len', type=int, default=32,
                        help="Truncate source sentence length")
    parser.add_argument('--max_sent_num', type=int, default=16,
                        help="Truncate source sentence number")
    parser.add_argument('--min_dec_len', type=int, default=30,
                        help="Minimum decoding length")
    parser.add_argument('--lower', action='store_true',
                        help='lower the case')
    parser.add_argument('--pointer', action='store_false',
                        help='pointer network')
    parser.add_argument('--gnn', default='classify', type=str,
                        help="sentence inference module")
    parser.add_argument('--num_hops', default='3', type=int,
                        help="number of hops")
    parser.add_argument('--relations', default='0 1 2', type=str,
                        help="0: sem, 1: top, 2: ref")
    return parser.parse_args()
