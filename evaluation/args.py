import argparse
from argparse import ArgumentParser
import os
import os.path as osp


def parse_arguments():
    parser = argparse.ArgumentParser(description="Text2Loc Evaluation")

    # Paths
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="K360")
    parser.add_argument("--base_path", type=str)
    parser.add_argument("--path_coarse", type=str, help="The path to the Cell-Retrieval model")
    parser.add_argument(
        "--path_fine", type=str, help="The path to the Hints-to-Objects matching model"
    )

    # Options
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument(
        "--threshs", type=int, nargs="+", default=[5, 10, 15]
    )  # Possibly over-write here when it becomes a list of tuples
    parser.add_argument("--use_features", nargs="+", default=["class", "color", "position", "num"])
    parser.add_argument(
        "--use_test_set", action="store_true", help="Run test-set instead of the validation set."
    )

    parser.add_argument("--no_pc_augment", action="store_true")
    parser.add_argument("--no_pc_augment_fine", action="store_true")

    parser.add_argument(
        "--plot_retrievals",
        action="store_true",
        help="Plot 3 success and fail examples, then quit.",
    )
    parser.add_argument("--plot_matches", action="store_true")
    parser.add_argument("--coarse_only", action="store_true")

    # Fine
    parser.add_argument("--fine_embed_dim", type=int, default=128)
    parser.add_argument("--fine_num_decoder_heads", type=int, default=4)
    parser.add_argument("--fine_num_decoder_layers", type=int, default=2)

    parser.add_argument("--pad_size", type=int, default=16)
    parser.add_argument("--num_mentioned", type=int, default=6)
    parser.add_argument("--describe_by", type=str, default="all")

    # Loss
    parser.add_argument("--margin", type=float, default=0.35)  # Before: 0.5
    parser.add_argument("--temperature", type=float, default=0.1) 
    parser.add_argument("--ranking_loss", type=str, default="pairwise")

    # Object-encoder / PointNet
    parser.add_argument("--coarse_embed_dim", type=int, default=256)
    parser.add_argument("--pointnet_layers", type=int, default=3)
    parser.add_argument("--pointnet_variation", type=int, default=0)
    parser.add_argument("--pointnet_numpoints", type=int, default=256)
    parser.add_argument(
        "--pointnet_path", type=str, default="./checkpoints/pointnet_acc0.86_lr1_p256.pth"
    )
    parser.add_argument("--pointnet_freeze", action="store_true")
    parser.add_argument("--pointnet_features", type=int, default=2)

    parser.add_argument("--class_embed", action="store_true")
    parser.add_argument("--color_embed", action="store_true")
    
    parser.add_argument("--object_size", type=int, default=28)
    parser.add_argument("--object_inter_module_num_heads", type=int, default=4)
    parser.add_argument("--object_inter_module_num_layers", type=int, default=2)

    # Language Encoder
    parser.add_argument("--hungging_model", type=str, help="hugging face model")
    parser.add_argument("--fixed_embedding", action="store_true")

    parser.add_argument("--inter_module_num_heads", type=int, default=4)
    parser.add_argument("--inter_module_num_layers", type=int, default=1)
    parser.add_argument("--intra_module_num_heads", type=int, default=4)
    parser.add_argument("--intra_module_num_layers", type=int, default=1)
    parser.add_argument("--fine_intra_module_num_heads", type=int, default=4)
    parser.add_argument("--fine_intra_module_num_layers", type=int, default=1)

    args = parser.parse_args()

    assert osp.isfile(args.path_coarse)
    if args.path_fine:
        assert osp.isfile(args.path_fine)

    return args

