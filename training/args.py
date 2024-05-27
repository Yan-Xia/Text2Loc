import argparse
from argparse import ArgumentParser
import os.path as osp


def parse_arguments():
    parser = argparse.ArgumentParser(description="Text2Loc Training")

    # General

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="K360", help="Currently only K360")
    parser.add_argument("--base_path", type=str, help="Root path of Kitti360Pose")

    # Model
    parser.add_argument("--use_features", nargs="+", default=["class", "color", "position", "num"])
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")

    parser.add_argument(
        "--continue_path", type=str, help="Set to continue from a previous checkpoint"
    )

    parser.add_argument("--no_pc_augment", action="store_true")

    # Fine
    parser.add_argument("--fine_embed_dim", type=int, default=128)
    parser.add_argument("--offset_lambda", type=float, default=5)
    parser.add_argument("--pmc_prob", type=float, default=0.0)
    parser.add_argument("--pmc_threshold", type=float, default=0.4)

    parser.add_argument("--fine_num_decoder_heads", type=int, default=4)
    parser.add_argument("--fine_num_decoder_layers", type=int, default=2)

    parser.add_argument("--pad_size", type=int, default=16)
    parser.add_argument("--num_mentioned", type=int, default=6)
    parser.add_argument("--describe_by", type=str, default="all")

    # Loss
    parser.add_argument("--margin", type=float, default=0.35)  # Before: 0.5
    parser.add_argument("--temperature", type=float, default=0.1) 
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 3, 5])
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

    # Variations which tranlations are fed into the network for training/evaluation.
    # NOTE: These variations did not make much difference.
    parser.add_argument("--regressor_cell", type=str, default="pose")  # Pose or best
    parser.add_argument("--regressor_learn", type=str, default="center")  # Center or closest
    parser.add_argument("--regressor_eval", type=str, default="center")  # Center or closest

    # Others
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--lr_gamma", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="exponential")
    parser.add_argument("--lr_step", type=float, default=10)
    parser.add_argument("--folder_name", type=str, default="folder_name")

    parser.add_argument("--cpus", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="adam")


    args = parser.parse_args()

    if bool(args.continue_path):
        assert osp.isfile(args.continue_path)

    assert args.regressor_cell in ("pose", "best", "all")
    assert args.regressor_learn in ("center", "closest")
    assert args.regressor_eval in ("center", "closest")

    args.dataset = args.dataset.upper()
    assert args.dataset in ("S3D", "K360")

    assert args.ranking_loss in ("triplet", "pairwise", "hardest", "contrastive")

    for feat in args.use_features:
        assert feat in ["class", "color", "position", "num"], "Unexpected feature"

    if args.pointnet_path:
        assert osp.isfile(args.pointnet_path)

    assert osp.isdir(args.base_path)

    assert args.describe_by in ("closest", "class", "direction", "random", "all")

    return args


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
