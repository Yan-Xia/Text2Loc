import argparse
from argparse import ArgumentParser
import os
import os.path as osp


def parse_arguments():
    parser = argparse.ArgumentParser(description="K360 data preparation")

    parser.add_argument("--path_in", type=str, default="./data/kitti360")
    parser.add_argument("--path_out", type=str, default="./data/k360")
    parser.add_argument("--scene_name", type=str)
    parser.add_argument("--num_mentioned", type=int, default=6)
    parser.add_argument("--describe_by", type=str, default="all")

    parser.add_argument("--cell_size", type=int, default=30)
    parser.add_argument(
        "--cell_dist", type=int, default=30, help="The minimum distance between two cells"
    )
    parser.add_argument("--shift_cells", action="store_true")
    parser.add_argument("--grid_cells", action="store_true")

    parser.add_argument("--pose_dist", type=int, default=30)
    parser.add_argument("--pose_count", type=int, default=4)
    parser.add_argument("--shift_poses", action="store_true")

    parser.add_argument("--describe_best_cell", action="store_true")
    parser.add_argument("--no_ontop", action="store_true")

    parser.add_argument(
        "--all_cells", action="store_true", help="Do not reject cells with too few objects."
    )

    args = parser.parse_args()

    assert osp.isdir(args.path_in)
    assert osp.isdir(
        osp.join(args.path_in, "data_3d_semantics", args.scene_name)
    ), f'Input folder not found {osp.join(args.path_in, "data_3d_semantics", args.scene_name)}'

    attribs = [
        args.path_out,
        "allCells" if args.all_cells else None,
        f"{args.cell_size}-{args.cell_dist}",
        "gridCells" if args.grid_cells else ("shiftCells" if args.shift_cells else "noCellShift"),
        f"pd{args.pose_dist}",
        f"pc{args.pose_count}",
        "shiftPoses" if args.shift_poses else None,
        args.describe_by,
        f"nm-{args.num_mentioned}",
        "bestCell" if args.describe_best_cell else None,
        "noOntop" if args.no_ontop else None,
    ]
    args.path_out = "_".join([a for a in attribs if a != None])

    print(f"Folders: {args.path_in} -> {args.path_out}")

    assert args.describe_by in ("closest", "class", "direction", "random", "all")
    assert args.shift_cells + args.grid_cells < 2  # Only one of them
    assert args.shift_poses == True, "Not shifting poses should not be used anymore"

    # Create dirs
    try:
        os.mkdir(args.path_out)
    except:
        pass
    try:
        os.mkdir(osp.join(args.path_out, "cells"))
    except:
        pass
    try:
        os.mkdir(osp.join(args.path_out, "poses"))
    except:
        pass

    return args
