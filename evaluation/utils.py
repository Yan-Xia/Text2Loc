import numpy as np

from datapreparation.kitti360pose.drawing import plot_matches_in_best_cell


def plot_matches(all_matches, dataset, count=10):
    for i in range(count):
        idx = np.random.randint(len(dataset))
        data = dataset[idx]
        matches = all_matches[i]
        gt_matches = data["matches"]

        # Gather matches
        true_positives = []  # As obj_idx
        false_positives = []  # As obj_idx
        false_negatives = [
            obj_idx for (obj_idx, hint_idx) in gt_matches
        ]  # As obj_idx, pre-fill this with all matched objects from gt
        for obj_idx, hint_idx in enumerate(matches):
            if hint_idx == -1:
                continue
            if (obj_idx, hint_idx) in gt_matches:
                true_positives.append(obj_idx)
            else:
                false_positives.append(obj_idx)
            false_negatives.remove(
                obj_idx
            )  # If the object was matched, its not false-negative anymore.


def calc_sample_accuracies(pose, top_cells, pos_in_cells, top_k, threshs):
    pose_w = pose.pose_w
    assert len(top_cells) == max(top_k) == len(pos_in_cells)
    num_samples = len(top_cells)

    # Calc the pose-prediction in world coordinates for each cell
    pred_w = np.array(
        [
            top_cells[i].bbox_w[0:2] + pos_in_cells[i, :] * top_cells[i].cell_size
            for i in range(num_samples)
        ]
    )

    # Calc the distances to the gt-pose
    dists = np.linalg.norm(pose_w[0:2] - pred_w, axis=1)
    assert len(dists) == max(top_k)

    # Discard close-by distances from different scenes
    pose_scene_name = pose.cell_id.split("_")[0]
    cell_scene_names = np.array([cell.id.split("_")[0] for cell in top_cells])
    dists[pose_scene_name != cell_scene_names] = np.inf

    # Calculate the accuracy: is one of the top-k dists small enough?
    return {k: {t: np.min(dists[0:k]) <= t for t in threshs} for k in top_k}


def print_accuracies(accs, name=""):
    if name:
        print(f"\t\t{name}:")
    top_k = list(accs.keys())
    threshs = list(accs[top_k[0]].keys())
    print("", end="")
    for k in top_k:
        print(f"\t\t\t\t{k}", end="")
    print()
    print("/".join([str(t) for t in threshs]) + ":", end="")
    for k in top_k:
        print("\t" + "/".join([f"{accs[k][t]:0.4f}" for t in threshs]), end="")
    print("\n\n", flush=True)
