import json
import os
import argparse

def validate():
    with open(args.eval_json_path, "r") as f:
        eval_json = json.load(f)

    with open(args.train_json_path, "r") as f:
        train_json = json.load(f)

    with open(args.all_json_path, "r") as f:
        all_json = json.load(f)

    eval_len = len(eval_json)
    train_len = len(train_json)
    assert eval_len + train_len == 64 * 13 * 12 * 10

    eval_keys = set(eval_json.keys())
    train_keys = set(train_json.keys())
    assert len(eval_keys.intersection(train_keys)) == 0

    assert len(all_json) == 64 * 13 * 10

    print("Validation passed!")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--lighting_pattern_dir", required=True, type=str, default="/workspace/dataset/lighting_patterns")
    args.add_argument("--eval_json_path", type=str, default="/workspace/Diffusion-Project-Illumination/eval.json")
    args.add_argument("--train_json_path", type=str, default="./train.json")
    args.add_argument("--all_json_path", type=str, default="./all.json")
    args = args.parse_args()

    object_names = [d for d in os.listdir(args.lighting_pattern_dir) if os.path.isdir(os.path.join(args.lighting_pattern_dir, d))]

    # Limited pose set of our project
    pose = ["NA3", "NE7", "CB5", "CF8", "NA7", "CC7", "CA2", "NE1", "NC3", "CE2"]

    with open(args.eval_json_path, "r") as f:
        eval_json = json.load(f)

    all_json = {}
    train_json = {}
    for obj in object_names:
        for p in pose:
            for src in range(1, 14):
                key = f"{obj}_{src:03d}_{p}"
                all_json[key] = {
                    "tgt_img_path": f"{obj}/Lights/{src:03d}/raw_undistorted/{p}.JPG",
                    "mask_path": f"{obj}/output/obj_masks/{p}.png"
                }
                for tgt in range(1, 14):
                    if src != tgt:
                        key = f"{obj}_src_{src:03d}_tgt_{tgt:03d}_{p}"
                        if key in eval_json.keys():
                            print(key)
                        else:
                            value = {
                                "src_light": f"{src:03d}",
                                "src_img_path": f"{obj}/Lights/{src:03d}/raw_undistorted/{p}.JPG",
                                "tgt_light": f"{tgt:03d}",
                                "tgt_img_path": f"{obj}/Lights/{tgt:03d}/raw_undistorted/{p}.JPG",
                                "mask_path": f"{obj}/output/obj_masks/{p}.png"
                            }
                            train_json[key] = value

    with open(args.train_json_path, "w") as f:
        json.dump(train_json, f, indent=4)

    with open(args.all_json_path, "w") as f:
        json.dump(all_json, f, indent=4)

    validate()
