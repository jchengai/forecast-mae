from argparse import ArgumentParser
from pathlib import Path
from typing import List

import ray
from tqdm import tqdm

from src.datamodule.av2_extractor import Av2Extractor
from src.datamodule.av2_extractor_multiagent import Av2ExtractorMultiAgent
from src.utils.ray_utils import ActorHandle, ProgressBar

ray.init(num_cpus=16)


def glob_files(data_root: Path, mode: str):
    file_root = data_root / mode
    scenario_files = list(file_root.rglob("*.parquet"))
    return scenario_files


@ray.remote
def preprocess_batch(extractor: Av2Extractor, file_list: List[Path], pb: ActorHandle):
    for file in file_list:
        extractor.save(file)
        pb.update.remote(1)


def preprocess(args):
    batch = args.batch
    data_root = Path(args.data_root)

    for mode in ["train", "val", "test"]:
        if args.multiagent:
            save_dir = data_root / "multiagent-baseline" / mode
            extractor = Av2ExtractorMultiAgent(save_path=save_dir, mode=mode)
        else:
            save_dir = data_root / "forecast-mae" / mode
            extractor = Av2Extractor(save_path=save_dir, mode=mode)

        save_dir.mkdir(exist_ok=True, parents=True)
        scenario_files = glob_files(data_root, mode)

        if args.parallel:
            pb = ProgressBar(len(scenario_files), f"preprocess {mode}-set")
            pb_actor = pb.actor

            for i in range(0, len(scenario_files), batch):
                preprocess_batch.remote(
                    extractor, scenario_files[i : i + batch], pb_actor
                )

            pb.print_until_done()
        else:
            for file in tqdm(scenario_files):
                extractor.save(file)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_root", "-d", type=str, required=True)
    parser.add_argument("--batch", "-b", type=int, default=50)
    parser.add_argument("--parallel", "-p", action="store_true")
    parser.add_argument("--multiagent", "-m", action="store_true")

    args = parser.parse_args()
    preprocess(args)
