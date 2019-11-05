import os
import argparse
import yaml
import sys
from pprint import pprint

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Show config of runs.")

    parser.add_argument("run_id", nargs="?", default=-1)

    parser.add_argument("-a", action="store_true", dest="active")

    parser.add_argument("-p", type=str, nargs="*", dest="params")

    args = parser.parse_args()


    file_path = ".running" if args.active else ".archive"


    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            runs = [r.strip("\n") for r in file.readlines()]
    else:
        print("No file {} found.".format(file_path))
        sys.exit()

    if len(runs) == 0:
        print("No runs found in {}.".format(file_path))
        sys.exit()
    else:

        if args.run_id == -1 or str(args.run_id).isnumeric():
            config_path = runs[args.run_id]
        else:
            for r in runs:
                if args.run_id in r:
                    config_path = r
                    break

        if os.path.exists(config_path):
            with open(config_path, "r") as config_file:
                config = yaml.load(config_file, Loader=yaml.FullLoader)
        else:
            print("Config file does not (yet) exist.")
            sys.exit()

        if args.params is not None:
            for p in args.params:
                if p in config:
                    print("{:15} {}".format(p, config[p]))
                else:
                    print("{:15} ??".format(p))
        else:
            pprint(config)
