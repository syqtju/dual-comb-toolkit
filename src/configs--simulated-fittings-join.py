from time import strftime

from lib.files import get_reports_path

timestr = strftime("%Y%m%d-%H%M%S")


DEFAULT_BASE_NAME = "report"
DEFAULT_N = 1


def parse_args():
    """
    Parse command line arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch fitting of simulated measurements in separate terminals."
    )
    parser.add_argument(
        "-b",
        "--base-name",
        type=str,
        default=DEFAULT_BASE_NAME,
        help="Base name for the reports to join. "
        "The n-th report file is constructed as `<base_name>_<n>.txt`.",
    )
    parser.add_argument(
        "-n",
        "--number-reports",
        type=int,
        default=DEFAULT_N,
        help="Number of reports to join.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_name = getattr(args, "base_name", DEFAULT_BASE_NAME)
    n = getattr(args, "number_reports", DEFAULT_N)

    reports = [f"{base_name}_{i}.csv" for i in range(1, n + 1)]
    reports_path = get_reports_path()
    headings = ""
    lines = []

    for report in reports:
        report_path = reports_path + report

        with open(report_path, "r") as f:
            headings = f.readline().strip()
            lines.extend(f.readlines())

    output_file = reports_path + f"{base_name}_joined_{timestr}.csv"

    with open(output_file, "w") as f:
        f.write(headings + "\n")
        for line in lines:
            f.write(line)


if __name__ == "__main__":
    main()
