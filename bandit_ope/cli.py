"""Command-line arguments shared by all experiment scripts."""


def add_data_args(parser):
    """Add the dataset-related options used by every script."""
    parser.add_argument("--data-path", default=None, help="Folder containing random/ and bts/ (see README)")
    parser.add_argument("--campaign", default="men", choices=["men", "women", "all"])
    parser.add_argument("--position", type=int, default=1, choices=[1, 2, 3], help="Slot to study (OBD uses 1-3)")
    parser.add_argument("--no-affinity", action="store_true", help="Drop the user-item affinity features")
    parser.add_argument("--max-rows", type=int, default=None, help="Keep only the first N rows of each log")
    parser.add_argument("--seed", type=int, default=0)
    return parser
