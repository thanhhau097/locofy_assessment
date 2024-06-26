from dataclasses import dataclass, field


@dataclass
class DataArguments:
    """
    Arguments relating to data.
    """

    data_folder: str = field(
        default="data/",
        metadata={"help": "The folder containing the data files."},
    )
