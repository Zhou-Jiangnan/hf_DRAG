import csv
import os
from typing import Any, Dict, List, Optional, Set, Union

from loguru import logger
from torch import Tensor


class CSVLogger:
    r"""Log to the local file system in CSV format.

    Logs are saved to ``os.path.join(root_dir, name, version)``.

    Args:
        root_dir: The root directory in which all your experiments with different names and versions will be stored.
        log_dir_name: Log directory name. Defaults to ``'logs'``. If name is ``None``, logs
            (versions) will be stored to the save dir directly.
        version: Experiment version. If version is not specified the logger inspects the save
            directory for existing versions, then automatically assigns the next available version.
            If the version is specified, and the directory already contains a metrics file for that version, it will be
            overwritten.

    Example::

        csv_logger = CSVLogger("path/to/logs/root")
        metrics_logger = csv_logger.logger("metrics")
        metrics_logger.log({"loss": 0.235, "acc": 0.75})
        metrics_logger.save()

    """

    def __init__(
        self,
        root_dir: str = "./",
        log_dir_name: str = "logs",
        version: Optional[Union[int, str]] = None,
    ):
        super().__init__()
        root_dir = os.fspath(root_dir)
        self._root_dir = root_dir
        self._log_dir_name = log_dir_name
        self._version = version
        self._experiment: Optional[_CSVWriter] = None
        self._loggers: Dict[str, _CSVWriter] = {}

    @property
    def version(self) -> Union[int, str]:
        """Gets the version of the experiment.

        Returns:
            The version of the experiment if it is specified, else the next version.

        """
        if self._version is None:
            self._version = self._get_next_version()
        return self._version

    @property
    def log_dir(self) -> str:
        """The log directory for this run.

        By default, it is named ``'version_${self.version}'`` but it can be overridden by passing a string value for the
        constructor's version parameter instead of ``None`` or an int.

        """
        # create a pseudo standard path
        version = self.version if isinstance(self.version, str) else f"version_{self.version}"
        return os.path.join(self._root_dir, self._log_dir_name, version)

    def logger(self, logger_name: str):
        """Actual CSVWriter object."""
        if logger_name in self._loggers:
            return self._loggers[logger_name]

        os.makedirs(self._root_dir, exist_ok=True)
        self._loggers[logger_name] = _CSVWriter(log_dir=self.log_dir, logger_name=logger_name)
        return self._loggers[logger_name]

    def _get_next_version(self) -> int:
        versions_root = os.path.join(self._root_dir, self._log_dir_name)

        if not os.path.isdir(versions_root):
            return 0

        existing_versions = []
        for dir_name in os.listdir(versions_root):
            full_path = os.path.join(versions_root, dir_name)
            name = os.path.basename(full_path)
            if os.path.isdir(full_path) and name.startswith("version_"):
                dir_ver = name.split("_")[1]
                if dir_ver.isdigit():
                    existing_versions.append(int(dir_ver))

        if len(existing_versions) == 0:
            return 0

        return max(existing_versions) + 1


class _CSVWriter:
    r"""CSV writer for CSVLogger.

    Args:
        log_dir: Directory for the logs
        logger_name: Name of the logger

    """

    def __init__(self, log_dir: str, logger_name: str="metrics") -> None:
        self.metrics: List[Dict[str, float]] = []
        self.metrics_keys: List[str] = []

        self.log_dir = log_dir
        self.log_file_name = logger_name + ".csv"
        self.metrics_file_path = os.path.join(self.log_dir, self.log_file_name)

        self._check_log_dir_exists()
        os.makedirs(self.log_dir, exist_ok=True)

    def log(self, data_dict: Dict[str, Union[Tensor, float]]) -> None:
        """Record data dict."""

        def _handle_value(value: Union[Tensor, Any]) -> Any:
            if isinstance(value, Tensor):
                return value.item()
            return value

        metrics = {k: _handle_value(v) for k, v in data_dict.items()}
        self.metrics.append(metrics)

    def save(self) -> None:
        """Save recorded metrics into files."""
        if not self.metrics:
            return

        new_keys = self._record_new_keys()
        file_exists = os.path.isfile(self.metrics_file_path)

        if new_keys and file_exists:
            # we need to re-write the file if the keys (header) change
            self._rewrite_with_new_header(self.metrics_keys)

        with open(self.metrics_file_path, mode=("a" if file_exists else "w"), newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.metrics_keys)
            if not file_exists:
                # only write the header if we're writing a fresh file
                writer.writeheader()
            writer.writerows(self.metrics)

        self.metrics = []  # reset

    def _record_new_keys(self) -> Set[str]:
        """Records new keys that have not been logged before."""
        current_keys = set().union(*self.metrics)
        new_keys = current_keys - set(self.metrics_keys)
        self.metrics_keys.extend(new_keys)
        self.metrics_keys.sort()
        return new_keys

    def _rewrite_with_new_header(self, fieldnames: List[str]) -> None:
        with open(self.metrics_file_path, "r", newline="") as csvfile:
            metrics = list(csv.DictReader(csvfile))

        with open(self.metrics_file_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics)

    def _check_log_dir_exists(self) -> None:
        if os.path.exists(self.log_dir) and os.listdir(self.log_dir):
            logger.warning(
                f"Experiment logs directory {self.log_dir} exists and is not empty."
                " Previous log files in this directory will be deleted when the new ones are saved!"
            )
            if os.path.isfile(self.metrics_file_path):
                os.remove(self.metrics_file_path)
