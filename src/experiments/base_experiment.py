"""
Base class for all experiments.

Provides common functionality for running experiments and saving results.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.config import EXPERIMENTS_DIR, GRAPHS_DIR, ensure_directories
from src.utils.helpers import save_json, get_timestamp
from src.utils.logger import get_logger


class BaseExperiment(ABC):
    """Abstract base class for experiments."""

    def __init__(self, name: str, description: str):
        """
        Initialize experiment.

        Args:
            name: Experiment name (e.g., 'exp1_needle_in_haystack')
            description: Brief description of the experiment
        """
        self.name = name
        self.description = description
        self.logger = get_logger(f"experiment.{name}")
        self.results: Dict[str, Any] = {}
        ensure_directories()

    @abstractmethod
    def run(self) -> Dict[str, Any]:
        """
        Run the experiment.

        Returns:
            Dictionary containing experiment results
        """
        pass

    @abstractmethod
    def generate_graphs(self, results: Dict[str, Any]) -> List[Path]:
        """
        Generate visualization graphs from results.

        Args:
            results: Experiment results dictionary

        Returns:
            List of paths to generated graph files
        """
        pass

    def save_results(self, results: Dict[str, Any]) -> Path:
        """
        Save experiment results to JSON file.

        Args:
            results: Results dictionary to save

        Returns:
            Path to saved results file
        """
        output_path = EXPERIMENTS_DIR / f"{self.name}_{get_timestamp()}.json"

        # Add metadata
        results_with_meta = {
            "experiment_name": self.name,
            "description": self.description,
            "timestamp": get_timestamp(),
            "results": results
        }

        save_json(results_with_meta, output_path)
        self.logger.info(f"Results saved to {output_path}")
        return output_path

    def get_graph_path(self, graph_name: str) -> Path:
        """
        Get path for a graph file.

        Args:
            graph_name: Name of the graph

        Returns:
            Path for the graph file
        """
        return GRAPHS_DIR / f"{self.name}_{graph_name}.png"

    def execute(self) -> Dict[str, Any]:
        """
        Execute full experiment workflow.

        Returns:
            Complete results including paths to outputs
        """
        self.logger.info(f"Starting experiment: {self.name}")
        self.logger.info(f"Description: {self.description}")

        # Run experiment
        results = self.run()
        self.results = results

        # Save results
        results_path = self.save_results(results)

        # Generate graphs
        graph_paths = self.generate_graphs(results)

        self.logger.info(f"Experiment {self.name} completed")

        return {
            "results": results,
            "results_path": str(results_path),
            "graph_paths": [str(p) for p in graph_paths]
        }
