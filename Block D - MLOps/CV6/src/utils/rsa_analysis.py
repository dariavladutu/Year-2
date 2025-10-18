"""RSA Analysis Module.

This module segments the root system mask into vertical parts and filters out
noise using contour analysis. It supports downstream processing and visualization
of root architecture structures.
"""

import logging
import cv2
import networkx as nx
import numpy as np
from skan import Skeleton, summarize
from skimage.morphology import skeletonize
from typing import List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def extract_rsa(mask: np.ndarray) -> list[np.ndarray]:
    """Extract vertical segments of root system architecture (RSA).

    This function divides the mask into 5 vertical parts, performs thresholding 
    and contour detection, and filters out small objects to isolate significant 
    root regions.

    Args:
        mask (np.ndarray): A binary mask image of the root system (2D array).

    Returns:
        List[np.ndarray]: A list of 5 binary mask segments representing cleaned 
        root regions.
    """
    try:
        mask = (mask > 0).astype(np.uint8)
        height, width = mask.shape
        part_width = width // 5

        logging.info("Dividing mask into 5 vertical parts...")
        image_parts_vertical = [
            mask[:, i * part_width: (i + 1) * part_width if i < 4 else width]
            for i in range(5)
        ]

        min_area_threshold = 200
        processed_parts = []

        for i, part in enumerate(image_parts_vertical):
            _, binary_mask = cv2.threshold(
                part, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            contour_mask = np.zeros_like(part, dtype=np.uint8)
            kept = 0
            for contour in contours:
                if cv2.contourArea(contour) > min_area_threshold:
                    cv2.drawContours(
                        contour_mask,
                        [contour],
                        -1,
                        255,
                        thickness=cv2.FILLED
                    )
                    kept += 1

            logging.info("Part %d: %d contours kept", i + 1, kept)
            processed_parts.append(contour_mask)

        return processed_parts

    except Exception as e:
        logging.error("RSA extraction failed")
        raise RuntimeError(f"extract_rsa failed: {e}") from e


def measure_root(plant: np.ndarray) -> Tuple:
    """Measure the length of the longest primary root in a binary root mask.

    Skeletonizes the input image, builds a graph from skeleton 
    branches using `skan`, and finds the longest path through 
    the graph using Dijkstra's algorithm.

    Args:
        plant (np.ndarray): Binary mask image of a plant root (2D array).

    Returns:
        Tuple:
            branch_data (pd.DataFrame or None): Table of skeleton branches with 
            distance metadata.
            G (networkx.Graph or None): Graph representation of skeleton.
            longest_path (List[int] or None): Node path representing the 
            longest root.
            longest_path_len (float): Distance of the longest path found.
    """
    plant = np.array(plant, dtype=np.uint8)
    if plant.ndim != 2 or np.max(plant) == 0:
        logging.warning("Empty mask or invalid input detected.")
        return None, None, None, 0

    skeletonized_plant = skeletonize(plant)
    if np.max(skeletonized_plant) == 0:
        logging.warning("No skeleton detected.")
        return None, None, None, 0

    try:
        skeleton_object = Skeleton(skeletonized_plant)
        branch_data = summarize(skeleton_object)
        G = nx.from_pandas_edgelist(
            branch_data,
            source="node-id-src",
            target="node-id-dst",
            edge_attr="branch-distance",
        )
        longest_path_len = 0
        longest_path = None

        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            nodes = list(subgraph.nodes)
            for i, src in enumerate(nodes):
                for dst in nodes[i + 1 :]:
                    try:
                        path_len = nx.dijkstra_path_length(
                            subgraph, src, dst, weight="branch-distance"
                        )
                        if path_len > longest_path_len:
                            longest_path_len = path_len
                            longest_path = nx.dijkstra_path(
                                subgraph, src, dst, weight="branch-distance"
                            )
                    except nx.NetworkXNoPath:
                        continue

        return branch_data, G, longest_path, longest_path_len

    except Exception as e:
        logging.error("Error processing skeleton: {}".format(e))
        return None, None, None, 0
