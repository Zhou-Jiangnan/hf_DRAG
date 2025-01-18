from collections import deque
import random
from typing import Dict, List, Optional

from loguru import logger
import networkx as nx
from tqdm import tqdm

from modules.data_types import DRAGAnswer, Datapoint
from modules.peer import Peer


class DRAGNetwork:
    def __init__(self, num_peers: int, num_peer_attachments: int, llm_url: str, llm_name: str):
        self.num_peers = num_peers
        self.network = nx.barabasi_albert_graph(num_peers, num_peer_attachments)
        self.peers = [Peer(peer_id, llm_url, llm_name) for peer_id in range(num_peers)]

        self.peer_topics: Dict[int, List[str]] = {peer_id: [] for peer_id in range(self.num_peers)}
        self.topic_peers: Dict[str, List[int]] = {}
        self.all_topics: List[str] = []

    def init_knowledge(self, data_points: List[Datapoint]):
        """
        Distributes data points to peers based on their topics (Uniform).

        Args:
            data_points: A list of Datapoint objects.
        """
        # Count all topics
        topic_check: Dict[str, bool] = {}
        for data_point in data_points:
            topic_check[data_point.topic] = True
        self.all_topics = list(topic_check.keys())

        # Assign topics to peers
        self.topic_peers = {topic: [] for topic in self.all_topics}
        for topic in self.all_topics:
            peer_id = random.choice(range(self.num_peers))
            self.topic_peers[topic].append(peer_id)
            self.peer_topics[peer_id].append(topic)

        # Distribute data points to peers based on assigned topics
        for data_point in tqdm(data_points, desc=f"Distributing data points to peers"):
            for peer_id in self.topic_peers[data_point.topic]:
                self.peers[peer_id].add_knowledge(data_point)

    def query(
            self, 
            question: str, 
            query_peer_id: Optional[int] = None, 
            num_query_neighbor: int = 2,
            query_confidence_threshold: float = 0.5,
            ttl: int = 6
    ) -> DRAGAnswer:
        """
        Queries the network for an answer to a question, spreading the query hop-by-hop.

        Args:
            question: The question to ask.
            query_peer_id: The ID of the peer initiating the query. If None, a random peer is selected.
            num_query_neighbor: The maximum number of neighbors to query at each hop.
            query_confidence_threshold: The confidence threshold required for an answer to be accepted.
            ttl: The time-to-live for the query (maximum number of hops).

        Returns:
            The answer to the question if found, otherwise None.
        """

        # Initialize query peer
        if query_peer_id is None:
            query_peer_id = random.choice(range(self.num_peers))

        # Determine the topic of the question first
        question_topic = self.peers[query_peer_id].parse_topic(question, self.all_topics)

        # Use a queue for Breadth-First Search (BFS)
        neighbor_id_queue = deque()
        neighbor_id_queue.append(query_peer_id)

        # Keep track of visited peers to avoid cycles
        visited_ids: Dict[int, bool] = {query_peer_id: True}

        # Iterate through hops
        for hop in range(ttl):
            current_peer_id = neighbor_id_queue.popleft()
            logger.debug(f"\Processing question [{question}] by peer [{current_peer_id}]")

            # Query the current peer
            current_answer, relevant_knowledge, relevant_score = \
                self.peers[current_peer_id].query(question, query_confidence_threshold)
            if current_answer is not None:
                answer = DRAGAnswer(
                    answer=str(current_answer),
                    relevant_knowledge=relevant_knowledge,
                    relevant_score=relevant_score,
                    num_hops=hop
                )
                logger.debug(f"\Got answer for question [{question}] by peer [{current_peer_id}]: {answer}")
                return answer  # Return the answer if found

            # Get neighbors of the current peer
            current_neighbor_ids = list(self.network.neighbors(current_peer_id))

            # Prioritize neighbors based on topic similarity
            picked_neighbor_ids = []
            if len(current_neighbor_ids) > num_query_neighbor:
                
                # Find neighbors whose topics match the question's topic
                topic_matched_neighbors = []
                for neighbor_id in current_neighbor_ids:
                    if question_topic in self.peer_topics[neighbor_id]:
                        topic_matched_neighbors.append(neighbor_id)

                # If more topic-matched neighbors than num_query_neighbor, select randomly
                if len(topic_matched_neighbors) > num_query_neighbor:
                    picked_neighbor_ids = random.sample(topic_matched_neighbors, num_query_neighbor)
                else:
                    picked_neighbor_ids = topic_matched_neighbors
                    # Fill up with other neighbors if necessary
                    remaining_neighbor_ids = list(set(current_neighbor_ids) - set(picked_neighbor_ids))
                    remaining_num = min(num_query_neighbor - len(picked_neighbor_ids), len(remaining_neighbor_ids))
                    picked_neighbor_ids += random.sample(remaining_neighbor_ids, remaining_num)
            
            else:
                # If not enough neighbors, take all
                picked_neighbor_ids = current_neighbor_ids

            # Add picked neighbors to the queue for the next hop
            for neighbor_id in picked_neighbor_ids:
                if neighbor_id not in visited_ids:
                    visited_ids[neighbor_id] = True
                    neighbor_id_queue.append(neighbor_id)

        # if no answer is found within the TTL
        answer = DRAGAnswer(
            answer="",
            relevant_knowledge="",
            relevant_score=0.0,
            num_hops=ttl
        )
        return answer
