from collections import deque
import random
from typing import Dict, List, Optional

from loguru import logger
import networkx as nx
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm

from modules.data_types import RAGAnswer, Datapoint
from modules.ppo_router import PPORouter
from modules.peer import Peer


class DRAGNetwork:
    def __init__(
            self, 
            num_peers: int, 
            num_peer_attachments: int, 
            llm_url: str, 
            llm_name: str, 
            llm_num_ctx: int,
            llm_seed: int, 
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
        ):
        self.num_peers = num_peers
        self.network = nx.barabasi_albert_graph(num_peers, num_peer_attachments)
        self.text_embedding_model = SentenceTransformer(embedding_model)
        self.peers = [Peer(peer_id, llm_url, llm_name, llm_num_ctx, llm_seed, self.text_embedding_model) 
                      for peer_id in range(num_peers)]

        self.peer_topics: Dict[int, List[str]] = {peer_id: [] for peer_id in range(self.num_peers)}
        self.topic_peers: Dict[str, List[int]] = {}
        self.all_topics: List[str] = []

    def get_peer_relevance_score(self, peer_id: int, question: str) -> float:
        """
        Estimates peer relevance score without LLM generation, using only semantic retrieval.
        """
        top_results = self.peers[peer_id].knowledge_base.semantic_search(question, 1)
        if len(top_results) == 0:
            return 0.0
        _, relevance_score = top_results[0]
        return float(relevance_score)

    def build_ppo_state_features(
            self,
            current_peer_id: int,
            hop: int,
            max_ttl: int,
            current_score: float,
            visited_count: int
    ) -> List[float]:
        degree = self.network.degree(current_peer_id)
        max_degree = max(1, self.num_peers - 1)
        return [
            float(current_score),
            float(hop / max(1, max_ttl)),
            float((max_ttl - hop) / max(1, max_ttl)),
            float(degree / max_degree),
            float(visited_count / max(1, self.num_peers)),
            float(current_peer_id / max(1, self.num_peers - 1)),
            1.0,
        ]

    def build_ppo_candidate_features(
            self,
            question: str,
            question_topic: Optional[str],
            neighbor_ids: List[int],
            visited_ids: set,
            max_candidates: int
    ) -> tuple[List[int], List[List[float]], List[bool]]:
        scored_neighbors = []
        for neighbor_id in neighbor_ids:
            neighbor_score = self.get_peer_relevance_score(neighbor_id, question)
            topic_match = 1.0 if question_topic and question_topic in self.peer_topics[neighbor_id] else 0.0
            visited_flag = 1.0 if neighbor_id in visited_ids else 0.0
            scored_neighbors.append((neighbor_id, neighbor_score, topic_match, visited_flag))

        # Prefer unvisited + topic-matched + higher relevance neighbors
        scored_neighbors.sort(key=lambda x: (x[3], -x[2], -x[1]))
        selected = scored_neighbors[:max_candidates]

        max_degree = max(1, self.num_peers - 1)
        candidate_features = []
        mask = []
        selected_neighbors = []

        for neighbor_id, neighbor_score, topic_match, visited_flag in selected:
            selected_neighbors.append(neighbor_id)
            candidate_features.append([
                float(neighbor_score),
                float(self.network.degree(neighbor_id) / max_degree),
                float(topic_match),
                float(visited_flag),
            ])
            mask.append(True)

        while len(candidate_features) < max_candidates:
            candidate_features.append([0.0, 0.0, 0.0, 1.0])
            mask.append(False)

        return selected_neighbors, candidate_features, mask

    def ppo_train(
            self,
            router: PPORouter,
            data_points: List[Datapoint],
            num_episodes: int = 200,
            max_ttl: int = 6,
            query_confidence_threshold: float = 0.5,
            reward_hit: float = 1.0,
            reward_miss: float = -0.5,
            message_penalty: float = 0.02,
            hop_penalty: float = 0.01,
            relevance_weight: float = 0.2,
      
            progress_weight: float = 0.3,
            topic_match_bonus: float = 0.2,
            revisit_penalty: float = 0.1,
    ):
        if len(data_points) == 0:
            logger.warning("Skip PPO training since no datapoints are provided")
            return

        for episode in tqdm(range(num_episodes), desc="Training PPO router"):
            data_point = random.choice(data_points)
            question = data_point.question
            query_peer_id = random.choice(range(self.num_peers))
            question_topic = self.peers[query_peer_id].parse_topic(question, self.all_topics)

            current_peer_id = query_peer_id
            visited_ids = {current_peer_id}
            transitions = []
            done = False

            for hop in range(max_ttl):
                current_score = self.get_peer_relevance_score(current_peer_id, question)

                if current_score > query_confidence_threshold:
                    if transitions:
                        transitions[-1]["reward"] += reward_hit
                        transitions[-1]["done"] = 1.0
                    done = True
                    break

                neighbor_ids = list(self.network.neighbors(current_peer_id))
                if len(neighbor_ids) == 0:
                    if transitions:
                        transitions[-1]["reward"] += reward_miss
                        transitions[-1]["done"] = 1.0
                    done = True
                    break

                state_features = self.build_ppo_state_features(
                    current_peer_id=current_peer_id,
                    hop=hop,
                    max_ttl=max_ttl,
                    current_score=current_score,
                    visited_count=len(visited_ids),
                )
                selected_neighbors, candidate_features, mask = self.build_ppo_candidate_features(
                    question=question,
                    question_topic=question_topic,
                    neighbor_ids=neighbor_ids,
                    visited_ids=visited_ids,
                    max_candidates=router.cfg.max_candidates,
                )
                if len(selected_neighbors) == 0:
                    break

                state_tensor = torch.tensor(state_features, dtype=torch.float, device=router.device)
                candidates_tensor = torch.tensor(candidate_features, dtype=torch.float, device=router.device)
                mask_tensor = torch.tensor(mask, dtype=torch.bool, device=router.device)

                action, log_prob, value, _ = router.act(
                    state=state_tensor,
                    candidates=candidates_tensor,
                    mask=mask_tensor,
                    deterministic=False,
                )

                if action >= len(selected_neighbors):
                    action = len(selected_neighbors) - 1
                next_peer_id = selected_neighbors[action]
                
                next_score = self.get_peer_relevance_score(next_peer_id, question)
                topic_reward = topic_match_bonus if (question_topic and question_topic in self.peer_topics[next_peer_id]) else 0.0
                revisit_cost = revisit_penalty if next_peer_id in visited_ids else 0.0
                progress_reward = progress_weight * (next_score - current_score)
                reward = (
                    relevance_weight * next_score
                    + progress_reward
                    + topic_reward
                    - message_penalty
                    - hop_penalty
                    - revisit_cost
                )

                step_done = 1.0 if next_score > query_confidence_threshold else 0.0
                if step_done:
                    reward += reward_hit
                    

                transitions.append({
                    "state": state_tensor.detach().cpu(),
                    "candidates": candidates_tensor.detach().cpu(),
                    "mask": mask_tensor.detach().cpu(),
                    "action": action,
                    "log_prob": log_prob,
                    "value": value,
                    "reward": reward,
                  
                    "done": step_done,
                })

                current_peer_id = next_peer_id
                visited_ids.add(current_peer_id)
                
                if step_done:
                    done = True
                    break

            if not done and transitions:
                transitions[-1]["reward"] += reward_miss
                transitions[-1]["done"] = 1.0

            router.update(transitions)

            if (episode + 1) % 50 == 0:
                logger.debug(f"PPO training episode {episode + 1}/{num_episodes} finished")

    def ppo_query(
            self,
            question: str,
            router: PPORouter,
            query_peer_id: Optional[int] = None,
            query_confidence_threshold: float = 0.5,
            max_ttl: int = 6,
    ) -> RAGAnswer:
        """
        Queries the network using a trained PPO policy to select the next hop.
        """
        num_messages = 0

        if query_peer_id is None:
            query_peer_id = random.choice(range(self.num_peers))
            logger.debug(f"Randomly selected starting peer: {query_peer_id}")

        question_topic = self.peers[query_peer_id].parse_topic(question, self.all_topics)
        current_peer_id = query_peer_id
        visited_ids = {current_peer_id}

        for hop in range(max_ttl):
            current_answer, relevant_knowledge, relevant_score, is_query_hit = \
                self.peers[current_peer_id].query(question, query_confidence_threshold)
            num_messages += 1

            if current_answer is not None:
                return RAGAnswer(
                    answer=str(current_answer),
                    relevant_knowledge=relevant_knowledge,
                    relevant_score=relevant_score,
                    num_hops=hop,
                    num_messages=num_messages,
                    is_query_hit=is_query_hit
                )

            neighbor_ids = list(self.network.neighbors(current_peer_id))
            if len(neighbor_ids) == 0:
                break

            current_score = self.get_peer_relevance_score(current_peer_id, question)
            state_features = self.build_ppo_state_features(
                current_peer_id=current_peer_id,
                hop=hop,
                max_ttl=max_ttl,
                current_score=current_score,
                visited_count=len(visited_ids),
            )
            selected_neighbors, candidate_features, mask = self.build_ppo_candidate_features(
                question=question,
                question_topic=question_topic,
                neighbor_ids=neighbor_ids,
                visited_ids=visited_ids,
                max_candidates=router.cfg.max_candidates,
            )
            if len(selected_neighbors) == 0:
                break

            state_tensor = torch.tensor(state_features, dtype=torch.float, device=router.device)
            candidates_tensor = torch.tensor(candidate_features, dtype=torch.float, device=router.device)
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=router.device)

            action, _, _, _ = router.act(
                state=state_tensor,
                candidates=candidates_tensor,
                mask=mask_tensor,
                deterministic=True,
            )

            if action >= len(selected_neighbors):
                break

            next_peer_id = selected_neighbors[action]
            if next_peer_id in visited_ids and len(visited_ids) < self.num_peers:
                non_visited = [nid for nid in selected_neighbors if nid not in visited_ids]
                if non_visited:
                    next_peer_id = non_visited[0]

            current_peer_id = next_peer_id
            visited_ids.add(current_peer_id)

        return RAGAnswer(
            answer="",
            relevant_knowledge="",
            relevant_score=0.0,
            num_hops=max_ttl,
            num_messages=num_messages,
            is_query_hit=False
        )

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

    def topic_query(
            self, 
            question: str, 
            query_peer_id: Optional[int] = None, 
            num_query_neighbor: int = 2,
            query_confidence_threshold: float = 0.5,
            max_ttl: int = 6
    ) -> RAGAnswer:
        """
        Topic-based network query

        Args:
            question: The question to ask.
            query_peer_id: The ID of the peer initiating the query. If None, a random peer is selected.
            num_query_neighbor: The maximum number of neighbors to query at each hop.
            query_confidence_threshold: The confidence threshold required for an answer to be accepted.
            max_ttl: The time-to-live for the query (maximum number of hops).

        Returns:
            The answer to the question if found, otherwise None.
        """
        # Track number of messages (queries) sent
        num_messages = 0

        # Initialize query peer
        if query_peer_id is None:
            query_peer_id = random.choice(range(self.num_peers))
            logger.debug(f"Randomly selected starting peer: {query_peer_id}")
        
        logger.debug(f"Starting topic-based search from peer {query_peer_id}")
        logger.debug(f"Parameters: num_neighbors={num_query_neighbor}, max_ttl={max_ttl}")

        # Determine the topic of the question first
        question_topic = self.peers[query_peer_id].parse_topic(question, self.all_topics)
        logger.debug(f"Parsed question topic: {question_topic}")

        # Keep track of visited peers to avoid cycles
        visited_ids = {query_peer_id}

        # Queue for BFS: (peer_id, hop)
        queue = deque([(query_peer_id, 0)])

        # Perform topic-based search
        while queue:
            current_peer_id, hop = queue.popleft()

            if hop >= max_ttl:
                continue

            logger.debug(f"Hop {hop}: Querying peer {current_peer_id}")

            # Query the current peer and track message
            current_answer, relevant_knowledge, relevant_score, is_query_hit = \
                self.peers[current_peer_id].query(question, query_confidence_threshold)
            num_messages += 1

            if current_answer is not None:
                logger.debug(f"Answer found at peer {current_peer_id} after {hop} hops")
                logger.debug(f"Total messages sent: {num_messages}")
                answer = RAGAnswer(
                    answer=str(current_answer),
                    relevant_knowledge=relevant_knowledge,
                    relevant_score=relevant_score,
                    num_hops=hop,
                    num_messages=num_messages,
                    is_query_hit=is_query_hit
                )
                return answer  # Return the answer if found

            # Get and prioritize neighbors
            current_neighbor_ids = list(self.network.neighbors(current_peer_id))
            picked_neighbor_ids = []

            if len(current_neighbor_ids) > num_query_neighbor:
                # Find topic-matched neighbors
                topic_matched_neighbors = []
                for neighbor_id in current_neighbor_ids:
                    if question_topic in self.peer_topics[neighbor_id]:
                        topic_matched_neighbors.append(neighbor_id)

                # Add topic-matched peers to the neighbor list of current peer
                # So that in the future, there is no need to flood other peers
                logger.debug(f"Found {len(topic_matched_neighbors)} topic-matched neighbors, update neighbor list")
                for neighbor_id in topic_matched_neighbors:
                    self.network.add_edge(query_peer_id, neighbor_id)

                # Select neighbors based on topic matching
                if len(topic_matched_neighbors) > num_query_neighbor:
                    picked_neighbor_ids = random.sample(topic_matched_neighbors, num_query_neighbor)
                else:
                    picked_neighbor_ids = topic_matched_neighbors
                    # Fill remaining slots with random neighbors
                    remaining_neighbor_ids = list(set(current_neighbor_ids) - set(picked_neighbor_ids))
                    remaining_num = min(num_query_neighbor - len(picked_neighbor_ids), len(remaining_neighbor_ids))
                    picked_neighbor_ids += random.sample(remaining_neighbor_ids, remaining_num)
            else:
                picked_neighbor_ids = current_neighbor_ids

            logger.debug(f"Selected neighbors for next hop: {picked_neighbor_ids}")

            # Add picked neighbors to the queue
            for neighbor_id in picked_neighbor_ids:
                if neighbor_id not in visited_ids:
                    visited_ids.add(neighbor_id)
                    queue.append((neighbor_id, hop + 1))
                    logger.debug(f"Added neighbor {neighbor_id} to queue at Hop {hop + 1}")

        logger.debug(f"Search failed after {max_ttl} hops")
        logger.debug(f"Total messages sent: {num_messages}")

        # Return empty answer if no result found
        answer = RAGAnswer(
            answer="",
            relevant_knowledge="",
            relevant_score=0.0,
            num_hops=max_ttl,
            num_messages=num_messages,
            is_query_hit=False
        )
        return answer

    def random_walk_query(
            self,
            question: str,
            query_peer_id: Optional[int] = None,
            query_confidence_threshold: float = 0.5,
            max_ttl: int = 6,
            restart_probability: float = 0.1
    ) -> RAGAnswer:
        """
        Queries the network using a random walk algorithm with restart probability.
        
        Args:
            question: The question to ask
            query_peer_id: Starting peer ID (random if None)
            query_confidence_threshold: Minimum confidence required for an answer
            max_ttl: Maximum number of steps in the walk
            restart_probability: Probability of restarting from the initial peer
        
        Returns:
            RAGAnswer object containing the response
        """
        # Track number of messages (queries) sent
        num_messages = 0

        # Initialize starting peer
        if query_peer_id is None:
            query_peer_id = random.choice(range(self.num_peers))
            logger.debug(f"Randomly selected starting peer: {query_peer_id}")
        
        current_peer_id = query_peer_id
        initial_peer_id = query_peer_id
        
        # Keep track of path for debugging
        path = [current_peer_id]

        logger.debug(f"Starting random walk search from peer {initial_peer_id}")
        logger.debug(f"Parameters: max_ttl={max_ttl}, restart_prob={restart_probability}")
        
        # Perform random walk
        for hop in range(max_ttl):
            logger.debug(f"Hop {hop}: Querying peer {current_peer_id}")
            
            # Query current peer
            current_answer, relevant_knowledge, relevant_score, is_query_hit = \
                self.peers[current_peer_id].query(question, query_confidence_threshold)
            num_messages += 1
            
            # Return if answer found
            if current_answer is not None:
                logger.debug(f"Answer found at peer {current_peer_id} after {hop} hops")
                logger.debug(f"Total messages sent: {num_messages}")
                return RAGAnswer(
                    answer=str(current_answer),
                    relevant_knowledge=relevant_knowledge,
                    relevant_score=relevant_score,
                    num_hops=hop,
                    num_messages=num_messages,
                    is_query_hit=is_query_hit
                )
            
            # Decide whether to restart
            if random.random() < restart_probability:
                current_peer_id = initial_peer_id
                logger.debug(f"Random restart to initial peer: {initial_peer_id}")
            else:
                # Get neighbors and randomly select next peer
                neighbors = list(self.network.neighbors(current_peer_id))
                if not neighbors:
                    logger.debug(f"Dead end at peer {current_peer_id}, restarting")
                    current_peer_id = initial_peer_id
                else:
                    current_peer_id = random.choice(neighbors)
                    logger.debug(f"Moving to neighbor peer {current_peer_id}")
            
            path.append(current_peer_id)
        
        logger.debug(f"Search failed after {max_ttl} hops")
        logger.debug(f"Path taken: {path}")
        logger.debug(f"Total messages sent: {num_messages}")
        
        # Return empty answer if no result found
        return RAGAnswer(
            answer="",
            relevant_knowledge="",
            relevant_score=0.0,
            num_hops=max_ttl,
            num_messages=num_messages,
            is_query_hit=False
        )

    def flooding_query(
            self,
            question: str,
            query_peer_id: Optional[int] = None,
            query_confidence_threshold: float = 0.5,
            max_ttl: int = 6
    ) -> RAGAnswer:
        """
        Queries the network using a flooding algorithm.
        
        Args:
            question: The question to ask
            query_peer_id: Starting peer ID (random if None)
            query_confidence_threshold: Minimum confidence required for an answer
            max_ttl: Maximum time-to-live (network depth to explore)
        
        Returns:
            RAGAnswer object containing the response
        """
        # Track number of messages (queries) sent
        num_messages = 0

        # Initialize starting peer
        if query_peer_id is None:
            query_peer_id = random.choice(range(self.num_peers))
            logger.debug(f"Randomly selected starting peer: {query_peer_id}")
        
        logger.debug(f"Starting flooding search from peer {query_peer_id}")
        logger.debug(f"Parameters: max_ttl={max_ttl}")
        
        # Use set for visited nodes to avoid cycles
        visited_ids = {query_peer_id}
        
        # Queue for BFS: (peer_id, hop)
        queue = deque([(query_peer_id, 0)])
        
        while queue:
            current_peer_id, hop = queue.popleft()
            
            if hop >= max_ttl:
                continue

            logger.debug(f"Flooding at Hop {hop}, peer: {current_peer_id}")
            
            # Query current peer and track message
            current_answer, relevant_knowledge, relevant_score, is_query_hit = \
                self.peers[current_peer_id].query(question, query_confidence_threshold)
            num_messages += 1
            
            # Return if answer found
            if current_answer is not None:
                logger.debug(f"Answer found at peer {current_peer_id}, Hop {hop}")
                logger.debug(f"Total messages sent: {num_messages}")
                return RAGAnswer(
                    answer=str(current_answer),
                    relevant_knowledge=relevant_knowledge,
                    relevant_score=relevant_score,
                    num_hops=hop,
                    num_messages=num_messages,
                    is_query_hit=is_query_hit
                )
            
            # Add all unvisited neighbors to queue
            neighbors = self.network.neighbors(current_peer_id)
            for neighbor_id in neighbors:
                if neighbor_id not in visited_ids:
                    visited_ids.add(neighbor_id)
                    queue.append((neighbor_id, hop + 1))
                    logger.debug(f"Added neighbor {neighbor_id} to queue at Hop {hop + 1}")
        
        # Log search statistics
        logger.debug("Search failed to find answer")
        logger.debug(f"Total messages sent: {num_messages}")
        
        # Return empty answer if no result found
        return RAGAnswer(
            answer="",
            relevant_knowledge="",
            relevant_score=0.0,
            num_hops=max_ttl,
            num_messages=num_messages,
            is_query_hit=False
        )


class CRAGNetwork:
    def __init__(
            self, 
            llm_url: str, 
            llm_name: str, 
            llm_num_ctx: int,
            llm_seed: int, 
            embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
        ):
        self.text_embedding_model = SentenceTransformer(embedding_model)
        self.peer = Peer(0, llm_url, llm_name, llm_num_ctx, llm_seed, self.text_embedding_model)

    def init_knowledge(self, data_points: List[Datapoint]):
        """
        Distributes data points to peers based on their topics (Uniform).

        Args:
            data_points: A list of Datapoint objects.
        """
        # Distribute data points to peers based on assigned topics
        for data_point in tqdm(data_points, desc=f"Distributing data points to peers"):
            self.peer.add_knowledge(data_point)

    def query(
            self,
            question: str,
            query_confidence_threshold: float = 0.5
    ) -> RAGAnswer:
        """
        Queries the network using a random walk algorithm with restart probability.
        
        Args:
            question: The question to ask
            query_confidence_threshold: Minimum confidence required for an answer
        
        Returns:
            RAGAnswer object containing the response
        """
        # Query current peer
        current_answer, relevant_knowledge, relevant_score, is_query_hit = \
            self.peer.query(question, query_confidence_threshold)

        # Return if answer found
        if current_answer is not None:
            logger.debug(f"Answer found at peer 0")
            return RAGAnswer(
                answer=str(current_answer),
                relevant_knowledge=relevant_knowledge,
                relevant_score=relevant_score,
                num_hops=0,
                num_messages=0,
                is_query_hit=is_query_hit
            )
            
        logger.debug(f"Search failed")
        
        # Return empty answer if no result found
        return RAGAnswer(
            answer="",
            relevant_knowledge="",
            relevant_score=0.0,
            num_hops=0,
            num_messages=0,
            is_query_hit=False
        )


class NoRAGNetwork:
    def __init__(
            self, 
            llm_url: str, 
            llm_name: str, 
            llm_num_ctx: int,
            llm_seed: int
        ):
        self.peer = Peer(0, llm_url, llm_name, llm_num_ctx, llm_seed)

    def init_knowledge(self, data_points: List[Datapoint]):
        """
        No need to initialize knowledge for No RAG mode

        Args:
            data_points: A list of Datapoint objects.
        """
        pass

    def query(
            self,
            question: str
    ) -> RAGAnswer:
        """
        Queries the network using a random walk algorithm with restart probability.

        Args:
            question: The question to ask
            query_confidence_threshold: Minimum confidence required for an answer

        Returns:
            RAGAnswer object containing the response
        """
        # Query current peer
        generated_answer, relevant_knowledge, relevant_score, is_query_hit = \
            self.peer.query_no_rag(question)

        # Return empty answer if no result found
        return RAGAnswer(
            answer=str(generated_answer),
            relevant_knowledge=relevant_knowledge,
            relevant_score=relevant_score,
            num_hops=0,
            num_messages=0,
            is_query_hit=is_query_hit
        )
