from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from modules.ppo_router import PPORouter
from modules.rag_network import DRAGNetwork


@dataclass
class MIAResult:
    auc: float
    attack_advantage: float
    best_threshold: float
    member_mean_score: float
    nonmember_mean_score: float
    num_member_samples: int
    num_nonmember_samples: int


class MembershipInferenceAttack:
    """Threshold-based MIA over query confidence/relevance scores."""

    @staticmethod
    def _auc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5

        # Mann–Whitney U based AUC estimate
        wins = 0.0
        for p in pos:
            wins += np.sum(p > neg)
            wins += 0.5 * np.sum(p == neg)
        return float(wins / (len(pos) * len(neg)))

    @staticmethod
    def _best_attack_advantage(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
        thresholds = np.unique(scores)
        if thresholds.size == 0:
            return 0.0, 0.0

        best_adv = -1.0
        best_thr = float(thresholds[0])

        positives = labels == 1
        negatives = labels == 0

        for thr in thresholds:
            pred_member = scores >= thr
            tpr = float(np.mean(pred_member[positives])) if np.any(positives) else 0.0
            fpr = float(np.mean(pred_member[negatives])) if np.any(negatives) else 0.0
            advantage = tpr - fpr
            if advantage > best_adv:
                best_adv = advantage
                best_thr = float(thr)

        return float(max(best_adv, 0.0)), best_thr

    def evaluate(self, member_scores: List[float], nonmember_scores: List[float]) -> Optional[MIAResult]:
        if len(member_scores) == 0 or len(nonmember_scores) == 0:
            return None

        member = np.array(member_scores, dtype=float)
        nonmember = np.array(nonmember_scores, dtype=float)

        labels = np.concatenate([np.ones_like(member), np.zeros_like(nonmember)])
        scores = np.concatenate([member, nonmember])

        auc = self._auc_from_scores(labels, scores)
        adv, thr = self._best_attack_advantage(labels, scores)

        return MIAResult(
            auc=auc,
            attack_advantage=adv,
            best_threshold=thr,
            member_mean_score=float(np.mean(member)),
            nonmember_mean_score=float(np.mean(nonmember)),
            num_member_samples=int(member.shape[0]),
            num_nonmember_samples=int(nonmember.shape[0]),
        )


def collect_ppo_membership_scores(
    rag_net: DRAGNetwork,
    router: PPORouter,
    questions: List[str],
    query_confidence_threshold: float,
    max_ttl: int,
) -> List[float]:
    scores: List[float] = []
    for q in questions:
        ans = rag_net.ppo_query(
            question=q,
            router=router,
            query_confidence_threshold=query_confidence_threshold,
            max_ttl=max_ttl,
        )
        # relevance_score is a natural confidence signal for MIA
        scores.append(float(ans.relevant_score))
    return scores
