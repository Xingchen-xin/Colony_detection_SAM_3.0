# ============================================================================
# 8. colony_analysis/analysis/scoring.py - 评分系统
# ============================================================================

from typing import Any, Dict


class ScoringSystem:
    """菌落评分与分类系统"""

    def calculate_scores(self, features: Dict[str, Any]) -> Dict[str, float]:
        """计算菌落各项评分"""
        scores = {}

        # 计算气生菌丝评分
        aerial_score = self._calculate_aerial_score(features)
        scores["aerial_mycelium_score"] = aerial_score

        # 计算代谢产物评分
        metabolite_score = self._calculate_metabolite_score(features)
        scores["metabolite_score"] = metabolite_score

        # 计算形态学评分
        morphology_score = self._calculate_morphology_score(features)
        scores["morphology_score"] = morphology_score

        # 计算综合评分
        overall_score = (
            0.4 * aerial_score + 0.4 * metabolite_score + 0.2 * morphology_score
        )
        scores["overall_score"] = overall_score

        return scores

    def _calculate_aerial_score(self, features: Dict[str, Any]) -> float:
        """计算气生菌丝评分"""
        aerial_ratio = features.get("morphology_aerial_ratio", 0)
        aerial_height = features.get("morphology_aerial_height_mean", 0)

        # 比例评分(0-50分)
        ratio_score = min(aerial_ratio / 0.8, 1.0) * 50

        # 高度评分(0-50分)
        height_score = min(aerial_height / 200, 1.0) * 50

        total_score = ratio_score + height_score
        return min(total_score, 100)

    def _calculate_metabolite_score(self, features: Dict[str, Any]) -> float:
        """计算代谢产物评分"""
        blue_ratio = features.get("metabolite_blue_ratio", 0)
        red_ratio = features.get("metabolite_red_ratio", 0)
        blue_intensity = features.get("metabolite_blue_intensity_mean", 0)
        red_intensity = features.get("metabolite_red_intensity_mean", 0)

        # 蓝色素评分(0-50分)
        blue_score = 0
        if blue_ratio > 0:
            coverage_score = min(blue_ratio * 2, 1.0) * 30
            intensity_score = min(blue_intensity / 150, 1.0) * 20
            blue_score = coverage_score + intensity_score

        # 红色素评分(0-50分)
        red_score = 0
        if red_ratio > 0:
            coverage_score = min(red_ratio * 2, 1.0) * 30
            intensity_score = min(red_intensity / 150, 1.0) * 20
            red_score = coverage_score + intensity_score

        # 取蓝色和红色的最高分
        pigment_score = max(blue_score, red_score)

        # 如果两种色素都有，给予额外奖励
        if blue_ratio > 0.05 and red_ratio > 0.05:
            pigment_score += 15

        return min(pigment_score, 100)

    def _calculate_morphology_score(self, features: Dict[str, Any]) -> float:
        """计算形态学评分"""
        circularity = features.get("circularity", 0)
        edge_density = features.get("edge_density", 0)
        convexity = features.get("convexity", 0)

        # 形状规则性评分(0-40分)
        shape_score = circularity * 40

        # 边缘复杂度评分(0-30分)
        edge_score = min(edge_density * 3, 1.0) * 30 if edge_density > 0 else 0

        # 紧凑度评分(0-30分)
        compactness_score = min(convexity, 1.0) * 30

        total_score = shape_score + edge_score + compactness_score
        return min(total_score, 100)

    def classify_phenotype(self, features: Dict[str, Any]) -> Dict[str, str]:
        """根据特征对菌落进行表型分类"""
        # 发育状态分类
        aerial_ratio = features.get("morphology_aerial_ratio", 0)
        if aerial_ratio < 0.1:
            development_state = "substrate_mycelium"
        elif aerial_ratio < 0.3:
            development_state = "early_aerial_mycelium"
        elif aerial_ratio < 0.6:
            development_state = "aerial_mycelium"
        else:
            development_state = "mature_spores"

        # 判断是否存在蓝色或红色色素
        has_blue = features.get("metabolite_blue_ratio", 0) > 0
        has_red = features.get("metabolite_red_ratio", 0) > 0

        if has_blue and has_red:
            metabolite_production = "actinorhodin_and_prodigiosin"
        elif has_blue:
            metabolite_production = "actinorhodin"
        elif has_red:
            metabolite_production = "prodigiosin"
        else:
            metabolite_production = "none"

        # 菌落形态分类
        circularity = features.get("circularity", 0)
        edge_density = features.get("edge_density", 0)

        if edge_density > 0.3:
            colony_morphology = "wrinkled"
        elif edge_density > 0.1:
            colony_morphology = "semi_wrinkled"
        else:
            colony_morphology = "smooth"

        return {
            "development_state": development_state,
            "metabolite_production": metabolite_production,
            "colony_morphology": colony_morphology,
        }
