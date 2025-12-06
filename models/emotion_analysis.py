"""增强版情绪分析模块，支持多种模型选择。"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
import onnxruntime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import joblib
import jieba
import jieba.posseg as pseg

import database
from schemas import AffectiveAnalysisRequest, AffectiveAnalysisResponse, AffectiveState, SentimentRequest, SentimentResponse

# 模型类型枚举
class ModelType(str, Enum):
    TRANSFORMER = "transformer"  # 预训练Transformer模型
    ENSEMBLE = "ensemble"        # 集成模型
    DEEP_LEARNING = "deep_learning"  # 深度学习模型
    RULE_BASED = "rule_based"    # 回退规则
    ONNX = "onnx"               # 优化推理模型

@dataclass
class ModelConfig:
    """模型配置"""
    model_type: ModelType
    model_path: str
    threshold: float = 0.7
    use_gpu: bool = False
    batch_size: int = 32
    max_length: int = 512

class SentimentAnalyzer:
    """高级情感分析器"""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig(
            model_type=ModelType.TRANSFORMER,
            model_path="bert-base-chinese-sentiment",  # 或使用其他预训练模型
            threshold=0.7
        )
        self.model = None
        self.tokenizer = None
        self.vectorizer = None
        self._load_model()
        
    def _load_model(self):
        """加载模型"""
        if self.config.model_type == ModelType.TRANSFORMER:
            self._load_transformer_model()
        elif self.config.model_type == ModelType.ENSEMBLE:
            self._load_ensemble_model()
        elif self.config.model_type == ModelType.ONNX:
            self._load_onnx_model()
            
    def _load_transformer_model(self):
        """加载预训练Transformer模型"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                use_fast=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.config.model_path,
                num_labels=3
            )
            
            if self.config.use_gpu and torch.cuda.is_available():
                self.model = self.model.cuda()
                
            self.model.eval()
            print(f"✅ 加载Transformer模型: {self.config.model_path}")
            
        except Exception as e:
            print(f"❌ 加载Transformer模型失败: {e}")
            self._fallback_to_ensemble()
            
    def _load_ensemble_model(self):
        """加载集成模型（BERT + SVM + 规则）"""
        # 这里可以加载多个子模型
        self.ensemble_models = {
            'bert': pipeline(
                "sentiment-analysis",
                model="uer/roberta-base-finetuned-jd-binary-chinese",
                return_all_scores=True
            ),
            'svm': joblib.load("models/svm_sentiment.pkl"),
            'vectorizer': joblib.load("models/tfidf_vectorizer.pkl")
        }
        print("✅ 加载集成模型")
        
    def _load_onnx_model(self):
        """加载ONNX优化模型"""
        self.onnx_session = onnxruntime.InferenceSession(
            self.config.model_path,
            providers=['CUDAExecutionProvider' if self.config.use_gpu else 'CPUExecutionProvider']
        )
        print(f"✅ 加载ONNX模型: {self.config.model_path}")
        
    def _fallback_to_ensemble(self):
        """回退到集成模型"""
        # 实现简单的特征工程
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2)
        )
        print("🔄 使用回退集成模型")
        
    def _extract_features(self, text: str) -> Dict:
        """提取文本特征"""
        features = {}
        
        # 1. 情感词典特征
        positive_words = {"好", "满意", "喜欢", "清晰", "有趣", "赞", "棒"}
        negative_words = {"差", "糟", "难", "晦涩", "失望", "生气", "不满"}
        
        features['positive_count'] = sum(1 for w in jieba.lcut(text) if w in positive_words)
        features['negative_count'] = sum(1 for w in jieba.lcut(text) if w in negative_words)
        features['sentiment_ratio'] = (
            (features['positive_count'] - features['negative_count']) / 
            max(len(text.split()), 1)
        )
        
        # 2. 文本长度特征
        features['char_length'] = len(text)
        features['word_count'] = len(jieba.lcut(text))
        
        # 3. 标点特征
        features['exclamation_count'] = text.count('!') + text.count('！')
        features['question_count'] = text.count('?') + text.count('？')
        
        # 4. 程度副词
        intensity_words = {"非常", "很", "特别", "极其", "超级"}
        features['intensity_words'] = sum(1 for w in jieba.lcut(text) if w in intensity_words)
        
        return features
        
    def analyze_with_transformer(self, text: str) -> Dict[str, float]:
        """使用Transformer模型分析"""
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        if self.config.use_gpu and torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probabilities[0].cpu().numpy()
            
        return {
            "负面": float(probs[0]),
            "中性": float(probs[1]),
            "正面": float(probs[2])
        }
        
    def analyze_with_ensemble(self, text: str) -> Dict[str, float]:
        """集成模型分析"""
        # 获取BERT预测
        bert_result = self.ensemble_models['bert'](text)[0]
        bert_probs = {item['label']: item['score'] for item in bert_result}
        
        # 获取SVM预测
        if hasattr(self, 'vectorizer'):
            features = self.vectorizer.transform([text])
            svm_proba = self.ensemble_models['svm'].predict_proba(features)[0]
            
            # 融合结果
            final_probs = {}
            labels = ["负面", "中性", "正面"]
            for i, label in enumerate(labels):
                bert_score = bert_probs.get(label, 0.0)
                svm_score = svm_proba[i] if i < len(svm_proba) else 0.0
                # 加权平均
                final_probs[label] = 0.6 * bert_score + 0.4 * svm_score
                
            return final_probs
            
        return bert_probs
        
    def analyze(self, text: str) -> Dict[str, float]:
        """主分析函数"""
        if not text.strip():
            return {"负面": 0.333, "中性": 0.334, "正面": 0.333}
            
        try:
            if self.config.model_type == ModelType.TRANSFORMER and self.model:
                return self.analyze_with_transformer(text)
            elif self.config.model_type == ModelType.ENSEMBLE:
                return self.analyze_with_ensemble(text)
            elif self.config.model_type == ModelType.ONNX:
                return self._analyze_with_onnx(text)
        except Exception as e:
            print(f"⚠️ 模型分析失败，使用回退规则: {e}")
            
        # 回退到规则
        return self._rule_based_analysis(text)
        
    def _rule_based_analysis(self, text: str) -> Dict[str, float]:
        """规则回退分析"""
        features = self._extract_features(text)
        
        # 基于特征的启发式规则
        if features['sentiment_ratio'] > 0.1:
            return {"负面": 0.1, "中性": 0.2, "正面": 0.7}
        elif features['sentiment_ratio'] < -0.1:
            return {"负面": 0.7, "中性": 0.2, "正面": 0.1}
        else:
            return {"负面": 0.2, "中性": 0.6, "正面": 0.2}
            
    def _analyze_with_onnx(self, text: str) -> Dict[str, float]:
        """ONNX推理"""
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config.max_length,
            return_tensors="np"
        )
        
        ort_inputs = {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask']
        }
        
        outputs = self.onnx_session.run(None, ort_inputs)
        probs = torch.nn.functional.softmax(torch.tensor(outputs[0]), dim=-1)[0]
        
        return {
            "负面": float(probs[0]),
            "中性": float(probs[1]),
            "正面": float(probs[2])
        }

# 全局分析器实例
_global_analyzer = None

def get_analyzer() -> SentimentAnalyzer:
    """获取或创建分析器实例"""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = SentimentAnalyzer()
    return _global_analyzer

def analyze_sentiment(payload: SentimentRequest) -> SentimentResponse:
    """情感分析接口"""
    analyzer = get_analyzer()
    prob_dict = analyzer.analyze(payload.text)
    
    # 获取最高概率标签
    label = max(prob_dict.items(), key=lambda x: x[1])[0]
    
    # 计算置信度
    confidence = prob_dict[label]
    
    return SentimentResponse(
        request_id=payload.request_id,
        probabilities={k: round(v, 3) for k, v in prob_dict.items()},
        label=label,
        confidence=round(confidence, 3),
        model_version="enhanced-v1.0",
        metadata={
            "model_type": analyzer.config.model_type.value,
            "text_length": len(payload.text),
            "timestamp": database.get_current_timestamp()
        }
    )

# 情感状态分析增强
class AffectiveAnalyzer:
    """高级情感状态分析器"""
    
    def __init__(self):
        # 情感转移矩阵
        self.transition_matrix = self._load_transition_matrix()
        
    def _load_transition_matrix(self) -> Dict:
        """加载情感转移概率矩阵"""
        return {
            "frustration": {"next_emotions": ["confident", "calm"], "probabilities": [0.6, 0.4]},
            "bored": {"next_emotions": ["engaged", "neutral"], "probabilities": [0.7, 0.3]},
            "confident": {"next_emotions": ["excited", "confident"], "probabilities": [0.8, 0.2]},
        }
        
    def analyze_emotion_trend(self, 
                            student_id: str, 
                            recent_emotions: List[Dict]) -> Dict:
        """分析情感趋势"""
        if not recent_emotions:
            return {"trend": "stable", "confidence": 0.5}
            
        # 计算情感变化
        emotions = [e['emotion'] for e in recent_emotions]
        intensities = [e['intensity'] for e in recent_emotions]
        
        # 线性回归判断趋势
        if len(emotions) >= 3:
            x = np.arange(len(intensities))
            slope = np.polyfit(x, intensities, 1)[0]
            
            if slope > 0.1:
                return {"trend": "improving", "confidence": abs(slope)}
            elif slope < -0.1:
                return {"trend": "worsening", "confidence": abs(slope)}
                
        return {"trend": "stable", "confidence": 0.5}
        
    def generate_personalized_nudges(self, 
                                   student_id: str,
                                   emotion_history: List[Dict],
                                   current_emotion: str) -> List[str]:
        """生成个性化干预策略"""
        nudges = []
        
        # 基于历史数据的个性化策略
        history_stats = database.get_student_emotion_stats(student_id)
        
        if current_emotion in ["frustration", "anxious"]:
            if history_stats.get('frustration_frequency', 0) > 0.3:
                nudges.append("检测到该学生频繁受挫，建议调整任务难度梯度")
                nudges.append("提供分步演示和即时反馈")
            else:
                nudges.append("给出分步提示或降低任务难度")
                
        elif current_emotion in ["bored", "disengaged"]:
            nudges.append("增加挑战性任务或引入游戏化元素")
            nudges.append("安排2分钟休息或切换轻量任务")
            
        elif current_emotion in ["confident", "excited"]:
            nudges.append("趁势引入高级概念或扩展任务")
            nudges.append("鼓励学生帮助同伴或分享思路")
            
        return nudges

def analyze_affective_state(payload: AffectiveAnalysisRequest) -> AffectiveAnalysisResponse:
    """增强版情感状态分析"""
    # 1. 整合多种情感信号
    emotion_weights = {}
    emotion_history = []
    
    for sig in payload.affective_signals:
        emotion = sig.emotion.lower()
        emotion_weights[emotion] = emotion_weights.get(emotion, 0.0) + sig.intensity
        
        # 记录历史用于趋势分析
        emotion_history.append({
            "emotion": emotion,
            "intensity": sig.intensity,
            "timestamp": database.get_current_timestamp()
        })
    
    # 2. 计算主导情感
    if not emotion_weights:
        dominant, confidence = "neutral", 0.2
    else:
        dominant, score = max(emotion_weights.items(), key=lambda item: item[1])
        total = sum(emotion_weights.values())
        confidence = score / total if total else 0.0
        
    # 3. 分析情感趋势
    affective_analyzer = AffectiveAnalyzer()
    trend_analysis = affective_analyzer.analyze_emotion_trend(
        payload.student_id, 
        emotion_history
    )
    
    # 4. 生成个性化消息
    student_profile = database.get_student_profile(payload.student_id)
    
    if dominant in {"frustration", "anxious", "沮丧"}:
        if trend_analysis['trend'] == "worsening":
            message = f"{payload.student_id} 在「{payload.current_task}」中挫折感持续增强，建议立即干预。"
        else:
            message = f"检测到 {payload.student_id} 在「{payload.current_task}」中可能感到挫折。"
            
    elif dominant in {"bored", "disengaged"}:
        engagement_level = student_profile.get('engagement_level', 'medium')
        if engagement_level == 'low':
            message = f"{payload.student_id} 参与度较低，建议调整教学策略。"
        else:
            message = f"{payload.student_id} 的情绪趋于低唤醒，可尝试更具挑战性的任务。"
            
    elif dominant in {"confident", "excited", "积极"}:
        if trend_analysis['confidence'] > 0.7:
            message = f"{payload.student_id} 状态非常积极，是深入学习的良机。"
        else:
            message = f"{payload.student_id} 状态积极，可趁势加强学习。"
    else:
        message = f"{payload.student_id} 情绪平稳。"
        
    # 5. 结合近期表现
    if payload.recent_performance:
        performance_score = analyze_performance(payload.recent_performance)
        if performance_score < 0.5 and dominant not in ["frustration", "anxious"]:
            message += f" 但近期表现({performance_score:.1%})提示需要关注理解深度。"
        else:
            message += f" 近期表现({performance_score:.1%})良好。"
    
    # 6. 记录分析结果
    analysis_id = database.log_emotion_analysis(
        student_id=payload.student_id,
        dominant_emotion=dominant,
        confidence=confidence,
        trend=trend_analysis['trend'],
        task=payload.current_task,
        message=message
    )
    
    # 7. 生成个性化干预策略
    nudges = affective_analyzer.generate_personalized_nudges(
        payload.student_id,
        emotion_history,
        dominant
    )
    
    # 8. 构建脚手架调整建议
    regulation_strategies = [
        f"根据{dominant}情绪状态，调整「{payload.current_task}」脚手架层级。",
        "使用情感智能反馈语句回应学生。"
    ]
    
    if trend_analysis['trend'] == "worsening":
        regulation_strategies.append("考虑降低难度或提供更多示例。")
        
    # 9. 构建响应
    state = AffectiveState(
        dominant_emotion=dominant,
        confidence=round(confidence, 3),
        message=message,
        trend=trend_analysis['trend'],
        trend_confidence=round(trend_analysis['confidence'], 3),
        regulation_strategies=regulation_strategies,
        analysis_id=analysis_id
    )
    
    return AffectiveAnalysisResponse(
        request_id=payload.request_id,
        student_id=payload.student_id,
        state=state,
        nudges=nudges,
        trend_analysis=trend_analysis,
        model_version="enhanced-v1.0",
        metadata={
            "emotion_count": len(payload.affective_signals),
            "task_context": payload.current_task,
            "performance_included": payload.recent_performance is not None
        }
    )

def analyze_performance(performance_text: str) -> float:
    """分析表现文本，返回0-1的分数"""
    # 这里可以扩展为更复杂的表现分析
    positive_indicators = {"完成", "正确", "优秀", "进步", "理解"}
    negative_indicators = {"未完成", "错误", "困难", "不理解", "卡住"}
    
    score = 0.5
    for word in positive_indicators:
        if word in performance_text:
            score += 0.1
    for word in negative_indicators:
        if word in performance_text:
            score -= 0.1
            
    return max(0.0, min(1.0, score))
