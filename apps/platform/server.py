# -*- coding: utf-8 -*-
"""
材料智能处理平台 - 统一后端服务
整合标注和编码功能
"""

import os
import sys
import logging
import yaml
from typing import List, Optional, Dict, Any
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import json
import asyncio
import copy

# 导入配置模块
from src.config import get_platform_config, get_tokenizer_config, get_semantic_config, get_ner_config

# 导入分词模块
from src.tokenizer_utils.llm_tokenizer import LLMTokenizer
from src.tokenizer_utils.preprocessor import TextPreprocessor

# 导入NER预测器
from src.bert_ner.predictor import PipePredictor


# 导入编码模块
from src.encoder.pipe_encoder import PipeEncoder, get_pipe_encoder
from src.encoder.semantic_matcher import get_semantic_matcher

# 导入第三方集成模块
from src.integrations import get_h3yun_client

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

H3YUN_REVIEW_TASK_SCHEMA_CODE = "D148357c862f0c8cdfa41418c55cef288f8d83c"
H3YUN_REVIEW_TASK_SUBTABLE_CODE = "D148357F17c2e0548b94497f873300934ea06164"
H3YUN_REVIEW_TASK_APP_CODE = "D148357CLDGGL"

# ============================================================
# FastAPI应用
# ============================================================

app = FastAPI(
    title="材料智能处理平台",
    description="提供材料标注和编码功能",
    version="2.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """服务启动事件"""
    platform_config = get_platform_config()
    server_config = platform_config.get("server", {})
    
    # 根据配置决定是否预加载模型
    if server_config.get("preload_models", False):
        logger.info("预加载模型已启用，正在加载...")
        
        # 预加载语义匹配器（包含 SentenceTransformer 模型）
        logger.info("加载语义匹配模型...")
        get_semantic_matcher()
        
        # 预加载管道编码器
        logger.info("加载管道编码器...")
        get_pipe_encoder()
        
        logger.info("模型预加载完成！")
    else:
        logger.info("预加载模型已禁用，将在首次使用时加载")


# ============================================================
# 全局实例
# ============================================================

_tokenizers: Dict[str, LLMTokenizer] = {}
_preprocessor = TextPreprocessor()
_ner_predictor = None
_ner_model_type: str = "bert"


def get_ner_predictor():
    """获取NER预测器实例（惰性加载，根据配置选择模型）"""
    global _ner_predictor, _ner_model_type
    
    ner_config = get_ner_config()
    model_type = ner_config.get("model_type", "bert")
    
    # 如果模型类型改变，需要重新加载
    if _ner_predictor is not None and _ner_model_type == model_type:
        return _ner_predictor
    
    if model_type == "qwen3":
        # 使用 Qwen3-4B 微调模型
        logger.info("使用 Qwen3 NER 模型")
        
        from src.llm_ner.predictor import Qwen3Predictor
        
        qwen3_config = ner_config.get("qwen3", {})
        backend = qwen3_config.get("backend", "ollama")
        
        if backend == "ollama":
            model_name = qwen3_config.get("model_name", "qwen3-pipe")
            ollama_url = qwen3_config.get("ollama_url", "http://localhost:11434")
            logger.info(f"加载 Qwen3 NER 模型: Ollama 后端, 模型: {model_name}")
            _ner_predictor = Qwen3Predictor(
                model_name=model_name,
                backend="ollama",
                ollama_url=ollama_url,
            )
        else:
            model_path = str(PROJECT_ROOT / qwen3_config.get("model_path", "models/qwen3_pipe"))
            device = qwen3_config.get("device", "auto")
            logger.info(f"加载 Qwen3 NER 模型: Transformers 后端, 路径: {model_path}, 设备: {device}")
            _ner_predictor = Qwen3Predictor(
                model_path=model_path,
                backend="transformers",
                device=device,
            )
        _ner_model_type = "qwen3"
    elif model_type == "globalpointer":
        # 使用 GlobalPointer 模型
        logger.info("使用 GlobalPointer NER 模型")
        
        try:
            from apps.trainer.globalpointer_ner.predict import GlobalPointerPredictor
        except ImportError as e:
            logger.error(f"无法导入 GlobalPointer 预测器: {e}")
            raise RuntimeError("GlobalPointer NER 模型依赖未安装")
        
        gp_config = ner_config.get("globalpointer", {})
        model_path = str(PROJECT_ROOT / gp_config.get("model_path", "outputs/globalpointer_ner/best_model"))
        threshold = gp_config.get("threshold", 0.0)
        device = gp_config.get("device", "auto")
        
        logger.info(f"加载 GlobalPointer NER 模型: {model_path}, threshold={threshold}")
        _ner_predictor = GlobalPointerPredictor(
            model_path=model_path,
            threshold=threshold,
            device=device
        )
        _ner_model_type = "globalpointer"
    else:
        # 使用 BERT NER 模型（默认）
        logger.info("使用 BERT NER 模型")
        bert_config = ner_config.get("bert", {})
        model_path = PROJECT_ROOT / bert_config.get("model_path", "models/pipe_model")
        if not model_path.exists():
            raise RuntimeError(f"NER模型不存在: {model_path}")
        
        # O标签偏置：正值使模型更倾向于预测O（对未知token更保守）
        o_bias = ner_config.get("o_bias", 0.0)
        # 设备配置
        device = bert_config.get("device", "auto")
        
        logger.info(f"加载 BERT NER 模型: {model_path}, O标签偏置: {o_bias}, 设备: {device}")
        _ner_predictor = PipePredictor(str(model_path), device=device, o_bias=o_bias)
        _ner_model_type = "bert"
    
    return _ner_predictor


def get_ner_confidence_threshold() -> float:
    """获取NER置信度阈值"""
    ner_config = get_ner_config()
    return ner_config.get("confidence_threshold", 0.9)


def get_batch_max_concurrent(default_value: int = 3) -> int:
    """获取批处理并发上限（最小为1）"""
    platform_config = get_platform_config()
    batch_cfg = platform_config.get("batch_processing", {}) or {}
    try:
        n = int(batch_cfg.get("max_concurrent", default_value))
    except Exception:
        n = default_value
    return max(1, n)


def get_tokenizer(model: str, platform: str = "pipe") -> LLMTokenizer:
    """获取分词器实例"""
    cache_key = f"{model}_{platform}"
    if cache_key not in _tokenizers:
        logger.info(f"创建分词器: model={model}, platform={platform}")
        _tokenizers[cache_key] = LLMTokenizer(model=model, platform=platform)
    return _tokenizers[cache_key]


# ============================================================
# 请求/响应模型
# ============================================================

class TokenizeRequest(BaseModel):
    """分词请求"""
    text: str = Field(..., description="待分词文本")
    preprocess: bool = Field(True, description="是否预处理")
    model: str = Field("deepseek-chat", description="使用的模型")
    platform: str = Field("pipe", description="平台类型")


class BatchTokenizeRequest(BaseModel):
    """批量分词请求"""
    texts: List[str] = Field(..., description="待分词文本列表")
    preprocess: bool = Field(True, description="是否预处理")
    model: str = Field("deepseek-chat", description="使用的模型")
    platform: str = Field("pipe", description="平台类型")


class TokenInfo(BaseModel):
    """分词信息"""
    word: str
    tag: str


class PipeEncodeRequest(BaseModel):
    """管道材料编码请求"""
    entities: Dict[str, Any] = Field(..., description="NER识别结果")
    text: str = Field("", description="原始描述")
    extract_confidence: Optional[Dict[str, Any]] = Field(None, description="第一阶段抽取置信度（按字段）")


class PipeEncodeFromTokensRequest(BaseModel):
    """管道材料编码请求（基于分词结果）"""
    tokens: List[TokenInfo] = Field(..., description="分词结果")
    text: str = Field("", description="原始描述")


class PipeBatchEncodeRequest(BaseModel):
    """批量管道材料编码请求"""
    items: List[Dict] = Field(..., description="编码项列表")


class H3yunImportItem(BaseModel):
    """氚云导入数据项"""
    description: str = Field(..., description="描述")
    code: str = Field(..., description="编码")
    type_raw: str = Field("", description="原始种类")
    type_code: str = Field("", description="标准化种类")
    size_raw: str = Field("", description="原始尺寸")
    size_code: str = Field("", description="标准化尺寸")
    thickness_raw: str = Field("", description="原始壁厚")
    thickness_code: str = Field("", description="标准化壁厚")
    pressure_raw: str = Field("", description="原始磅级")
    pressure_code: str = Field("", description="标准化磅级")
    material_raw: str = Field("", description="原始材质")
    material_code: str = Field("", description="标准化材质")
    standard_raw: str = Field("", description="原始规范")
    standard_code: str = Field("", description="标准化规范")


class H3yunImportRequest(BaseModel):
    """氚云导入请求"""
    items: List[H3yunImportItem] = Field(..., description="导入数据列表")
    encode_date: str = Field(..., description="编码日期时间，格式：YYYY-MM-DD HH:MM")


def _serialize_pipe_entity_for_encode(entity: Dict[str, Any]) -> Any:
    """保留 STANDARD_* 修饰项的 bind_to_index / 位置信息，其他字段保持原样值。"""
    value = entity.get("value") or entity.get("text", "")
    payload: Dict[str, Any] = {"value": value}

    for key in ("bind_to_index", "start", "end", "subtype"):
        if entity.get(key) is not None:
            payload[key] = entity.get(key)

    if len(payload) == 1:
        return value
    return payload


def _append_pipe_entity(entities: Dict[str, Any], field: str, val: Any):
    """
    聚合 predictor 输出的实体。

    - 普通字段: 保持旧的 string / list 形式
    - 带 subtype 的结构化字段（如 SIZE/THICKNESS）: 还原为嵌套对象
    """
    if isinstance(val, dict) and val.get("subtype") is not None:
        subtype = str(val["subtype"])
        value = val.get("value")
        if value in (None, ""):
            return
        field_obj = entities.get(field)
        if not isinstance(field_obj, dict):
            field_obj = {}
            entities[field] = field_obj
        bucket = field_obj.get(subtype)
        if not isinstance(bucket, list):
            bucket = []
            field_obj[subtype] = bucket
        bucket.append(value)
        return

    if field in entities:
        prev = entities[field]
        entities[field] = [prev, val] if not isinstance(prev, list) else prev + [val]
    else:
        entities[field] = val


def _build_pipe_entities_for_encode(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    优先使用语义解析器输出中的 decisions 作为编码输入。
    若不存在结构化输出，则回退到旧的打平聚合逻辑。
    """
    model_output = result.get("model_output")
    if isinstance(model_output, dict):
        decisions = model_output.get("decisions")
        if isinstance(decisions, dict) and decisions:
            return copy.deepcopy(decisions)
        structured = {
            k: copy.deepcopy(v)
            for k, v in model_output.items()
            if not str(k).startswith("_") and k != "model_raw_response"
        }
        if structured:
            return structured

    entities: Dict[str, Any] = {}
    for e in result.get("entities", []):
        field = e["type"]
        val = _serialize_pipe_entity_for_encode(e)
        _append_pipe_entity(entities, field, val)
    return entities


# ============================================================
# 标注相关API
# ============================================================

@app.get("/")
async def root():
    """根路径"""
    return {"message": "材料智能处理平台", "version": "2.0.0"}


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "ok"}


@app.get("/api/models")
async def get_models():
    """获取可用模型列表"""
    tokenizer_config = get_tokenizer_config()
    default_model = tokenizer_config.get("default_model", "deepseek-chat")
    
    models = [
        {"value": "ner_pipe", "label": "NER模型（本地）", "available": True},
        {"value": "deepseek-chat", "label": "DeepSeek API", "available": True},
        {"value": "qwen3:8b", "label": "Qwen3 8B", "available": True},
        {"value": "qwen3:0.6b", "label": "Qwen3 0.6B", "available": True},
    ]
    return {
        "models": models,
        "default": default_model
    }


@app.get("/api/config")
async def get_config():
    """获取平台配置"""
    tokenizer_config = get_tokenizer_config()
    semantic_config = get_semantic_config()
    platform_config = get_platform_config()
    batch_config = platform_config.get("batch_processing", {})
    
    return {
        "tokenizer": {
            "default_model": tokenizer_config.get("default_model", "deepseek-chat"),
            "default_platform": tokenizer_config.get("default_platform", "pipe"),
        },
        "semantic": {
            "model_name": semantic_config.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2"),
            "similarity_threshold": semantic_config.get("similarity_threshold", 0.9),
        },
        "batch_processing": {
            "max_concurrent": int(batch_config.get("max_concurrent", 3)),
            "progress_interval_ms": int(batch_config.get("progress_interval_ms", 100)),
        },
    }


@app.post("/api/tokenize")
async def tokenize(request: TokenizeRequest):
    """分词接口"""
    text = request.text
    
    if not text or not text.strip():
        return {"success": False, "error": "文本为空"}
    
    # 预处理
    processed_text = text
    if request.preprocess:
        processed_text = _preprocessor.process(text)
    
    # 分词
    try:
        # 如果是 NER 模型，使用本地 BERT 预测器
        if request.model == "ner_pipe":
            predictor = get_ner_predictor()
            result = predictor.predict(processed_text)
            
            # 打印 STANDARD 实体识别结果（调试用）
            standard_tokens = [t for t in result["tokens"] if t["tag"] == "STANDARD"]
            if standard_tokens:
                logger.info(f"[NER识别] STANDARD实体: {[(t['word'], t.get('confidence', 'N/A')) for t in standard_tokens]}")
            
            # 获取置信度阈值
            confidence_threshold = get_ner_confidence_threshold()
            
            # 过滤低置信度的实体（将其标记为 O）
            tokens = []
            for t in result["tokens"]:
                conf = t.get("confidence", 1.0)
                tag = t["tag"]
                # 如果置信度低于阈值，将非 O 标签改为 O
                if tag != "O" and conf < confidence_threshold:
                    logger.debug(f"过滤低置信度实体: {t['word']} ({tag}) 置信度={conf:.2%}")
                    tag = "O"
                tokens.append({
                    "word": t["word"], 
                    "tag": tag, 
                    "confidence": conf,
                    "start": t.get("start"),
                    "end": t.get("end")
                })
            
            type_class = result.get("type_class")
        else:
            # 使用 LLM 分词器
            tokenizer = get_tokenizer(request.model, request.platform)
            result = await tokenizer.tokenize(processed_text)
            tokens = result.get("tokens", [])
            type_class = result.get("type_class")
        
        return {
            "success": True,
            "original_text": text,
            "processed_text": processed_text,
            "tokens": tokens,
            "type_class": type_class
        }
    except Exception as e:
        logger.error(f"分词失败: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


@app.post("/api/tokenize/batch")
async def batch_tokenize(request: BatchTokenizeRequest):
    """批量分词（异步并发）"""
    
    # 如果是 NER 模型，使用本地预测器（同步）
    if request.model == "ner_pipe":
        predictor = get_ner_predictor()
        confidence_threshold = get_ner_confidence_threshold()
        results = []
        for text in request.texts:
            if not text or not text.strip():
                results.append({"success": False, "error": "文本为空"})
                continue
            
            processed_text = text
            if request.preprocess:
                processed_text = _preprocessor.process(text)
            
            try:
                result = predictor.predict(processed_text)
                # 过滤低置信度的实体
                tokens = []
                for t in result["tokens"]:
                    conf = t.get("confidence", 1.0)
                    tag = t["tag"]
                    if tag != "O" and conf < confidence_threshold:
                        tag = "O"
                    tokens.append({"word": t["word"], "tag": tag, "confidence": conf})
                results.append({
                    "success": True,
                    "original_text": text,
                    "processed_text": processed_text,
                    "tokens": tokens,
                    "type_class": result.get("type_class")
                })
            except Exception as e:
                results.append({"success": False, "error": str(e)})
        
        return {
            "success": True,
            "total": len(results),
            "results": results
        }
    
    # LLM 模型，使用异步并发
    async def process_single(text: str):
        if not text or not text.strip():
            return {"success": False, "error": "文本为空"}
        
        processed_text = text
        if request.preprocess:
            processed_text = _preprocessor.process(text)
        
        try:
            tokenizer = get_tokenizer(request.model, request.platform)
            result = await tokenizer.tokenize(processed_text)
            return {
                "success": True,
                "original_text": text,
                "processed_text": processed_text,
                "tokens": result.get("tokens", []),
                "type_class": result.get("type_class")
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # 并发执行
    tasks = [process_single(text) for text in request.texts]
    results = await asyncio.gather(*tasks)
    
    return {
        "success": True,
        "total": len(results),
        "results": results
    }


# ============================================================
# 编码相关API
# ============================================================

class PipePredictRequest(BaseModel):
    """管道材料NER预测请求（直接返回实体，不走token中转）"""
    text: str = Field(..., description="待提取文本")
    preprocess: bool = Field(True, description="是否预处理")


class PipeBatchPredictRequest(BaseModel):
    """批量管道材料NER预测请求"""
    texts: List[str] = Field(..., description="待提取文本列表")
    preprocess: bool = Field(True, description="是否预处理")


@app.post("/api/pipe/predict")
def pipe_predict(request: PipePredictRequest):
    """管道材料NER预测（直接返回JSON实体，供编码平台使用）"""
    text = request.text
    if not text or not text.strip():
        return {"success": False, "error": "文本为空"}

    processed_text = text
    if request.preprocess:
        processed_text = _preprocessor.process(text)

    try:
        predictor = get_ner_predictor()
        result = predictor.predict(processed_text)

        entities = _build_pipe_entities_for_encode(result)
        extract_confidence = result.get("extract_confidence", {}) or {}

        return {
            "success": True,
            "original_text": text,
            "processed_text": processed_text,
            "entities": entities,
            "extract_confidence": extract_confidence,
            "type_class": result.get("type_class"),
            "model_output": result.get("model_output", {}),
            "model_raw_response": result.get("model_raw_response", ""),
        }
    except Exception as e:
        logger.error(f"NER预测失败: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/pipe/predict/batch")
async def pipe_batch_predict(request: PipeBatchPredictRequest):
    """批量管道材料NER预测"""
    predictor = get_ner_predictor()
    semaphore = asyncio.Semaphore(get_batch_max_concurrent())

    async def process_one(text: str):
        if not text or not text.strip():
            return {"success": False, "error": "文本为空"}

        processed_text = _preprocessor.process(text) if request.preprocess else text
        try:
            async with semaphore:
                result = await asyncio.to_thread(predictor.predict, processed_text)
            entities = _build_pipe_entities_for_encode(result)
            extract_confidence = result.get("extract_confidence", {}) or {}

            return {
                "success": True,
                "original_text": text,
                "processed_text": processed_text,
                "entities": entities,
                "extract_confidence": extract_confidence,
                "type_class": result.get("type_class"),
                "model_output": result.get("model_output", {}),
                "model_raw_response": result.get("model_raw_response", ""),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    results = await asyncio.gather(*(process_one(t) for t in request.texts))

    return {"success": True, "total": len(results), "results": results}


@app.post("/api/pipe/encode")
def pipe_encode(request: PipeEncodeRequest):
    """管道材料编码（基于实体）"""
    encoder = get_pipe_encoder()
    result = encoder.encode(request.entities, request.text, request.extract_confidence)
    return _convert_pipe_result(result)


@app.post("/api/pipe/encode/tokens")
def pipe_encode_from_tokens(request: PipeEncodeFromTokensRequest):
    """管道材料编码（基于分词结果）"""
    encoder = get_pipe_encoder()
    tokens = [{"word": t.word, "tag": t.tag} for t in request.tokens]
    result = encoder.encode_from_tokens(tokens, request.text)
    return _convert_pipe_result(result)


@app.post("/api/pipe/encode/batch")
async def pipe_batch_encode(request: PipeBatchEncodeRequest):
    """批量管道材料编码"""
    encoder = get_pipe_encoder()
    semaphore = asyncio.Semaphore(get_batch_max_concurrent())

    async def process_item(item: Dict[str, Any]):
        async with semaphore:
            if 'tokens' in item:
                return await asyncio.to_thread(
                    encoder.encode_from_tokens,
                    item['tokens'],
                    item.get('text', '')
                )
            return await asyncio.to_thread(
                encoder.encode,
                item.get('entities', {}),
                item.get('text', ''),
                item.get('extract_confidence')
            )

    results = await asyncio.gather(*(process_item(item) for item in request.items))
    
    total = len(results)
    success_count = sum(1 for r in results if r.success)
    review_count = sum(1 for r in results if r.need_review)
    
    return {
        "total": total,
        "success_count": success_count,
        "review_count": review_count,
        "threshold": encoder.get_threshold(),
        "results": [_convert_pipe_result(r) for r in results]
    }


@app.post("/api/pipe/encode/batch/stream")
async def pipe_batch_encode_stream(request: PipeBatchEncodeRequest):
    """批量管道材料编码（SSE流式返回）"""
    encoder = get_pipe_encoder()
    items = request.items
    total = len(items)
    threshold = encoder.get_threshold()
    
    async def generate_sse():
        success_count = 0
        review_count = 0
        
        # 开始事件
        start_event = {"type": "start", "total": total, "threshold": threshold}
        yield f"data: {json.dumps(start_event, ensure_ascii=False)}\n\n"
        
        for index, item in enumerate(items):
            try:
                if 'tokens' in item:
                    result = encoder.encode_from_tokens(item['tokens'], item.get('text', ''))
                else:
                    result = encoder.encode(
                        item.get('entities', {}),
                        item.get('text', ''),
                        item.get('extract_confidence')
                    )
                
                if result.success:
                    success_count += 1
                if result.need_review:
                    review_count += 1
                
                converted = _convert_pipe_result(result)
                
                progress_event = {
                    "type": "progress",
                    "index": index,
                    "total": total,
                    "current": index + 1,
                    "success_count": success_count,
                    "review_count": review_count,
                    "result": converted
                }
                yield f"data: {json.dumps(progress_event, ensure_ascii=False)}\n\n"
                
            except Exception as e:
                logger.error(f"处理第 {index + 1} 条数据失败: {e}")
                review_count += 1
                error_event = {
                    "type": "progress",
                    "index": index,
                    "total": total,
                    "current": index + 1,
                    "success_count": success_count,
                    "review_count": review_count,
                    "result": {
                        "original_text": item.get('text', ''),
                        "final_code": "",
                        "success": False,
                        "need_review": True,
                        "errors": [str(e)]
                    }
                }
                yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"
            
            await asyncio.sleep(0)
        
        # 完成事件
        end_event = {
            "type": "end",
            "total": total,
            "success_count": success_count,
            "review_count": review_count,
            "threshold": threshold
        }
        yield f"data: {json.dumps(end_event, ensure_ascii=False)}\n\n"
    
    return StreamingResponse(
        generate_sse(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/pipe/config")
async def get_pipe_config():
    """获取管道编码配置"""
    encoder = get_pipe_encoder()
    config_path = PROJECT_ROOT / "src" / "encoder" / "config" / "encoder_config.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return {
            "threshold": encoder.get_threshold(),
            "field_order": encoder.FIELD_ORDER,
            "config": config
        }
    except Exception as e:
        return {
            "threshold": encoder.get_threshold(),
            "field_order": encoder.FIELD_ORDER,
            "error": str(e)
        }


@app.post("/api/pipe/threshold")
async def set_pipe_threshold(threshold: float = Query(..., ge=0.0, le=1.0)):
    """设置相似度阈值"""
    encoder = get_pipe_encoder()
    encoder.set_threshold(threshold)
    return {"message": "阈值更新成功", "threshold": threshold}


@app.get("/api/pipe/mapping/{field_type}")
async def get_pipe_mapping(field_type: str):
    """获取管道材料指定字段的映射表"""
    valid_types = ['TYPE', 'MATERIAL', 'MANU', 'CONN', 'SEAL', 'ENDS']
    if field_type.upper() not in valid_types:
        raise HTTPException(status_code=400, detail=f"无效的字段类型，可选: {valid_types}")
    
    matcher = get_semantic_matcher()
    mapping_data = matcher.mapping.get(field_type.upper(), {})
    
    return {"field_type": field_type.upper(), "mapping": mapping_data}


@app.post("/api/pipe/mapping/reload")
async def reload_pipe_mapping():
    """重新加载管道编码映射表"""
    encoder = get_pipe_encoder()
    encoder.reload_mapping()
    return {"message": "映射表重新加载成功"}


# ============================================================
# 氚云导入API
# ============================================================

@app.post("/api/h3yun/import")
async def import_to_h3yun(request: H3yunImportRequest):
    """导入编码结果到氚云"""
    # 转换请求数据
    items = [
        {
            "description": item.description,
            "code": item.code,
            "type_raw": item.type_raw,
            "type_code": item.type_code,
            "size_raw": item.size_raw,
            "size_code": item.size_code,
            "thickness_raw": item.thickness_raw,
            "thickness_code": item.thickness_code,
            "pressure_raw": item.pressure_raw,
            "pressure_code": item.pressure_code,
            "material_raw": item.material_raw,
            "material_code": item.material_code,
            "standard_raw": item.standard_raw,
            "standard_code": item.standard_code,
        }
        for item in request.items
    ]
    
    # 调用氚云客户端
    client = get_h3yun_client()
    result = client.import_encodings(items, request.encode_date)
    
    return {
        "success": result.success,
        "message": result.message,
        "count": result.count,
        "ids": result.ids,
        "task_code": result.task_code
    }


class TaskListRequest(BaseModel):
    """任务列表请求"""
    appCode: str = Field(..., description="应用编码")
    controller: str = Field("ReviewTaskListApiController", description="控制器名称")
    pageIndex: int = Field(1, ge=1, description="页码")
    pageSize: int = Field(20, ge=1, le=100, description="每页数量")
    filterTaskCode: Optional[str] = Field(None, description="任务编号筛选")
    filterReviewer: Optional[str] = Field(None, description="审核人筛选")
    filterCreatedTimeStart: Optional[str] = Field(None, description="创建时间开始")
    filterCreatedTimeEnd: Optional[str] = Field(None, description="创建时间结束")
    filterFeedbackTimeStart: Optional[str] = Field(None, description="反馈时间开始")
    filterFeedbackTimeEnd: Optional[str] = Field(None, description="反馈时间结束")


class TaskDetailRequest(BaseModel):
    """任务详情请求"""
    appCode: str = Field(..., description="应用编码")
    controller: str = Field("MyApiController", description="控制器名称")
    taskCode: str = Field(..., description="任务编号")
    pageIndex: int = Field(1, ge=1, description="页码")
    pageSize: int = Field(50, ge=1, le=500, description="每页数量")
    sortField: str = Field("encodeDate", description="排序字段")
    sortOrder: str = Field("desc", description="排序方向")
    # 筛选条件
    filterDescription: Optional[str] = Field(None, description="描述筛选")
    filterCode: Optional[str] = Field(None, description="编码筛选")
    filterCorrectedCode: Optional[str] = Field(None, description="修正编码筛选")
    filterReasonCategory: Optional[str] = Field(None, description="原因分类筛选")
    filterIsResolved: Optional[str] = Field(None, description="是否解决筛选")
    filterVerifyDateStart: Optional[str] = Field(None, description="核对日期开始")
    filterVerifyDateEnd: Optional[str] = Field(None, description="核对日期结束")
    filterEncodeDateStart: Optional[str] = Field(None, description="编码日期开始")
    filterEncodeDateEnd: Optional[str] = Field(None, description="编码日期结束")


class TaskObjectDetailRequest(BaseModel):
    """按业务对象ID查询任务详情请求"""
    bizObjectId: str = Field(..., description="氚云业务对象ID")
    schemaCode: str = Field(H3YUN_REVIEW_TASK_SCHEMA_CODE, description="表单编码")


class ReviewCorrectionItemRequest(BaseModel):
    """审核修正项"""
    id: str = Field(..., description="子表行ObjectId")
    correctedCode: str = Field("", description="修正后编码")
    correctedType: str = Field("", description="修正种类")
    correctedSize: str = Field("", description="修正尺寸")
    correctedThickness: str = Field("", description="修正壁厚")
    correctedPressure: str = Field("", description="修正磅级")
    correctedMaterial: str = Field("", description="修正材质")
    correctedStandard: str = Field("", description="修正规范")


class TaskCorrectionWriteRequest(BaseModel):
    """审核修正写入请求"""
    bizObjectId: str = Field(..., description="主表业务对象ID")
    appCode: str = Field(H3YUN_REVIEW_TASK_APP_CODE, description="应用编码")
    controller: str = Field("ReviewTaskListApiController", description="控制器名称")
    items: List[ReviewCorrectionItemRequest] = Field(default_factory=list, description="需要写入的修正项")


class ReasonCategoryRequest(BaseModel):
    """原因分类请求"""
    appCode: str = Field(..., description="应用编码")
    controller: str = Field("MyApiController", description="控制器名称")


@app.post("/api/h3yun/tasks")
async def get_h3yun_task_list(request: TaskListRequest):
    """获取氚云任务列表（代理接口）"""
    try:
        created_start = request.filterCreatedTimeStart
        created_end = request.filterCreatedTimeEnd
        feedback_start = request.filterFeedbackTimeStart
        feedback_end = request.filterFeedbackTimeEnd

        if created_start and not created_end:
            created_end = created_start
        elif created_end and not created_start:
            created_start = created_end

        if feedback_start and not feedback_end:
            feedback_end = feedback_start
        elif feedback_end and not feedback_start:
            feedback_start = feedback_end

        filters = {}
        if request.filterTaskCode:
            filters["filterTaskCode"] = request.filterTaskCode
        if request.filterReviewer:
            filters["filterReviewer"] = request.filterReviewer
        if created_start:
            filters["filterCreatedTimeStart"] = created_start
        if created_end:
            filters["filterCreatedTimeEnd"] = created_end
        if feedback_start:
            filters["filterFeedbackTimeStart"] = feedback_start
        if feedback_end:
            filters["filterFeedbackTimeEnd"] = feedback_end

        client = get_h3yun_client()
        result = client.get_task_list(
            app_code=request.appCode,
            controller=request.controller,
            page_index=request.pageIndex,
            page_size=request.pageSize,
            filters=filters if filters else None
        )

        for item in result.get("data", []) or []:
            review_date = (item.get("reviewDate") or "").strip()
            item["status"] = "已核对" if review_date else "待核对"

        return result
    except Exception as e:
        logger.error(f"获取任务列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/h3yun/tasks/detail")
async def get_h3yun_task_detail(request: TaskDetailRequest):
    """获取氚云任务详情（代理接口，支持排序和筛选）"""
    try:
        # 构建筛选条件
        filters = {}
        if request.filterDescription:
            filters["filterDescription"] = request.filterDescription
        if request.filterCode:
            filters["filterCode"] = request.filterCode
        if request.filterCorrectedCode:
            filters["filterCorrectedCode"] = request.filterCorrectedCode
        if request.filterReasonCategory:
            filters["filterReasonCategory"] = request.filterReasonCategory
        if request.filterIsResolved:
            filters["filterIsResolved"] = request.filterIsResolved
        if request.filterVerifyDateStart:
            filters["filterVerifyDateStart"] = request.filterVerifyDateStart
        if request.filterVerifyDateEnd:
            filters["filterVerifyDateEnd"] = request.filterVerifyDateEnd
        if request.filterEncodeDateStart:
            filters["filterEncodeDateStart"] = request.filterEncodeDateStart
        if request.filterEncodeDateEnd:
            filters["filterEncodeDateEnd"] = request.filterEncodeDateEnd
        
        client = get_h3yun_client()
        result = client.get_task_detail(
            task_code=request.taskCode,
            app_code=request.appCode,
            controller=request.controller,
            page_index=request.pageIndex,
            page_size=request.pageSize,
            sort_field=request.sortField,
            sort_order=request.sortOrder,
            filters=filters if filters else None
        )
        return result
    except Exception as e:
        logger.error(f"获取任务详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/h3yun/tasks/object-detail")
async def get_h3yun_task_object_detail(request: TaskObjectDetailRequest):
    """按BizObjectId获取审核任务主表及子表详情"""
    try:
        client = get_h3yun_client()
        biz_object = client.load_biz_object(
            schema_code=request.schemaCode,
            biz_object_id=request.bizObjectId
        )

        sub_items = biz_object.get(H3YUN_REVIEW_TASK_SUBTABLE_CODE, []) or []
        items = []
        for row in sub_items:
            if not isinstance(row, dict):
                continue
            items.append({
                "id": row.get("ObjectId", "") or "",
                "name": row.get("Name", "") or "",
                "description": row.get("F0000003", "") or "",
                "code": row.get("F0000004", "") or "",
                "correctedCode": row.get("F0000008", "") or "",
                "typeRaw": row.get("F0000009", "") or "",
                "typeCode": row.get("F0000010", "") or "",
                "correctedType": row.get("F0000011", "") or "",
                "sizeRaw": row.get("F0000012", "") or "",
                "sizeCode": row.get("F0000013", "") or "",
                "correctedSize": row.get("F0000014", "") or "",
                "thicknessRaw": row.get("F0000015", "") or "",
                "thicknessCode": row.get("F0000016", "") or "",
                "correctedThickness": row.get("F0000017", "") or "",
                "pressureRaw": row.get("F0000018", "") or "",
                "pressureCode": row.get("F0000019", "") or "",
                "correctedPressure": row.get("F0000020", "") or "",
                "materialRaw": row.get("F0000021", "") or "",
                "materialCode": row.get("F0000022", "") or "",
                "correctedMaterial": row.get("F0000023", "") or "",
                "standardRaw": row.get("F0000024", "") or "",
                "standardCode": row.get("F0000025", "") or "",
                "correctedStandard": row.get("F0000026", "") or "",
            })

        review_date = (biz_object.get("F0000043", "") or "").strip()

        return {
            "Success": True,
            "data": {
                "id": biz_object.get("ObjectId", "") or "",
                "taskCode": biz_object.get("SeqNo", "") or "",
                "name": biz_object.get("Name", "") or "",
                "reviewer": biz_object.get("F0000042", "") or "",
                "reviewerId": biz_object.get("F0000048", "") or "",
                "reviewDate": review_date,
                "creator": biz_object.get("CreatedBy", "") or "",
                "createdTime": biz_object.get("CreatedTime", "") or "",
                "modifiedBy": biz_object.get("ModifiedBy", "") or "",
                "modifiedTime": biz_object.get("ModifiedTime", "") or "",
                "status": "已核对" if review_date else "待核对",
                "items": items,
            }
        }
    except Exception as e:
        logger.error(f"按对象ID获取任务详情失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/h3yun/tasks/write-corrections")
async def write_h3yun_task_corrections(request: TaskCorrectionWriteRequest):
    """批量写入审核修正结果到氚云"""
    try:
        client = get_h3yun_client()
        result = client.save_review_task_corrections(
            app_code=request.appCode,
            controller=request.controller,
            biz_object_id=request.bizObjectId,
            items=[item.dict() for item in request.items]
        )
        return {
            "Success": True,
            "data": result
        }
    except Exception as e:
        logger.error(f"写入审核修正结果失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/h3yun/reason-categories")
async def get_h3yun_reason_categories(request: ReasonCategoryRequest):
    """获取原因分类列表"""
    try:
        client = get_h3yun_client()
        result = client.get_reason_categories(
            app_code=request.appCode,
            controller=request.controller
        )
        return {"Success": True, "data": result}
    except Exception as e:
        logger.error(f"获取原因分类失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 辅助函数
# ============================================================

def _convert_pipe_result(result) -> dict:
    """转换管道材料编码结果为响应格式"""
    fields = {}
    for field_type, field_data in result.fields.items():
        fields[field_type] = {
            "field_type": field_data.field_type,
            "original_value": field_data.original_value,
            "matched_name": field_data.matched_name,
            "code": field_data.code,
            "similarity": round(field_data.similarity, 4),
            "is_exact_match": field_data.is_exact_match,
            "need_review": field_data.need_review,
            "candidates": field_data.candidates,
            "display": field_data.display or "",  # 分类显示信息
            "items": field_data.items or []  # 多值分行显示
        }
    
    return {
        "original_text": result.original_text,
        "final_code": result.final_code,
        "success": result.success,
        "need_review": result.need_review,
        "hard_rule_hit": getattr(result, "hard_rule_hit", False),
        "confidence": round(getattr(result, "confidence", 0.0), 4),
        "min_similarity": round(result.min_similarity, 4),
        "review_fields": result.review_fields,
        "missing_fields": result.missing_fields,
        "errors": result.errors,
        "warnings": result.warnings,
        "fields": fields
    }


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":
    import argparse
    import uvicorn
    
    parser = argparse.ArgumentParser(description="材料智能处理平台")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="开发模式")
    
    args = parser.parse_args()
    
    logger.info(f"启动材料智能处理平台: http://{args.host}:{args.port}")
    uvicorn.run(
        "server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )
