# -*- coding: utf-8 -*-
"""
氚云 API 客户端
用于与氚云平台进行数据交互
"""

import json
import logging
import random
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# 导入配置模块
from src.config import get_platform_config

logger = logging.getLogger(__name__)


@dataclass
class H3yunConfig:
    """氚云配置"""
    url: str = "https://www.h3yun.com/OpenApi/Invoke"
    engine_code: str = ""
    engine_secret: str = ""
    timeout: int = 300  # 5分钟超时
    batch_size: int = 500  # 每批最多500条
    
    @classmethod
    def from_yaml(cls) -> 'H3yunConfig':
        """从配置文件加载"""
        platform_config = get_platform_config()
        h3yun_config = platform_config.get("h3yun", {})
        return cls(
            url=h3yun_config.get("url", cls.url),
            engine_code=h3yun_config.get("engine_code", ""),
            engine_secret=h3yun_config.get("engine_secret", ""),
            timeout=h3yun_config.get("timeout", 300),
            batch_size=h3yun_config.get("batch_size", 500)
        )


@dataclass
class H3yunResult:
    """氚云操作结果"""
    success: bool
    message: str
    count: int = 0
    ids: List[str] = field(default_factory=list)
    task_code: str = ""  # 任务编号


class H3yunClient:
    """氚云 API 客户端"""
    
    # 编码表字段映射
    ENCODING_FIELD_MAPPING = {
        "description": "F0000001",      # 描述
        "code": "F0000002",             # 编码
        "type_raw": "F0000003",         # 原始种类
        "type_code": "F0000004",        # 标准化种类
        "size_raw": "F0000005",         # 原始尺寸
        "size_code": "F0000006",        # 标准化尺寸
        "thickness_raw": "F0000007",    # 原始壁厚
        "thickness_code": "F0000008",   # 标准化壁厚
        "pressure_raw": "F0000009",     # 原始磅级
        "pressure_code": "F0000010",    # 标准化磅级
        "material_raw": "F0000011",     # 原始材质
        "material_code": "F0000012",    # 标准化材质
        "standard_raw": "F0000013",     # 原始规范
        "standard_code": "F0000014",    # 标准化规范
        "encode_date": "F0000016",      # 编码日期
        "task_code": "F0000024",        # 任务编号
    }
    
    # 编码表 Schema Code
    ENCODING_SCHEMA_CODE = "D148357d180db43c49f48a6938d9c751867b20b"
    
    @staticmethod
    def generate_task_code() -> str:
        """
        生成任务编号
        格式：AIGD-yyyy-MMDD + 3位随机数
        例如：AIGD-2026-0202123
        """
        now = datetime.now()
        date_part = now.strftime("%Y-%m%d")
        random_part = str(random.randint(100, 999))
        return f"AIGD-{date_part}{random_part}"
    
    def __init__(self, config: Optional[H3yunConfig] = None):
        """
        初始化客户端
        
        Args:
            config: 氚云配置，为空时从配置文件加载
        """
        self.config = config or H3yunConfig.from_yaml()
        self._headers = {
            "Content-Type": "application/json",
            "EngineCode": self.config.engine_code,
            "EngineSecret": self.config.engine_secret
        }
    
    def _request(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        发送请求到氚云
        
        Args:
            params: 请求参数
            
        Returns:
            响应结果
        """
        try:
            response = requests.post(
                self.config.url,
                headers=self._headers,
                json=params,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "Successful": False,
                    "ErrorMessage": f"HTTP {response.status_code}"
                }
        except requests.Timeout:
            return {
                "Successful": False,
                "ErrorMessage": "请求超时"
            }
        except Exception as e:
            return {
                "Successful": False,
                "ErrorMessage": str(e)
            }
    
    def create_objects(
        self,
        schema_code: str,
        data_list: List[Dict[str, Any]],
        is_submit: bool = True
    ) -> H3yunResult:
        """
        批量创建业务对象
        
        Args:
            schema_code: 表单编码
            data_list: 数据列表
            is_submit: 是否提交（True=生效，False=草稿）
            
        Returns:
            操作结果
        """
        if not data_list:
            return H3yunResult(
                success=False,
                message="数据列表为空"
            )
        
        # 转换数据格式
        biz_object_array = [json.dumps(data) for data in data_list]
        
        # 构建请求参数
        params = {
            "ActionName": "CreateBizObjects",
            "SchemaCode": schema_code,
            "BizObjectArray": biz_object_array,
            "IsSubmit": is_submit
        }
        
        logger.info(f"正在写入数据到氚云，共 {len(data_list)} 条记录...")
        
        # 发送请求
        result = self._request(params)
        
        if result.get("Successful", False):
            successful_ids = result.get("ReturnData", [])
            logger.info(f"氚云导入成功，实际记录数: {len(data_list)}")
            return H3yunResult(
                success=True,
                message=f"成功导入 {len(data_list)} 条记录",
                count=len(data_list),
                ids=successful_ids
            )
        else:
            error_msg = result.get("ErrorMessage", "未知错误")
            logger.error(f"氚云导入失败: {error_msg}")
            return H3yunResult(
                success=False,
                message=f"导入失败: {error_msg}"
            )
    
    def import_encodings(
        self,
        items: List[Dict[str, str]],
        encode_date: str
    ) -> H3yunResult:
        """
        导入编码数据到氚云（支持分批导入）
        
        Args:
            items: 编码数据列表，每项包含：
                - description: 描述
                - code: 编码
                - type_raw: 原始种类
                - type_code: 标准化种类
                - size_raw: 原始尺寸
                - size_code: 标准化尺寸
                - thickness_raw: 原始壁厚
                - thickness_code: 标准化壁厚
                - pressure_raw: 原始磅级
                - pressure_code: 标准化磅级
                - material_raw: 原始材质
                - material_code: 标准化材质
                - standard_raw: 原始规范
                - standard_code: 标准化规范
            encode_date: 编码日期时间，格式：YYYY-MM-DD HH:MM
            
        Returns:
            操作结果
        """
        # 生成本批次的任务编号（同一批次共用）
        task_code = self.generate_task_code()
        total_count = len(items)
        logger.info(f"创建编码任务: {task_code}, 包含 {total_count} 条编码记录")
        
        # 转换字段名
        data_list = []
        for item in items:
            data = {}
            for key, field_code in self.ENCODING_FIELD_MAPPING.items():
                if key == "encode_date":
                    data[field_code] = encode_date
                elif key == "task_code":
                    data[field_code] = task_code
                else:
                    data[field_code] = item.get(key, "")
            data_list.append(data)
        
        # 分批导入
        batch_size = self.config.batch_size
        total_success = 0
        all_ids = []
        
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(data_list) + batch_size - 1) // batch_size
            
            logger.info(f"导入批次 {batch_num}/{total_batches}, 本批 {len(batch)} 条...")
            
            result = self.create_objects(self.ENCODING_SCHEMA_CODE, batch)
            
            if result.success:
                total_success += result.count
                all_ids.extend(result.ids)
            else:
                # 某批失败，返回错误
                return H3yunResult(
                    success=False,
                    message=f"批次 {batch_num} 导入失败: {result.message}",
                    count=total_success,
                    ids=all_ids,
                    task_code=task_code
                )
        
        return H3yunResult(
            success=True,
            message=f"任务 {task_code} 创建成功，导入 {total_success} 条编码",
            count=total_success,
            ids=all_ids,
            task_code=task_code
        )
    
    def call_custom_api(
        self, 
        action_name: str, 
        app_code: str,
        controller: str = "MyApiController",
        params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        调用自定义 API（氚云后端自定义接口）
        
        Args:
            action_name: 接口名称，如 loadPageTaskList, loadTaskDetail
            app_code: 应用编码
            controller: 控制器名称
            params: 接口参数
            
        Returns:
            API 返回数据
        """
        request_params = {
            "ActionName": action_name,
            "Controller": controller,
            "AppCode": app_code,
        }
        
        if params:
            request_params["para1"] = json.dumps(params, ensure_ascii=False)
        
        result = self._request(request_params)
        
        if not result.get("Successful", False):
            error_msg = result.get("ErrorMessage", "请求失败")
            raise Exception(error_msg)
        
        return_data = result.get("ReturnData", {})
        if not return_data.get("Success", False):
            error_msg = return_data.get("Message", "操作失败")
            debug_parts = []
            for key in [
                "DebugStep",
                "DebugPara1",
                "DebugCountSql",
                "DebugDataSql",
                "DebugInnerMessage",
                "DebugReviewerFilter",
                "DebugReviewerWhere",
                "DebugReviewerSamples",
            ]:
                value = return_data.get(key)
                if value not in (None, ""):
                    debug_parts.append(f"{key}={value}")
            if debug_parts:
                error_msg = f"{error_msg} | " + " | ".join(debug_parts)
            raise Exception(error_msg)
        
        return return_data
    
    def get_task_list(
        self, 
        app_code: str,
        controller: str = "ReviewTaskListApiController",
        page_index: int = 1, 
        page_size: int = 20,
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        分页获取任务列表
        
        Args:
            app_code: 应用编码
            controller: 控制器名称
            page_index: 页码（从1开始）
            page_size: 每页数量
            filters: 筛选条件
            
        Returns:
            {data: [...], total, pageIndex, pageSize, totalPages}
        """
        params = {"pageIndex": page_index, "pageSize": page_size}
        if filters:
            for key, value in filters.items():
                if value:
                    params[key] = value

        return self.call_custom_api(
            "loadReviewTaskPageList",
            app_code,
            controller,
            params
        )

    def load_biz_object(
        self,
        schema_code: str,
        biz_object_id: str
    ) -> Dict[str, Any]:
        """
        通过 OpenApi 直接加载单个业务对象详情

        Args:
            schema_code: 表单编码
            biz_object_id: 业务对象ID

        Returns:
            BizObject 原始数据
        """
        request_params = {
            "ActionName": "LoadBizObject",
            "SchemaCode": schema_code,
            "BizObjectId": biz_object_id,
        }

        result = self._request(request_params)

        if not result.get("Successful", False):
            error_msg = result.get("ErrorMessage", "请求失败")
            raise Exception(error_msg)

        return_data = result.get("ReturnData", {}) or {}
        biz_object = return_data.get("BizObject")
        if not isinstance(biz_object, dict):
            raise Exception("未获取到BizObject详情")

        return biz_object

    def save_review_task_corrections(
        self,
        app_code: str,
        biz_object_id: str,
        items: List[Dict[str, Any]],
        controller: str = "ReviewTaskListApiController"
    ) -> Dict[str, Any]:
        """
        调用氚云自定义接口批量写入审核修正结果

        Args:
            app_code: 应用编码
            biz_object_id: 主表业务对象ID
            items: 需要写入的子表修正项列表
            controller: 控制器名称

        Returns:
            自定义接口返回数据
        """
        request_params = {
            "ActionName": "batchUpdateReviewCorrections",
            "Controller": controller,
            "AppCode": app_code,
            "BizObjectId": biz_object_id,
            "ItemsJson": json.dumps(items or [], ensure_ascii=False),
        }

        result = self._request(request_params)

        if not result.get("Successful", False):
            error_msg = result.get("ErrorMessage", "请求失败")
            raise Exception(error_msg)

        return_data = result.get("ReturnData", {})
        if not return_data.get("Success", False):
            error_msg = return_data.get("Message", "操作失败")
            raise Exception(error_msg)

        return return_data
    
    def get_task_detail(
        self, 
        task_code: str, 
        app_code: str,
        controller: str = "MyApiController",
        page_index: int = 1, 
        page_size: int = 50,
        sort_field: str = "encodeDate",
        sort_order: str = "desc",
        filters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        分页获取任务详情（支持排序和筛选）
        
        Args:
            task_code: 任务编号
            app_code: 应用编码
            controller: 控制器名称
            page_index: 页码（从1开始）
            page_size: 每页数量
            sort_field: 排序字段
            sort_order: 排序方向 (asc/desc)
            filters: 筛选条件
            
        Returns:
            {taskCode, data: [...], total, pageIndex, pageSize, totalPages}
        """
        params = {
            "taskCode": task_code, 
            "pageIndex": page_index, 
            "pageSize": page_size,
            "sortField": sort_field,
            "sortOrder": sort_order
        }
        
        # 添加筛选条件
        if filters:
            for key, value in filters.items():
                if value:
                    params[key] = value
        
        return self.call_custom_api("loadTaskDetail", app_code, controller, params)
    
    def get_reason_categories(
        self,
        app_code: str,
        controller: str = "MyApiController"
    ) -> List[str]:
        """
        获取原因分类列表
        
        Args:
            app_code: 应用编码
            controller: 控制器名称
            
        Returns:
            原因分类列表
        """
        result = self.call_custom_api("loadReasonCategories", app_code, controller, {})
        return result.get("data", [])


# 单例实例
_client: Optional[H3yunClient] = None


def get_h3yun_client() -> H3yunClient:
    """获取氚云客户端单例"""
    global _client
    if _client is None:
        _client = H3yunClient()
    return _client
