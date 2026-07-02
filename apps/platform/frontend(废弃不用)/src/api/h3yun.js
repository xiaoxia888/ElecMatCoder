/**
 * 氚云 API 服务
 * 通过后端代理调用氚云平台接口
 */
import axios from 'axios'

// 审核任务列表应用配置
const REVIEW_TASK_APP = {
  appCode: 'D148357CLDGGL',
  controller: 'ReviewTaskListApiController'
}

// 旧审核详情应用配置
const ENCODING_APP = {
  appCode: 'D148357CLDGGL',
  controller: 'MyApiController'
}

/**
 * 分页获取任务列表
 * @param {object} options - 查询参数
 * @returns {Promise<object>} { data, total, pageIndex, pageSize, totalPages }
 */
export async function getTaskList(options = {}) {
  const {
    pageIndex = 1,
    pageSize = 20,
    filters = {}
  } = options

  const response = await axios.post('/api/h3yun/tasks', {
    ...REVIEW_TASK_APP,
    pageIndex,
    pageSize,
    ...filters
  })
  return response.data
}

/**
 * 分页获取任务详情（支持排序和筛选）
 * @param {string} taskCode - 任务编号
 * @param {object} options - 选项
 * @returns {Promise<object>} { taskCode, data, total, pageIndex, pageSize, totalPages }
 */
export async function getTaskDetail(taskCode, options = {}) {
  const {
    pageIndex = 1,
    pageSize = 50,
    sortField = 'encodeDate',
    sortOrder = 'desc',
    filters = {}
  } = options
  
  const response = await axios.post('/api/h3yun/tasks/detail', {
    ...ENCODING_APP,
    taskCode,
    pageIndex,
    pageSize,
    sortField,
    sortOrder,
    ...filters
  })
  return response.data
}

/**
 * 按业务对象ID获取审核任务详情
 * @param {string} bizObjectId - 氚云业务对象ID
 * @param {string} schemaCode - 表单编码
 * @returns {Promise<object>}
 */
export async function getTaskObjectDetail(
  bizObjectId,
  schemaCode = 'D148357c862f0c8cdfa41418c55cef288f8d83c'
) {
  const response = await axios.post('/api/h3yun/tasks/object-detail', {
    bizObjectId,
    schemaCode
  })
  return response.data
}

/**
 * 批量写入审核修正结果
 * @param {string} bizObjectId - 主表业务对象ID
 * @param {Array<object>} items - 修正项列表
 * @returns {Promise<object>}
 */
export async function writeTaskCorrections(bizObjectId, items = []) {
  const response = await axios.post('/api/h3yun/tasks/write-corrections', {
    ...REVIEW_TASK_APP,
    bizObjectId,
    items
  })
  return response.data
}

/**
 * 获取原因分类列表
 * @returns {Promise<string[]>} 原因分类列表
 */
export async function getReasonCategories() {
  const response = await axios.post('/api/h3yun/reason-categories', ENCODING_APP)
  return response.data.data || []
}

export default {
  getTaskList,
  getTaskDetail,
  getTaskObjectDetail,
  writeTaskCorrections,
  getReasonCategories
}
