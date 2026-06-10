<template>
  <div class="export-panel">
    <div class="panel-header">
      <span class="panel-title">导出编码结果</span>
    </div>
    
    <div class="export-buttons">
      <button class="btn btn-secondary btn-sm" @click="exportCSV">
        导出 CSV
      </button>
      <button class="btn btn-primary btn-sm" @click="exportExcel">
        导出 Excel
      </button>
      <button class="btn btn-secondary btn-sm" @click="exportStage1Dataset">
        导出一阶段数据集
      </button>
      <button class="btn btn-success btn-sm" @click="showImportDialog" :disabled="importing">
        {{ importing ? '导入中...' : '导入氚云' }}
      </button>
    </div>
    
    <!-- 氚云导入弹窗 -->
    <div v-if="showH3yunDialog" class="dialog-overlay" @click.self="showH3yunDialog = false">
      <div class="dialog-content">
        <div class="dialog-header">
          <span class="dialog-title">导入到氚云</span>
          <button class="dialog-close" @click="showH3yunDialog = false">&times;</button>
        </div>
        <div class="dialog-body">
          <div class="form-group">
            <label>编码日期时间</label>
            <input type="datetime-local" v-model="encodeDateTime" class="form-input" />
          </div>
          <div class="import-info">
            <p>将导入 <strong>{{ Object.keys(encodings).length }}</strong> 条编码记录到氚云</p>
          </div>
        </div>
        <div class="dialog-footer">
          <button class="btn btn-secondary btn-sm" @click="showH3yunDialog = false">取消</button>
          <button class="btn btn-success btn-sm" @click="importToH3yun" :disabled="!encodeDateTime || importing">
            {{ importing ? '导入中...' : '确认导入' }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { inject, ref } from 'vue'
import * as XLSX from 'xlsx'
import { saveAs } from 'file-saver'
import axios from 'axios'
import { buildDifficultyReason, getDifficultyLabel, getDisplayDifficultyLevel } from '../utils/difficulty'

const props = defineProps({
  encodings: { type: Object, default: () => ({}) },
  dataList: { type: Array, default: () => [] }
})

const showToast = inject('showToast')

// 氚云导入相关状态
const showH3yunDialog = ref(false)
// 默认当前日期时间（格式：YYYY-MM-DDTHH:mm 用于 datetime-local 输入）
const encodeDateTime = ref(getCurrentDateTime())
const importing = ref(false)

function getCurrentDateTime() {
  const now = new Date()
  const year = now.getFullYear()
  const month = String(now.getMonth() + 1).padStart(2, '0')
  const day = String(now.getDate()).padStart(2, '0')
  const hours = String(now.getHours()).padStart(2, '0')
  const minutes = String(now.getMinutes()).padStart(2, '0')
  return `${year}-${month}-${day}T${hours}:${minutes}`
}

function formatDateTimeForApi(datetimeLocal) {
  // 将 datetime-local 格式 (YYYY-MM-DDTHH:mm) 转换为 API 格式 (YYYY-MM-DD HH:MM)
  return datetimeLocal.replace('T', ' ')
}

function showImportDialog() {
  if (Object.keys(props.encodings).length === 0) {
    showToast('没有可导入的编码数据', 'warning')
    return
  }
  // 重置为当前时间
  encodeDateTime.value = getCurrentDateTime()
  showH3yunDialog.value = true
}

async function importToH3yun() {
  if (!encodeDateTime.value) {
    showToast('请选择编码日期时间', 'warning')
    return
  }
  
  importing.value = true
  
  try {
    // 构建导入数据
    const items = Object.entries(props.encodings).map(([index, enc]) => {
      const fields = enc.fields || {}
      return {
        description: enc.original_text || '',
        code: enc.final_code || '',
        type_raw: buildTypeRaw(fields),
        type_code: fields.TYPE?.code || '',
        size_raw: valueToText(fields.SIZE?.original_value),
        size_code: fields.SIZE?.code || '',
        thickness_raw: valueToText(fields.THICKNESS?.original_value),
        thickness_code: fields.THICKNESS?.code || '',
        pressure_raw: valueToText(fields.PRESSURE?.original_value),
        pressure_code: fields.PRESSURE?.code || '',
        material_raw: buildMaterialRaw(fields),
        material_code: fields.MATERIAL?.code || '',
        standard_raw: buildStandardRaw(fields),
        standard_code: fields.STANDARD?.code || ''
      }
    })
    
    const response = await axios.post('/api/h3yun/import', {
      items,
      encode_date: formatDateTimeForApi(encodeDateTime.value)
    })
    
    if (response.data.success) {
      const taskCode = response.data.task_code || ''
      showToast(`任务 ${taskCode} 创建成功，导入 ${response.data.count} 条编码`, 'success')
      showH3yunDialog.value = false
    } else {
      showToast(`导入失败: ${response.data.message}`, 'error')
    }
  } catch (error) {
    console.error('氚云导入错误:', error)
    showToast(`导入异常: ${error.message}`, 'error')
  } finally {
    importing.value = false
  }
}

function getFieldView(field, useOriginal = false) {
  if (!field) {
    return {
      original_value: '',
      original_values: [],
      code: '',
      codes: [],
      items: []
    }
  }
  const source = useOriginal && field.original_snapshot ? field.original_snapshot : field
  return {
    original_value: source.original_value || '',
    original_values: Array.isArray(source.original_values)
      ? source.original_values
      : (source.original_values != null ? [source.original_values] : []),
    code: source.code || '',
    codes: Array.isArray(source.codes)
      ? source.codes
      : (source.codes != null ? [source.codes] : []),
    items: Array.isArray(source.items) ? source.items : []
  }
}

function valueToText(value, joiner = ' ') {
  if (value == null) return ''
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)

  if (Array.isArray(value)) {
    return value
      .map(v => valueToText(v, joiner))
      .filter(Boolean)
      .join(joiner)
  }

  if (typeof value === 'object') {
    return Object.values(value)
      .map(v => valueToText(v, joiner))
      .filter(Boolean)
      .join(joiner)
  }

  return String(value)
}

function csvCell(value) {
  return `"${valueToText(value).replace(/"/g, '""')}"`
}

/**
 * 构建 TYPE_RAW：种类 + 附属属性（ENDS, SEAL, MANU, CONN 等）拼接
 * 从 original_values 数组中获取所有值，空格拼接
 * 种类、壁厚、尺寸、磅级不会出现多个，直接拼接
 */
function buildTypeRaw(fields, useOriginal = false) {
  const typeField = getFieldView(fields.TYPE, useOriginal)
  if (!typeField.original_value && typeField.original_values.length === 0) return ''
  
  // original_values 已经包含了 TYPE 及其附属属性（ENDS, SEAL, MANU, CONN）
  // 用空格拼接，不用 |
  if (typeField.original_values && typeField.original_values.length > 0) {
    return typeField.original_values
      .map(v => valueToText(v))
      .filter(Boolean)
      .join(' ')
  }
  // original_value 可能带 |，替换为空格
  return valueToText(typeField.original_value).replace(/\s*\|\s*/g, ' ')
}

/**
 * 构建 MATERIAL_RAW：多个材质用｜隔开，编码重复的去重
 */
function buildMaterialRaw(fields, useOriginal = false) {
  const materialField = getFieldView(fields.MATERIAL, useOriginal)
  if (!materialField.original_value && materialField.original_values.length === 0 && materialField.items.length === 0) return ''
  
  // 如果有 items，从 items 中获取并按编码去重
  if (materialField.items && materialField.items.length > 0) {
    const seenCodes = new Set()
    const uniqueOriginals = []
    
    for (const item of materialField.items) {
      const code = item.code || ''
      const original = item.original || ''
      if (code && !seenCodes.has(code)) {
        seenCodes.add(code)
        uniqueOriginals.push(original)
      }
    }
    return uniqueOriginals.join('｜')
  }
  
  // 否则使用 original_values 并按 codes 去重
  if (materialField.original_values && materialField.codes) {
    const seenCodes = new Set()
    const uniqueOriginals = []
    
    for (let i = 0; i < materialField.original_values.length; i++) {
      const code = materialField.codes[i] || ''
      const original = valueToText(materialField.original_values[i] || '')
      if (code && !seenCodes.has(code)) {
        seenCodes.add(code)
        uniqueOriginals.push(original)
      }
    }
    return uniqueOriginals.join('｜')
  }
  
  return valueToText(materialField.original_value)
}

/**
 * 构建 STANDARD_RAW：规范 + 规范等级，多个用｜隔开
 * 按 base_code 去重：如果两个规范的 base_code 相同，保留带等级的那个
 * 例如：NB/T47010 和 NB/T47010 type A 的 base_code 都是 NBT47010，保留后者
 */
function buildStandardRaw(fields, useOriginal = false) {
  const standardField = getFieldView(fields.STANDARD, useOriginal)
  if (!standardField.original_value && standardField.original_values.length === 0 && standardField.items.length === 0) return ''
  
  // 从 items 中获取规范和等级信息
  if (standardField.items && standardField.items.length > 0) {
    // 按 base_code 去重，保留带等级的版本
    const baseCodeMap = new Map() // base_code -> { original, hasGrade }
    
    for (const item of standardField.items) {
      const baseCode = item.base_code || item.code || ''
      const original = item.original || ''
      const grade = item.grade || ''
      const hasGrade = !!grade
      
      if (!baseCode) continue
      
      // 如果该 base_code 还没有记录，或者当前项有等级而之前的没有，则更新
      if (!baseCodeMap.has(baseCode) || (hasGrade && !baseCodeMap.get(baseCode).hasGrade)) {
        baseCodeMap.set(baseCode, { original, hasGrade })
      }
    }
    
    // 收集去重后的规范
    const uniqueStandards = Array.from(baseCodeMap.values()).map(v => v.original)
    return uniqueStandards.join('｜')
  }
  
  // 否则使用 original_values 并按 codes 去重
  if (standardField.original_values && standardField.codes) {
    const seenCodes = new Set()
    const uniqueOriginals = []
    
    for (let i = 0; i < standardField.original_values.length; i++) {
      const code = standardField.codes[i] || ''
      const original = valueToText(standardField.original_values[i] || '')
      if (code && !seenCodes.has(code)) {
        seenCodes.add(code)
        uniqueOriginals.push(original)
      }
    }
    return uniqueOriginals.join('｜')
  }
  
  return valueToText(standardField.original_value)
}

function formatPercent(value) {
  const num = Number(value ?? 0)
  if (Number.isNaN(num) || num <= 0) return ''
  return `${(num * 100).toFixed(2)}%`
}

function safeParseJson(value) {
  if (typeof value !== 'string') return value
  const text = value.trim()
  if (!text || (!text.startsWith('{') && !text.startsWith('['))) return value
  try {
    return JSON.parse(text)
  } catch {
    return value
  }
}

function buildStructuredRaw(type, value) {
  const parsed = safeParseJson(value)
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return valueToText(value)
  }

  if (type === 'PRESSURE') {
    return valueToText(value)
  }

  const subtypeOrderMap = {
    SIZE: ['DN', 'OD', 'INCH', 'LENGTH'],
    THICKNESS: ['MM', 'INCH', 'SCHEDULE', 'SERIES', 'BWG']
  }
  const order = subtypeOrderMap[type] || []

  return [
    ...order.filter(key => Object.prototype.hasOwnProperty.call(parsed, key)),
    ...Object.keys(parsed).filter(
      key =>
        !order.includes(key) &&
        !String(key).startsWith('_') &&
        typeof parsed[key] !== 'object'
    )
  ]
    .map(key => {
      const values = Array.isArray(parsed[key]) ? parsed[key] : [parsed[key]]
      const text = values
        .map(item => (item && typeof item === 'object') ? '' : String(item || '').trim())
        .filter(Boolean)
        .join(' x ')
      return text ? `${key}: ${text}` : ''
    })
    .filter(Boolean)
    .join(' ; ')
}

function exportCSV() {
  const headers = [
    '序号', '项目名称', '原始描述', '原始总编码', '修正总编码', '是否需审核', '总置信度', '最低相似度', '分流难度', '分流原因', '二次分流最终难度',
    'TYPE_原始结果', 'TYPE_原始编码', 'TYPE_修正结果', 'TYPE_修正编码',
    'SIZE_原始结果', 'SIZE_原始编码', 'SIZE_修正结果', 'SIZE_修正编码',
    'THICKNESS_原始结果', 'THICKNESS_原始编码', 'THICKNESS_修正结果', 'THICKNESS_修正编码',
    'PRESSURE_原始结果', 'PRESSURE_原始编码', 'PRESSURE_修正结果', 'PRESSURE_修正编码',
    'MATERIAL_原始结果', 'MATERIAL_原始编码', 'MATERIAL_修正结果', 'MATERIAL_修正编码',
    'STANDARD_原始结果', 'STANDARD_原始编码', 'STANDARD_修正结果', 'STANDARD_修正编码'
  ]
  let csv = headers.join(',') + '\n'
  
  Object.entries(props.encodings).forEach(([index, enc]) => {
    const fields = enc.fields || {}
    const projectName = props.dataList?.[parseInt(index)]?.projectName || ''
    const difficulty = enc.difficulty_split || {}
    const row = [
      parseInt(index) + 1,
      csvCell(projectName),
      csvCell(enc.original_text || ''),
      enc.original_final_code || enc.final_code || '',
      enc.final_code || '',
      enc.need_review ? '是' : '否',
      formatPercent(enc.confidence),
      formatPercent(enc.min_similarity),
      csvCell(getDifficultyLabel(difficulty.difficulty)),
      csvCell(buildDifficultyReason(enc)),
      csvCell(getDisplayDifficultyLevel(enc)),
      csvCell(buildTypeRaw(fields, true)),
      getFieldView(fields.TYPE, true).code || '',
      csvCell(buildTypeRaw(fields)),
      fields.TYPE?.code || '',
      csvCell(buildStructuredRaw('SIZE', getFieldView(fields.SIZE, true).original_value || '')),
      getFieldView(fields.SIZE, true).code || '',
      csvCell(buildStructuredRaw('SIZE', fields.SIZE?.original_value || '')),
      fields.SIZE?.code || '',
      csvCell(buildStructuredRaw('THICKNESS', getFieldView(fields.THICKNESS, true).original_value || '')),
      getFieldView(fields.THICKNESS, true).code || '',
      csvCell(buildStructuredRaw('THICKNESS', fields.THICKNESS?.original_value || '')),
      fields.THICKNESS?.code || '',
      csvCell(buildStructuredRaw('PRESSURE', getFieldView(fields.PRESSURE, true).original_value || '')),
      getFieldView(fields.PRESSURE, true).code || '',
      csvCell(buildStructuredRaw('PRESSURE', fields.PRESSURE?.original_value || '')),
      fields.PRESSURE?.code || '',
      csvCell(buildMaterialRaw(fields, true)),
      getFieldView(fields.MATERIAL, true).code || '',
      csvCell(buildMaterialRaw(fields)),
      fields.MATERIAL?.code || '',
      csvCell(buildStandardRaw(fields, true)),
      getFieldView(fields.STANDARD, true).code || '',
      csvCell(buildStandardRaw(fields)),
      fields.STANDARD?.code || ''
    ]
    csv += row.join(',') + '\n'
  })
  
  downloadFile('\ufeff' + csv, `encodings_${getDateStr()}.csv`, 'text/csv;charset=utf-8')
  showToast('CSV文件导出成功', 'success')
}

function exportExcel() {
  const headers = [
    '序号', '项目名称', '原始描述', '原始总编码', '修正总编码', '是否需审核', '总置信度', '最低相似度', '分流难度', '分流原因', '二次分流最终难度',
    'TYPE_原始结果', 'TYPE_原始编码', 'TYPE_修正结果', 'TYPE_修正编码',
    'SIZE_原始结果', 'SIZE_原始编码', 'SIZE_修正结果', 'SIZE_修正编码',
    'THICKNESS_原始结果', 'THICKNESS_原始编码', 'THICKNESS_修正结果', 'THICKNESS_修正编码',
    'PRESSURE_原始结果', 'PRESSURE_原始编码', 'PRESSURE_修正结果', 'PRESSURE_修正编码',
    'MATERIAL_原始结果', 'MATERIAL_原始编码', 'MATERIAL_修正结果', 'MATERIAL_修正编码',
    'STANDARD_原始结果', 'STANDARD_原始编码', 'STANDARD_修正结果', 'STANDARD_修正编码'
  ]
  
  const data = [headers]
  
  Object.entries(props.encodings).forEach(([index, enc]) => {
    const fields = enc.fields || {}
    const projectName = props.dataList?.[parseInt(index)]?.projectName || ''
    const difficulty = enc.difficulty_split || {}
    data.push([
      parseInt(index) + 1,
      projectName,
      enc.original_text || '',
      enc.original_final_code || enc.final_code || '',
      enc.final_code || '',
      enc.need_review ? '是' : '否',
      formatPercent(enc.confidence),
      formatPercent(enc.min_similarity),
      getDifficultyLabel(difficulty.difficulty),
      buildDifficultyReason(enc),
      getDisplayDifficultyLevel(enc),
      buildTypeRaw(fields, true),
      getFieldView(fields.TYPE, true).code || '',
      buildTypeRaw(fields),
      fields.TYPE?.code || '',
      buildStructuredRaw('SIZE', getFieldView(fields.SIZE, true).original_value || ''),
      getFieldView(fields.SIZE, true).code || '',
      buildStructuredRaw('SIZE', fields.SIZE?.original_value || ''),
      fields.SIZE?.code || '',
      buildStructuredRaw('THICKNESS', getFieldView(fields.THICKNESS, true).original_value || ''),
      getFieldView(fields.THICKNESS, true).code || '',
      buildStructuredRaw('THICKNESS', fields.THICKNESS?.original_value || ''),
      fields.THICKNESS?.code || '',
      buildStructuredRaw('PRESSURE', getFieldView(fields.PRESSURE, true).original_value || ''),
      getFieldView(fields.PRESSURE, true).code || '',
      buildStructuredRaw('PRESSURE', fields.PRESSURE?.original_value || ''),
      fields.PRESSURE?.code || '',
      buildMaterialRaw(fields, true),
      getFieldView(fields.MATERIAL, true).code || '',
      buildMaterialRaw(fields),
      fields.MATERIAL?.code || '',
      buildStandardRaw(fields, true),
      getFieldView(fields.STANDARD, true).code || '',
      buildStandardRaw(fields),
      fields.STANDARD?.code || ''
    ])
  })
  
  const ws = XLSX.utils.aoa_to_sheet(data)
  const wb = XLSX.utils.book_new()
  XLSX.utils.book_append_sheet(wb, ws, '编码结果')
  
  const excelBuffer = XLSX.write(wb, { bookType: 'xlsx', type: 'array' })
  const blob = new Blob([excelBuffer], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' })
  saveAs(blob, `encodings_${getDateStr()}.xlsx`)
  
  showToast('Excel文件导出成功', 'success')
}

function sanitizeStage1Output(value) {
  if (Array.isArray(value)) {
    return value.map(item => sanitizeStage1Output(item))
  }
  if (value && typeof value === 'object') {
    const out = {}
    Object.entries(value).forEach(([k, v]) => {
      if (k.startsWith('_')) return
      out[k] = sanitizeStage1Output(v)
    })
    return out
  }
  return value
}

function buildStage1Skeleton() {
  return {
    TYPE: {
      BODY: '',
      GEOMETRY: { ANGLE: '', RADIUS: '' },
      MANU: [],
      CONN: [],
      SEAL: [],
      ENDS: []
    },
    SIZE: {
      DN: [],
      OD: [],
      INCH: [],
      LENGTH: []
    },
    THICKNESS: {
      MM: [],
      SCHEDULE: [],
      SERIES: [],
      BWG: [],
      INCH: []
    },
    PRESSURE: '',
    MATERIAL: [],
    STANDARD: []
  }
}

function cloneValue(value) {
  return JSON.parse(JSON.stringify(value))
}

function normalizeToArray(value) {
  if (Array.isArray(value)) return value.filter(v => v != null && v !== '')
  if (value == null || value === '') return []
  return [value]
}

function normalizeMaterialEntries(material) {
  if (Array.isArray(material)) {
    return material.map(item => ({
      ROLE: item?.ROLE || 'MAIN',
      VALUE: item?.VALUE || '',
      SPECIAL_REQ: normalizeToArray(item?.SPECIAL_REQ)
    }))
  }

  if (material && typeof material === 'object') {
    // 新结构单项兜底
    if ('VALUE' in material || 'ROLE' in material) {
      return [{
        ROLE: material.ROLE || 'MAIN',
        VALUE: material.VALUE || '',
        SPECIAL_REQ: normalizeToArray(material.SPECIAL_REQ)
      }]
    }

    // 旧结构兼容: { RELATION, ITEMS }
    return normalizeToArray(material.ITEMS).map(item => ({
      ROLE: 'MAIN',
      VALUE: [item?.EXEC_STANDARD || '', item?.GRADE || ''].filter(Boolean).join(' ').trim(),
      SPECIAL_REQ: normalizeToArray(item?.SPECIAL_REQ)
    }))
  }

  if (material == null || material === '') return []
  return [{
    ROLE: 'MAIN',
    VALUE: String(material),
    SPECIAL_REQ: []
  }]
}

function normalizeStandardItems(items) {
  return normalizeToArray(items).map(item => ({
    BODY: item?.BODY || '',
    GRADE: item?.GRADE || '',
    METHOD: item?.METHOD || '',
    APPENDIX: item?.APPENDIX || ''
  }))
}

function looksLikeAngle(part) {
  const s = String(part || '').trim()
  return /^(\d+(?:\.\d+)?)°?$/.test(s)
}

function looksLikeRadius(part) {
  const s = String(part || '').trim().toUpperCase()
  return (
    /^R\s*=/.test(s) ||
    /(?:^|[\s])\d+(?:\.\d+)?D(?:N)?$/.test(s) ||
    /^(?:LR|SR|LR90|LR45|IR)$/i.test(s)
  )
}

function restoreTypeGeometry(typeObj) {
  const source = typeObj && typeof typeObj === 'object' ? typeObj : {}
  const mergedBody = String(source.BODY || '').trim()
  const geometry = source.GEOMETRY && typeof source.GEOMETRY === 'object'
    ? { ANGLE: source.GEOMETRY.ANGLE || '', RADIUS: source.GEOMETRY.RADIUS || '' }
    : { ANGLE: '', RADIUS: '' }

  let body = mergedBody
  if ((!geometry.ANGLE || !geometry.RADIUS) && mergedBody.includes(';')) {
    const parts = mergedBody.split(';').map(x => String(x).trim()).filter(Boolean)
    let angle = geometry.ANGLE
    let radius = geometry.RADIUS
    const bodyParts = []

    parts.forEach((part, idx) => {
      if (!angle && idx === 0 && looksLikeAngle(part)) {
        angle = part
        return
      }
      if (!radius && idx === parts.length - 1 && looksLikeRadius(part)) {
        radius = part
        return
      }
      bodyParts.push(part)
    })

    geometry.ANGLE = angle || ''
    geometry.RADIUS = radius || ''
    body = bodyParts.join(';')
  }

  return {
    BODY: body,
    GEOMETRY: geometry,
    MANU: normalizeToArray(source.MANU),
    CONN: normalizeToArray(source.CONN),
    SEAL: normalizeToArray(source.SEAL),
    ENDS: normalizeToArray(source.ENDS)
  }
}

function normalizeStage1Output(output) {
  const skeleton = buildStage1Skeleton()
  const source = sanitizeStage1Output(output || {})
  const normalized = cloneValue(skeleton)

  normalized.TYPE = restoreTypeGeometry(source.TYPE)

  const size = source.SIZE && typeof source.SIZE === 'object' ? source.SIZE : {}
  normalized.SIZE = {
    DN: normalizeToArray(size.DN),
    OD: normalizeToArray(size.OD),
    INCH: normalizeToArray(size.INCH),
    LENGTH: normalizeToArray(size.LENGTH)
  }

  const thickness = source.THICKNESS && typeof source.THICKNESS === 'object' ? source.THICKNESS : {}
  normalized.THICKNESS = {
    MM: normalizeToArray(thickness.MM),
    SCHEDULE: normalizeToArray(thickness.SCHEDULE),
    SERIES: normalizeToArray(thickness.SERIES),
    BWG: normalizeToArray(thickness.BWG),
    INCH: normalizeToArray(thickness.INCH)
  }

  normalized.PRESSURE = source.PRESSURE || ''

  normalized.MATERIAL = normalizeMaterialEntries(source.MATERIAL)

  normalized.STANDARD = normalizeStandardItems(source.STANDARD)

  return normalized
}

function buildStage1Output(enc) {
  const container = enc?.stage1_output || {}
  const source = (container && typeof container === 'object' && container.decisions && typeof container.decisions === 'object')
    ? container.decisions
    : container
  return normalizeStage1Output(source)
}

function exportStage1Dataset() {
  if (!props.dataList.length) {
    showToast('没有可导出的数据', 'warning')
    return
  }

  const rows = []
  props.dataList.forEach((item, idx) => {
    const enc = props.encodings?.[idx]
    if (!enc) return
    const output = buildStage1Output(enc)
    if (!output || Object.keys(output).length === 0) return
    rows.push({
      input: item.text || '',
      output
    })
  })

  if (!rows.length) {
    showToast('暂无一阶段结果可导出，请先执行识别/编码', 'warning')
    return
  }

  const json = JSON.stringify(rows, null, 2)
  downloadFile(json, `stage1_dataset_${getDateStr()}.json`, 'application/json;charset=utf-8')
  showToast(`一阶段数据集导出成功（${rows.length}条）`, 'success')
}

function downloadFile(content, filename, type) {
  const blob = new Blob([content], { type })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  a.click()
  URL.revokeObjectURL(url)
}

function getDateStr() {
  return new Date().toISOString().slice(0, 10)
}
</script>

<style scoped>
.export-panel {
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 16px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.panel-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}

.export-buttons {
  display: flex;
  gap: 8px;
}

.btn-success {
  background: #10b981;
  color: white;
  border: none;
}

.btn-success:hover:not(:disabled) {
  background: #059669;
}

.btn-success:disabled {
  background: #6ee7b7;
  cursor: not-allowed;
}

/* 弹窗样式 */
.dialog-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.dialog-content {
  background: var(--bg-primary);
  border-radius: 12px;
  width: 400px;
  max-width: 90vw;
  box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
}

.dialog-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid var(--border-color);
}

.dialog-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
}

.dialog-close {
  background: none;
  border: none;
  font-size: 24px;
  color: var(--text-secondary);
  cursor: pointer;
  line-height: 1;
}

.dialog-close:hover {
  color: var(--text-primary);
}

.dialog-body {
  padding: 20px;
}

.form-group {
  margin-bottom: 16px;
}

.form-group label {
  display: block;
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
  margin-bottom: 8px;
}

.form-input {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  font-size: 14px;
  background: var(--bg-secondary);
  color: var(--text-primary);
}

.form-input:focus {
  outline: none;
  border-color: var(--primary-color);
}

.import-info {
  padding: 12px;
  background: var(--bg-tertiary);
  border-radius: 6px;
  font-size: 14px;
  color: var(--text-secondary);
}

.import-info strong {
  color: var(--primary-color);
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  padding: 16px 20px;
  border-top: 1px solid var(--border-color);
}
</style>
