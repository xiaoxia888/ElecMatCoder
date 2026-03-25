<template>
  <div class="encoding-result">
    <!-- 编码结果头部 -->
    <div class="result-header">
      <div class="code-display">
        <span class="code-label">编码结果</span>
        <div class="code-box" :class="{ success: result.success && !result.need_review, warning: result.need_review }">
          {{ result.final_code || '—' }}
        </div>
      </div>
      <div class="status-info">
        <span class="confidence-text">总置信度 {{ totalConfidenceText }}</span>
        <span v-if="hasCorrection" class="correction-text">已修正</span>
        <span class="status-tag" :class="statusClass">{{ statusText }}</span>
      </div>
    </div>

    <!-- 字段分解 -->
    <div class="fields-section">
      <div class="fields-header">
        <span class="fields-title">字段分解</span>
        <span class="fields-hint">点击候选项可替换，双击字段可修正</span>
      </div>
      
      <div class="fields-list">
        <div 
          v-for="(field, type) in orderedFields" 
          :key="type"
          class="field-group"
          :class="{ 'need-review': field.need_review }"
        >
          <!-- 有 items 时分行显示 -->
          <template v-if="field.items && field.items.length > 1">
            <div 
              v-for="(item, idx) in field.items" 
              :key="idx"
              class="field-item field-item-sub"
              :class="{ 'need-review': item.need_review }"
              @dblclick.stop="emitEditField(type, idx)"
            >
              <div class="field-main">
                <!-- 第一行显示标签，后续行显示缩进符号 -->
                <span v-if="idx === 0" class="type-tag" :class="getTypeClass(type)">
                  {{ typeLabels[type] || type }}
                </span>
                <span v-else class="type-tag type-indent">↳</span>
                <span class="field-original" :title="getItemDisplayText(type, item)">{{ getItemDisplayText(type, item) || '—' }}</span>
                <template v-if="showItemMatchedName(type, item)">
                  <span class="field-arrow">→</span>
                  <span class="field-matched" :title="item.matched">{{ item.matched }}</span>
                </template>
                <span v-if="item.category" class="field-category-tag" :class="item.category === '生产' ? 'prod' : 'manu'">
                  {{ item.category }}
                </span>
                <span v-if="item.manual_override" class="field-manual-tag">修正</span>
                <!-- 每行显示各自的编码 -->
                <span class="field-code" :title="item.code">{{ item.code || '—' }}</span>
              </div>
              <!-- 每个 item 的候选项 -->
              <div v-if="item.candidates && item.candidates.length > 1" class="field-candidates">
                <span 
                  v-for="(c, ci) in item.candidates.slice(0, 3)" 
                  :key="ci" 
                  class="candidate-chip"
                  @click="$emit('select-candidate', { type, index: idx, candidate: c })"
                >
                  {{ c.name }}
                </span>
              </div>
            </div>
          </template>
          
          <!-- 单值或无 items 时正常显示 -->
          <template v-else>
            <div class="field-item" @dblclick.stop="emitEditField(type)">
              <div class="field-main">
                <span class="type-tag" :class="getTypeClass(type)">{{ typeLabels[type] || type }}</span>
                <span class="field-original" :title="getFieldPrimaryText(type, field)">{{ getFieldPrimaryText(type, field) || '—' }}</span>
                <!-- STANDARD 字段显示带分类的详情 -->
                <template v-if="type === 'STANDARD' && field.display">
                  <span class="field-arrow">→</span>
                  <span class="field-standard-display" v-html="formatStandardDisplay(field.display)"></span>
                </template>
                <!-- 语义匹配字段（TYPE/MATERIAL）显示匹配到的标准词 -->
                <template v-else-if="showMatchedName(type, field)">
                  <span class="field-arrow">→</span>
                  <span class="field-matched" :title="field.matched_name">{{ field.matched_name }}</span>
                </template>
                <span v-if="field.manual_override" class="field-manual-tag">修正</span>
                <!-- 最后一列：编码结果（靠右） -->
                <span class="field-code" :title="field.code">{{ field.code || '—' }}</span>
              </div>
              
              <!-- 候选项 -->
              <div v-if="field.candidates && field.candidates.length > 1" class="field-candidates">
                <span 
                  v-for="(c, i) in field.candidates.slice(0, 3)" 
                  :key="i" 
                  class="candidate-chip"
                  @click="$emit('select-candidate', { type, candidate: c })"
                >
                  {{ c.name }}
                </span>
              </div>
            </div>
          </template>
        </div>
      </div>
    </div>

    <!-- 警告信息 -->
    <div v-if="result.warnings && result.warnings.length > 0" class="warnings-section">
      <div v-for="(warn, i) in result.warnings" :key="i" class="warning-item">
        {{ warn }}
      </div>
    </div>

    <!-- Hybrid 调试信息 -->
    <div v-if="hasHybridDebug" class="hybrid-debug-section">
      <div class="hybrid-debug-title">Hybrid 调试信息</div>
      <div class="hybrid-debug-label">模型原始输出 / Hybrid输出 / 决策日志</div>
      <pre class="hybrid-debug-json" @wheel.stop>{{ formatJson(hybridDebugPayload) }}</pre>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  result: {
    type: Object,
    default: () => ({})
  }
})

const emit = defineEmits(['select-candidate', 'edit-field'])

const fieldOrder = ['TYPE', 'ENDS', 'SEAL', 'MANU', 'CONN', 'SIZE', 'THICKNESS', 'PRESSURE', 'MATERIAL', 'STANDARD']

const typeLabels = {
  TYPE: '种类',
  ENDS: '端部',
  SEAL: '密封',
  MANU: '工艺',
  CONN: '连接',
  SIZE: '尺寸',
  THICKNESS: '壁厚',
  PRESSURE: '磅级',
  MATERIAL: '材质',
  STANDARD: '规范'
}

const orderedFields = computed(() => {
  const fields = props.result.fields || {}
  const ordered = {}
  for (const type of fieldOrder) {
    if (fields[type]) {
      ordered[type] = fields[type]
    }
  }
  return ordered
})

const statusClass = computed(() => {
  if (props.result.need_review) return 'warning'
  if (props.result.success) return 'success'
  return 'error'
})

const statusText = computed(() => {
  if (props.result.need_review) return '需审核'
  if (props.result.success) return '成功'
  return '失败'
})

const totalConfidenceText = computed(() => {
  const conf = Number(props.result?.confidence ?? 0)
  if (Number.isNaN(conf) || conf <= 0) return '—'
  return `${(conf * 100).toFixed(2)}%`
})

const hasCorrection = computed(() => {
  const fields = props.result?.fields || {}
  return Object.values(fields).some(field => {
    if (!field) return
    if (field.manual_override) return true
    return (field.items || []).some(item => item?.manual_override)
  })
})

const hasHybridDebug = computed(() => {
  const d = props.result?.hybrid_debug
  return !!(d && (d.model_output_raw || d.model_output_hybrid || d.decision_log))
})
const hybridDebugPayload = computed(() => ({
  model_output_raw: props.result?.hybrid_debug?.model_output_raw || {},
  model_output_hybrid: props.result?.hybrid_debug?.model_output_hybrid || {},
  decision_log: props.result?.hybrid_debug?.decision_log || {}
}))

function getTypeClass(type) {
  const map = {
    TYPE: 'type-blue',
    ENDS: 'type-teal',
    SEAL: 'type-indigo',
    MANU: 'type-purple',
    CONN: 'type-cyan',
    SIZE: 'type-orange',
    THICKNESS: 'type-pink',
    PRESSURE: 'type-red',
    MATERIAL: 'type-green',
    STANDARD: 'type-gray'
  }
  return map[type] || ''
}

// 判断是否显示 matched_name（第二步）
// TYPE/MATERIAL 语义匹配字段：始终显示
// 其他字段：matched_name 与 original_value 和 code 都不同时才显示
function showMatchedName(type, field) {
  if (!field.matched_name) return false
  if (shouldHideOriginalValue(type)) {
    return false
  }
  // 语义匹配字段始终显示第二步
  if (type === 'TYPE' || type === 'MATERIAL') {
    return true
  }
  // 其他字段：matched_name 和 original/code 都不同才显示
  return field.matched_name !== field.original_value && field.matched_name !== field.code
}

function shouldHideOriginalValue(type) {
  return type === 'TYPE' || type === 'MATERIAL'
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

function formatJson(value) {
  try {
    return JSON.stringify(value || {}, null, 2)
  } catch {
    return '{}'
  }
}

function getFieldOriginalText(type, field) {
  if (type !== 'SIZE' && type !== 'THICKNESS') {
    return field.original_value || ''
  }
  const parsed = safeParseJson(field.original_value || '')
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return field.original_value || ''
  }
  const subtypeOrderMap = {
    SIZE: ['DN', 'OD', 'NPS', 'ID'],
    THICKNESS: ['MM', 'INCH', 'SCHEDULE', 'SERIES', 'BWG']
  }
  const order = subtypeOrderMap[type] || []
  return [
    ...order.filter(key => Object.prototype.hasOwnProperty.call(parsed, key)),
    ...Object.keys(parsed).filter(key => !order.includes(key))
  ]
    .map(key => {
      const values = Array.isArray(parsed[key]) ? parsed[key] : [parsed[key]]
      const text = values.map(item => String(item || '').trim()).filter(Boolean).join(' x ')
      return text ? `${key}: ${text}` : ''
    })
    .filter(Boolean)
    .join(' ; ')
}

function getFieldPrimaryText(type, field) {
  if (shouldHideOriginalValue(type)) {
    return field.matched_name || getFieldOriginalText(type, field) || ''
  }
  return getFieldOriginalText(type, field)
}

function showItemMatchedName(type, item) {
  if (!item?.matched) return false
  if (shouldHideOriginalValue(type)) {
    return false
  }
  return item.matched !== item.original && item.matched !== item.code
}

function getItemDisplayText(type, item) {
  if (shouldHideOriginalValue(type)) {
    return item?.matched || item?.original || ''
  }
  return item?.original || ''
}

// 格式化规范显示，用颜色标注分类
function formatStandardDisplay(display) {
  if (!display || display === '无') return ''
  // 输入: "GBT4237(生产) SHT3408(制造)"
  // 输出: "GBT4237(<span class="prod">生产</span>) SHT3408(<span class="manu">制造</span>)"
  return display
    .replace(/\(生产\)/g, '(<span class="category-prod">生产</span>)')
    .replace(/\(制造\)/g, '(<span class="category-manu">制造</span>)')
}

function emitEditField(type, index = null) {
  emit('edit-field', { type, index })
}

</script>

<style scoped>
.encoding-result {
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  overflow: hidden;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border-light);
}

.code-display {
  display: flex;
  align-items: center;
  gap: 10px;
  flex: 1;
  min-width: 0;
}

.code-label {
  font-size: 11px;
  color: var(--text-secondary);
  text-transform: uppercase;
  white-space: nowrap;
}

.code-box {
  font-size: 13px;
  font-weight: 600;
  font-family: 'SF Mono', Monaco, 'Courier New', monospace;
  padding: 4px 10px;
  border-radius: 4px;
  background: var(--bg-tertiary);
  color: var(--text-primary);
  word-break: break-all;
  max-width: 100%;
}

.code-box.success {
  background: var(--success-light);
  color: var(--success);
}

.code-box.warning {
  background: var(--warning-light);
  color: #8b5a00;
}

.status-info {
  margin-left: 12px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}

.confidence-text {
  font-size: 11px;
  color: var(--text-secondary);
  white-space: nowrap;
}

.correction-text {
  font-size: 11px;
  font-weight: 700;
  color: #fff;
  background: #dc2626;
  padding: 3px 8px;
  border-radius: 999px;
  white-space: nowrap;
}

.status-tag {
  font-size: 10px;
  font-weight: 600;
  padding: 3px 8px;
  border-radius: 3px;
  text-transform: uppercase;
}

.status-tag.success {
  background: var(--success-light);
  color: var(--success);
}

.status-tag.warning {
  background: var(--warning-light);
  color: #8b5a00;
}

.status-tag.error {
  background: var(--danger-light);
  color: var(--danger);
}

.fields-section {
  padding: 12px 16px;
}

.fields-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
}

.fields-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
}

.fields-hint {
  font-size: 10px;
  color: var(--text-muted);
}

.fields-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.field-group {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.field-item {
  background: var(--bg-secondary);
  border-radius: 6px;
  padding: 8px 10px;
  border-left: 3px solid transparent;
}

.field-item-sub {
  border-radius: 0;
  padding: 6px 10px;
}

.field-item-sub:first-child {
  border-radius: 6px 6px 0 0;
}

.field-item-sub:last-child {
  border-radius: 0 0 6px 6px;
}

.field-item-sub:only-child {
  border-radius: 6px;
}

.field-item.need-review {
  background: var(--warning-light);
  border-left-color: var(--warning);
}

.field-main {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 12px;
  flex-wrap: wrap;
}

.type-tag {
  font-size: 10px;
  font-weight: 600;
  padding: 2px 6px;
  border-radius: 3px;
  white-space: nowrap;
  flex-shrink: 0;
  min-width: 32px;
  text-align: center;
}

.type-blue { background: #e3f2fd; color: #1565c0; }
.type-teal { background: #e0f2f1; color: #00695c; }
.type-indigo { background: #e8eaf6; color: #3949ab; }
.type-purple { background: #f3e5f5; color: #7b1fa2; }
.type-cyan { background: #e0f7fa; color: #00838f; }
.type-orange { background: #fff3e0; color: #ef6c00; }
.type-pink { background: #fce4ec; color: #c2185b; }
.type-red { background: #ffebee; color: #c62828; }
.type-green { background: #e8f5e9; color: #2e7d32; }
.type-gray { background: #eceff1; color: #546e7a; }
.type-indent { background: transparent; color: var(--text-muted); font-weight: 400; }

.field-original {
  color: var(--text-primary);
  word-break: break-all;
}

.field-arrow {
  color: var(--text-muted);
  flex-shrink: 0;
}

.field-matched {
  color: var(--text-primary);
  font-weight: 500;
  word-break: break-all;
}

.field-category-tag {
  font-size: 9px;
  font-weight: 600;
  padding: 1px 4px;
  border-radius: 2px;
  flex-shrink: 0;
}

.field-category-tag.prod {
  background: #e8f5e9;
  color: #2e7d32;
}

.field-category-tag.manu {
  background: #e3f2fd;
  color: #1565c0;
}

.field-manual-tag {
  font-size: 10px;
  font-weight: 700;
  padding: 2px 6px;
  border-radius: 3px;
  background: #dc2626;
  color: #fff;
}

.field-standard-display {
  color: var(--text-primary);
  font-weight: 500;
  word-break: break-all;
}

.field-standard-display :deep(.category-prod) {
  color: #2e7d32;
  font-weight: 600;
}

.field-standard-display :deep(.category-manu) {
  color: #1565c0;
  font-weight: 600;
}

.field-code {
  font-family: 'SF Mono', Monaco, monospace;
  font-weight: 600;
  color: var(--primary);
  word-break: break-all;
  margin-left: auto;
  text-align: right;
}


.field-similarity {
  font-size: 11px;
  flex-shrink: 0;
  min-width: 36px;
  text-align: right;
}

.field-similarity.high { color: var(--success); }
.field-similarity.medium { color: var(--warning); }
.field-similarity.low { color: var(--danger); }

.field-candidates {
  display: flex;
  gap: 6px;
  margin-top: 6px;
  flex-wrap: wrap;
}

.candidate-chip {
  font-size: 10px;
  padding: 2px 6px;
  background: var(--bg-tertiary);
  border-radius: 3px;
  cursor: pointer;
  transition: all 0.15s;
}

.candidate-chip:hover {
  background: var(--primary-light);
  color: var(--primary);
}

.candidate-chip small {
  opacity: 0.6;
  margin-left: 3px;
}

.warnings-section {
  padding: 10px 16px;
  background: var(--warning-light);
  border-top: 1px solid var(--border-light);
}

.warning-item {
  font-size: 11px;
  color: #8b5a00;
  padding: 2px 0;
}

.hybrid-debug-section {
  margin-top: 10px;
  border-top: 1px dashed var(--border-light);
  padding: 10px 16px 14px;
  background: #fafbfc;
}

.hybrid-debug-title {
  font-size: 12px;
  font-weight: 700;
  color: #334155;
  margin-bottom: 4px;
}

.hybrid-debug-label {
  font-size: 11px;
  color: #64748b;
  margin-bottom: 6px;
}

.hybrid-debug-json {
  margin: 0;
  padding: 8px;
  border-radius: 6px;
  background: #f1f5f9;
  color: #0f172a;
  font-size: 11px;
  line-height: 1.35;
  white-space: pre-wrap;
  overflow-y: auto;
  overflow-x: auto;
  max-height: 520px;
  word-break: break-word;
  overflow-wrap: anywhere;
  box-sizing: border-box;
  overscroll-behavior: contain;
  -webkit-overflow-scrolling: touch;
}
</style>
