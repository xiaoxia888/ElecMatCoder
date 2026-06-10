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
          <!-- STANDARD 永远按 items 显示；其他字段有多个 items 时分行显示 -->
          <template v-if="shouldRenderItems(type, field)">
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
                <span class="field-original" :title="getFieldDisplayText(type, field)">{{ getFieldDisplayText(type, field) || '—' }}</span>
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

    <div v-if="result.thickness_conversion_notes && result.thickness_conversion_notes.length > 0" class="conversion-notes-section">
      <div v-for="(note, i) in result.thickness_conversion_notes" :key="i" class="conversion-note-item">
        {{ note }}
      </div>
    </div>

    <div v-if="hasDifficultySplit" class="route-section">
      <div class="route-header">
        <div class="route-header-main">
          <span class="route-title">分流结果</span>
          <span class="route-hint">一阶段 + 二次校验</span>
        </div>
      </div>
      <div class="route-meta-list">
        <div class="route-meta-chip">
          <span class="route-label">难度</span>
          <span class="route-value route-value-strong" :class="displayDifficultyClass">{{ displayDifficultyText }}</span>
        </div>
      </div>
      <div v-if="displayDifficultyReason" class="route-detail-line">
        <span class="route-label">原因</span>
        <span class="route-value">{{ displayDifficultyReason }}</span>
      </div>
    </div>

    <div v-if="hasRouteInfo" class="route-section">
      <div class="route-header route-header-collapsible" @click="routeCollapsed = !routeCollapsed">
        <div class="route-header-main">
          <span class="route-title">路由信息</span>
          <span class="route-hint">一阶段模型分发</span>
        </div>
        <span class="route-collapse-icon">{{ routeCollapsed ? '展开' : '收起' }}</span>
      </div>
      <template v-if="!routeCollapsed">
      <div class="route-meta-list">
        <div class="route-meta-chip">
          <span class="route-label">目标类别</span>
          <span class="route-value route-value-strong">{{ routeInfo.selected_category || routeInfo.category || '—' }}</span>
        </div>
        <div class="route-meta-chip">
          <span class="route-label">路由置信度</span>
          <span class="route-value">{{ routeConfidenceText }}</span>
        </div>
        <div class="route-meta-chip">
          <span class="route-label">路由来源</span>
          <span class="route-value">{{ routeInfo.source || '—' }}</span>
        </div>
        <div class="route-meta-chip">
          <span class="route-label">命中模型</span>
          <span class="route-value route-value-strong">{{ routeInfo.selected_model_scope || '—' }}</span>
        </div>
      </div>
      <div v-if="routeInfo.reason" class="route-detail-line">
        <span class="route-label">判定理由</span>
        <span class="route-value">{{ routeInfo.reason }}</span>
      </div>
      <div v-if="routeCandidatesText" class="route-detail-line">
        <span class="route-label">候选类别</span>
        <span class="route-value">{{ routeCandidatesText }}</span>
      </div>
      <div v-if="routeBreakdownText" class="route-detail-line">
        <span class="route-label">置信度分解</span>
        <span class="route-value">{{ routeBreakdownText }}</span>
      </div>
      </template>
    </div>

    <div v-if="extractConfidenceV2Rows.length > 0" class="route-section">
      <div class="route-header route-header-collapsible" @click="stage1Collapsed = !stage1Collapsed">
        <div class="route-header-main">
          <span class="route-title">一阶段提取置信度V2</span>
          <span class="route-hint">字段级调试信息</span>
        </div>
        <span class="route-collapse-icon">{{ stage1Collapsed ? '展开' : '收起' }}</span>
      </div>
      <template v-if="!stage1Collapsed">
      <div
        v-for="row in extractConfidenceV2Rows"
        :key="row.field"
        class="route-detail-line route-detail-line-block"
      >
        <span class="route-label route-label-strong">{{ row.fieldLabel }}</span>
        <span class="route-value">{{ row.summary }}</span>
      </div>
      </template>
    </div>

    <div v-if="encodeConfidenceV2Rows.length > 0" class="route-section">
      <div class="route-header route-header-collapsible" @click="stage2Collapsed = !stage2Collapsed">
        <div class="route-header-main">
          <span class="route-title">二阶段编码置信度V2</span>
          <span class="route-hint">字段级调试信息</span>
        </div>
        <span class="route-collapse-icon">{{ stage2Collapsed ? '展开' : '收起' }}</span>
      </div>
      <template v-if="!stage2Collapsed">
      <div
        v-for="row in encodeConfidenceV2Rows"
        :key="row.field"
        class="route-detail-line route-detail-line-block"
      >
        <span class="route-label route-label-strong">{{ row.fieldLabel }}</span>
        <span class="route-value">{{ row.summary }}</span>
      </div>
      </template>
    </div>

    <div v-if="fieldConfidenceRows.length > 0" class="route-section">
      <div class="route-header route-header-collapsible" @click="finalConfidenceCollapsed = !finalConfidenceCollapsed">
        <div class="route-header-main">
          <span class="route-title">总置信度信息</span>
          <span class="route-hint">一阶段 / 二阶段 / 字段最终</span>
        </div>
        <span class="route-collapse-icon">{{ finalConfidenceCollapsed ? '展开' : '收起' }}</span>
      </div>
      <template v-if="!finalConfidenceCollapsed">
      <div
        v-for="row in fieldConfidenceRows"
        :key="row.field"
        class="route-detail-line route-detail-line-block"
      >
        <span class="route-label route-label-strong">{{ row.fieldLabel }}</span>
        <span class="route-value">{{ row.summary }}</span>
      </div>
      </template>
    </div>

    <!-- 警告信息 -->
    <div v-if="result.warnings && result.warnings.length > 0" class="warnings-section">
      <div v-for="(warn, i) in result.warnings" :key="i" class="warning-item">
        {{ warn }}
      </div>
    </div>

  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import {
  DIFF_HARD,
  DIFF_SECOND_EASY,
  buildDifficultyReason,
  getDifficultyLabel,
  hasDifficultyLevel,
  normalizeDifficultyLevel
} from '../utils/difficulty'

const props = defineProps({
  result: {
    type: Object,
    default: () => ({})
  }
})

const emit = defineEmits(['select-candidate', 'edit-field'])

const routeCollapsed = ref(true)
const stage1Collapsed = ref(true)
const stage2Collapsed = ref(true)
const finalConfidenceCollapsed = ref(true)

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

const difficultySplit = computed(() => props.result?.difficulty_split || null)
const secondPass = computed(() => props.result?.second_pass || null)

const difficultyLevel = computed(() => normalizeDifficultyLevel(difficultySplit.value?.difficulty))
const secondPassLevel = computed(() => normalizeDifficultyLevel(secondPass.value?.final_level))

const hasDifficultySplit = computed(() => hasDifficultyLevel(difficultySplit.value?.difficulty))
const hasSecondPass = computed(() => hasDifficultyLevel(secondPass.value?.final_level))

const difficultyValueClass = computed(() => {
  return difficultyLevel.value === DIFF_HARD ? 'route-value-danger' : 'route-value-success'
})

const secondPassValueClass = computed(() => {
  const level = secondPassLevel.value
  if (level === DIFF_HARD) return 'route-value-danger'
  if (level === DIFF_SECOND_EASY) return 'route-value-success'
  return 'route-value-warning'
})

const displayDifficultyText = computed(() => {
  const raw = hasSecondPass.value ? secondPass.value?.final_level : difficultySplit.value?.difficulty
  return getDifficultyLabel(raw) || '—'
})

const displayDifficultyClass = computed(() => {
  return hasSecondPass.value ? secondPassValueClass.value : difficultyValueClass.value
})

const displayDifficultyReason = computed(() => buildDifficultyReason(props.result))

const routeInfo = computed(() => props.result?.route_info || null)

const hasRouteInfo = computed(() => !!routeInfo.value)

const routeConfidenceText = computed(() => {
  const conf = Number(routeInfo.value?.confidence ?? 0)
  if (Number.isNaN(conf) || conf <= 0) return '—'
  return `${(conf * 100).toFixed(2)}%`
})

const routeCandidatesText = computed(() => {
  const items = routeInfo.value?.candidates
  if (!Array.isArray(items) || items.length === 0) return ''
  return items
    .map(item => {
      const name = item?.category || ''
      const score = Number(item?.score ?? 0)
      if (!name) return ''
      if (Number.isNaN(score) || score <= 0) return name
      return `${name}(${(score * 100).toFixed(1)}%)`
    })
    .filter(Boolean)
    .join('，')
})

const routeBreakdownText = computed(() => {
  const breakdown = routeInfo.value?.confidence_breakdown
  if (!breakdown || typeof breakdown !== 'object') return ''

  const labelMap = {
    mode: '模式',
    evidence_score: '证据分',
    margin_score: '分差分',
    anchor_score: '主体词分',
    model_confidence: '模型分',
    llm_confidence: 'LLM分',
    rule_confidence: '规则分',
    rule_support_score: '规则支持分',
    agreement_score: '一致性分',
    strong_hit_count: '强词命中',
    keyword_hit_count: '普通词命中',
    top_raw_score: '原始分',
    same_category: '规则一致',
    rules_direct_threshold: '直通阈值'
  }

  return Object.entries(breakdown)
    .filter(([, value]) => value !== null && value !== undefined && value !== '')
    .map(([key, value]) => {
      const label = labelMap[key] || key
      if (typeof value === 'boolean') {
        return `${label}: ${value ? '是' : '否'}`
      }
      if (typeof value === 'number') {
        if (key.endsWith('_score') || key.endsWith('_confidence') || key === 'rules_direct_threshold') {
          return `${label}: ${(value * 100).toFixed(2)}%`
        }
        return `${label}: ${value}`
      }
      return `${label}: ${value}`
    })
    .join('；')
})

const extractConfidenceV2Rows = computed(() => {
  const raw = props.result?.extract_confidence_v2
  if (!raw || typeof raw !== 'object') return []

  const sourceLabelMap = {
    finetuned_model: '微调模型',
    prompt_extraction: '提示词抽取',
    unknown: '未知来源'
  }

  const reasonLabelMap = {
    field_present_and_schema_valid: '字段已提取，结构完整',
    field_missing_or_invalid: '字段缺失或结构无效',
    field_missing: '字段缺失',
    field_missing_with_anchor: '原文存在锚点，但字段未提取',
    explicit_pattern_match: '原文有明确模式，抽取较稳定',
    partial_pattern_match: '原文有部分模式，抽取存在缺口',
    no_signal_detected: '原文缺少明显信号',
    not_applicable: '原文无该字段信号，不适用'
  }

  const evidenceLabelMap = {
    field_present: '字段已提取',
    structure_valid: '结构有效',
    body_present: '主体值已提取',
    aux_signal_count: '辅助属性数',
    item_count: '提取项数',
    valid_item_count: '有效项数',
    valid_ratio: '有效比例',
    explicit_anchor: '原文有明确锚点',
    has_dn_pair: '存在双DN模式',
    has_length_anchor: '存在长度锚点',
    pair_expected: '应为双值结构',
    pair_complete: '双值提取完整',
    grouped_layers: '存在分层/复合壁厚',
    pressure_anchor: '存在压力锚点',
    value_present: '字段有值',
    base_confidence: '基础置信度',
    validation_issue_count: '验证问题数',
    validation_penalty_factor: '验证惩罚系数',
    validation_reasons: '验证原因'
  }

  const yesNoText = value => (value ? '是' : '否')
  const formatEvidenceValue = (key, value) => {
    if (typeof value === 'boolean') return yesNoText(value)
    if (typeof value === 'number') {
      if (key === 'valid_ratio' || key === 'base_confidence') return `${(value * 100).toFixed(0)}%`
      return String(value)
    }
    return String(value)
  }

  const fieldOrderV2 = ['TYPE', 'SIZE', 'THICKNESS', 'PRESSURE', 'MATERIAL', 'STANDARD']
  return fieldOrderV2
    .map(field => {
      const item = raw[field]
      if (!item || typeof item !== 'object') return null
      const confidence = item.confidence === null || item.confidence === undefined ? null : Number(item.confidence)
      const source = sourceLabelMap[item.source] || item.source || '—'
      const reason = reasonLabelMap[item.reason] || item.reason || '—'
      const evidence = item.evidence && typeof item.evidence === 'object' ? item.evidence : {}
      const evidenceText = Object.entries(evidence)
        .filter(([, value]) => value !== null && value !== undefined && value !== '')
        .map(([key, value]) => {
          const label = evidenceLabelMap[key] || key
          return `${label}: ${formatEvidenceValue(key, value)}`
        })
        .join('，')

      return {
        field,
        fieldLabel: typeLabels[field] || field,
        summary: `置信度 ${confidence === null || Number.isNaN(confidence) ? '—' : `${(confidence * 100).toFixed(2)}%`}；来源：${source}；判断：${reason}${evidenceText ? `；依据：${evidenceText}` : ''}`
      }
    })
    .filter(Boolean)
})

const encodeConfidenceV2Rows = computed(() => {
  const fields = props.result?.fields
  if (!fields || typeof fields !== 'object') return []

  const sourceLabelMap = {
    type_mapping: '类型映射',
    material_mapping: '材质映射',
    standard_processor: '规范规则编码器',
    size_processor: '尺寸编码器',
    thickness_processor: '壁厚编码器',
    pressure_processor: '压力编码器',
    llm_fallback: '大模型兜底',
    regex_direct: '正则直提',
    mixed: '混合来源',
    unknown: '未知来源'
  }

  const reasonLabelMap = {
    type_mapping_resolved: '类型映射命中',
    type_mapping_unresolved_llm_used: '类型映射未命中，转大模型兜底',
    type_mapping_unresolved_llm_failed: '类型映射未命中，且大模型兜底失败',
    material_mapping_resolved: '材质映射命中',
    standard_processor_resolved: '规范规则编码成功',
    standard_processor_unresolved_llm_used: '规范规则未命中，转大模型兜底',
    standard_processor_unresolved_llm_failed: '规范规则未命中，且大模型兜底失败',
    size_processor_resolved: '尺寸编码成功',
    size_processor_failed: '尺寸编码失败',
    thickness_processor_resolved: '壁厚编码成功',
    thickness_processor_failed: '壁厚编码失败',
    pressure_processor_resolved: '压力编码成功',
    pressure_processor_failed: '压力编码失败',
    llm_fallback_used: '使用大模型兜底',
    llm_fallback_failed: '大模型兜底失败',
    regex_direct_match: '正则直接命中',
    multi_item_aggregated: '多项结果聚合',
    fallback_from_similarity: '由旧相似度回填',
    no_item_results: '没有可用编码项'
  }

  const evidenceLabelMap = {
    strategy: '命中策略',
    matched_key: '命中键',
    body_present: '主体值存在',
    base_code: '基础编码',
    suffix_count: '后缀数',
    item_count: '编码项数',
    code_present: '编码结果存在',
    need_review: '需要审核',
    field_type: '字段类型',
    formatted_present: '格式化结果存在',
    category: '分类',
    source_count: '来源种类数',
    mapping_item_count: '映射项数',
    processor_item_count: '规则项数',
    llm_item_count: '大模型项数',
    used_model_confidence: '使用模型置信度',
    merged_value_present: '编码输入存在',
    value_present: '值存在'
  }

  const yesNoText = value => (value ? '是' : '否')
  const formatEvidenceValue = (key, value) => {
    if (typeof value === 'boolean') return yesNoText(value)
    if (typeof value === 'number') return String(value)
    return String(value)
  }

  const fieldOrderV2 = ['TYPE', 'SIZE', 'THICKNESS', 'PRESSURE', 'MATERIAL', 'STANDARD']
  return fieldOrderV2
    .map(field => {
      const item = fields[field]?.encode_confidence_v2
      if (!item || typeof item !== 'object') return null
      const confidence = Number(item.confidence ?? 0)
      const source = sourceLabelMap[item.source] || item.source || '—'
      const reason = reasonLabelMap[item.reason] || item.reason || '—'
      const evidence = item.evidence && typeof item.evidence === 'object' ? item.evidence : {}
      const evidenceText = Object.entries(evidence)
        .filter(([, value]) => value !== null && value !== undefined && value !== '')
        .map(([key, value]) => {
          const label = evidenceLabelMap[key] || key
          return `${label}: ${formatEvidenceValue(key, value)}`
        })
        .join('，')

      return {
        field,
        fieldLabel: typeLabels[field] || field,
        summary: `置信度 ${(confidence * 100).toFixed(2)}%；来源：${source}；判断：${reason}${evidenceText ? `；依据：${evidenceText}` : ''}`
      }
    })
    .filter(Boolean)
})

const fieldConfidenceRows = computed(() => {
  const fields = props.result?.fields
  if (!fields || typeof fields !== 'object') return []

  const fieldOrderV2 = ['TYPE', 'SIZE', 'THICKNESS', 'PRESSURE', 'MATERIAL', 'STANDARD']
  const formatPercent = value => {
    const num = Number(value)
    if (Number.isNaN(num) || num < 0) return '—'
    return `${(num * 100).toFixed(2)}%`
  }

  return fieldOrderV2
    .map(field => {
      const item = fields[field]
      if (!item || typeof item !== 'object') return null
      const hasAny =
        item.stage1_confidence !== null && item.stage1_confidence !== undefined ||
        item.stage2_confidence !== null && item.stage2_confidence !== undefined ||
        item.field_confidence !== null && item.field_confidence !== undefined
      if (!hasAny) return null
      return {
        field,
        fieldLabel: typeLabels[field] || field,
        summary: `一阶段 ${formatPercent(item.stage1_confidence)}；二阶段 ${formatPercent(item.stage2_confidence)}；字段最终 ${formatPercent(item.field_confidence)}`
      }
    })
    .filter(Boolean)
})

const hasCorrection = computed(() => {
  const fields = props.result?.fields || {}
  return Object.values(fields).some(field => {
    if (!field) return
    if (field.manual_override) return true
    return (field.items || []).some(item => item?.manual_override)
  })
})

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

function shouldRenderItems(type, field) {
  if (type === 'STANDARD') {
    return !!(field?.items && field.items.length > 0)
  }
  return !!(field?.items && field.items.length > 1)
}

function shouldHideOriginalValue(type) {
  return type === 'MATERIAL'
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

function formatTypeSummary(parsed) {
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return String(parsed || '')
  }

  const parts = []
  const body = String(parsed.BODY || '').trim()
  const geometry = parsed.GEOMETRY && typeof parsed.GEOMETRY === 'object' ? parsed.GEOMETRY : {}
  const angle = String(geometry.ANGLE || '').trim()
  const radius = String(geometry.RADIUS || '').trim()
  const manu = Array.isArray(parsed.MANU) ? parsed.MANU.map(item => String(item || '').trim()).filter(Boolean) : []
  const conn = Array.isArray(parsed.CONN) ? parsed.CONN.map(item => String(item || '').trim()).filter(Boolean) : []
  const seal = Array.isArray(parsed.SEAL) ? parsed.SEAL.map(item => String(item || '').trim()).filter(Boolean) : []
  const ends = Array.isArray(parsed.ENDS) ? parsed.ENDS.map(item => String(item || '').trim()).filter(Boolean) : []

  if (angle && body) {
    parts.push(`${angle}度${body}`)
  } else if (body) {
    parts.push(body)
  } else if (angle) {
    parts.push(`${angle}度`)
  }
  if (radius) parts.push(radius)
  if (manu.length) parts.push(manu.join(' x '))
  if (conn.length) parts.push(conn.join(' x '))
  if (seal.length) parts.push(seal.join(' x '))
  if (ends.length) parts.push(ends.join(' x '))
  return parts.join(';')
}

function getFieldOriginalText(type, field) {
  const parsed = safeParseJson(field.original_value || '')
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return field.original_value || ''
  }

  if (type === 'TYPE') {
    const parts = []
    const body = String(parsed.BODY || '').trim()
    const geometry = parsed.GEOMETRY && typeof parsed.GEOMETRY === 'object' ? parsed.GEOMETRY : {}
    const angle = String(geometry.ANGLE || '').trim()
    const radius = String(geometry.RADIUS || '').trim()
    const manu = Array.isArray(parsed.MANU) ? parsed.MANU.map(item => String(item || '').trim()).filter(Boolean) : []
    const conn = Array.isArray(parsed.CONN) ? parsed.CONN.map(item => String(item || '').trim()).filter(Boolean) : []
    const seal = Array.isArray(parsed.SEAL) ? parsed.SEAL.map(item => String(item || '').trim()).filter(Boolean) : []
    const ends = Array.isArray(parsed.ENDS) ? parsed.ENDS.map(item => String(item || '').trim()).filter(Boolean) : []
    if (angle) parts.push(`${angle}度`)
    if (body) parts.push(body)
    if (radius) parts.push(radius)
    if (manu.length) parts.push(manu.join(' x '))
    if (conn.length) parts.push(conn.join(' x '))
    if (seal.length) parts.push(seal.join(' x '))
    if (ends.length) parts.push(ends.join(' x '))
    return parts.join(';')
  }

  if (type !== 'SIZE' && type !== 'THICKNESS') {
    return field.original_value || ''
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
      const text = values.map(item => (item && typeof item === 'object') ? '' : String(item || '').trim()).filter(Boolean).join(' x ')
      return text ? `${key}: ${text}` : ''
    })
    .filter(Boolean)
    .join(' ; ')
}

function getFieldPrimaryText(type, field) {
  if (shouldHideOriginalValue(type)) {
    return field.encoding_input || field.matched_name || getFieldOriginalText(type, field) || ''
  }
  return getFieldOriginalText(type, field)
}

function formatFieldTextByType(type, value) {
  if (type === 'TYPE') {
    const parsed = safeParseJson(value)
    return formatTypeSummary(parsed)
  }
  const text = String(value || '').trim()
  if (!text) return ''
  if (type === 'SIZE' || type === 'THICKNESS' || type === 'PRESSURE') {
    return getFieldOriginalText(type, { original_value: text }) || text
  }
  return text
}

function normalizeComparableText(text) {
  return String(text || '')
    .replace(/\s+/g, ' ')
    .replace(/\s*;\s*/g, ';')
    .replace(/\s*[xX×]\s*/g, 'x')
    .trim()
}

function getFieldProcessedText(type, field) {
  let candidate = null
  if (type === 'TYPE') {
    candidate = field?.encoding_input ?? field?.stage1_final_value ?? null
  } else {
    candidate = field?.encoding_input ?? field?.stage1_final_value ?? null
  }
  if (candidate === null || candidate === undefined || candidate === '') return ''
  return formatFieldTextByType(type, candidate)
}

function getFieldDisplayText(type, field) {
  const originalText = getFieldPrimaryText(type, field)
  const processedText = getFieldProcessedText(type, field)
  if (!processedText) return originalText
  if (normalizeComparableText(processedText) === normalizeComparableText(originalText)) {
    return originalText
  }
  return `${originalText} → ${processedText}`
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

function emitEditField(type, index = null) {
  emit('edit-field', { type, index })
}

</script>

<style scoped>
.encoding-result {
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  overflow: visible;
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

.conversion-notes-section {
  padding: 10px 16px;
  background: #f6fbff;
  border-top: 1px solid #d9ebff;
}

.conversion-note-item {
  font-size: 11px;
  color: #2f527a;
  padding: 2px 0;
}

.route-section {
  padding: 10px 16px;
  background: #fafcff;
  border-top: 1px solid var(--border-light);
  overflow: visible;
}

.route-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.route-header-main {
  display: flex;
  flex-direction: column;
  gap: 2px;
  min-width: 0;
}

.route-header-collapsible {
  cursor: pointer;
  user-select: none;
}

.route-collapse-icon {
  font-size: 11px;
  color: var(--text-muted);
  flex-shrink: 0;
}

.route-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
}

.route-hint {
  font-size: 10px;
  color: var(--text-muted);
}

.route-meta-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 8px;
  margin-bottom: 8px;
}

.route-meta-chip {
  min-width: 0;
  background: var(--bg-secondary);
  border: 1px solid var(--border-light);
  border-radius: 6px;
  padding: 8px 10px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.route-detail-line {
  background: var(--bg-secondary);
  border: 1px solid var(--border-light);
  border-radius: 6px;
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
  overflow: visible;
}

.route-detail-line + .route-detail-line {
  margin-top: 8px;
}

.route-label {
  font-size: 10px;
  color: var(--text-secondary);
  text-transform: uppercase;
  line-height: 1.3;
}

.route-value {
  display: block;
  width: 100%;
  font-size: 12px;
  line-height: 1.6;
  color: var(--text-primary);
  white-space: normal;
  word-break: break-word;
  overflow-wrap: anywhere;
}

.route-value-strong {
  font-weight: 600;
}

.route-value-success {
  color: var(--success);
}

.route-value-danger {
  color: var(--danger);
}

.route-value-warning {
  color: #b26a00;
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

@media (max-width: 768px) {
  .route-meta-list {
    grid-template-columns: 1fr;
  }

  .route-meta-chip {
    min-width: 100%;
  }
}

</style>
