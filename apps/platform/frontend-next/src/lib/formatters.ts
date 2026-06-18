import type { EncodingResult, FieldPayload, JsonValue } from '@/types/encoding'

const DIFFICULTY_LABELS: Record<number, string> = {
  0: '困难',
  1: '中等',
  2: '简单',
}

export function getDifficultyLevel(result?: EncodingResult): number | null {
  const rawLevel =
    result?.routing?.final_level ??
    result?.second_pass?.final_level ??
    result?.difficulty_split?.level ??
    result?.difficulty_split?.difficulty
  if (rawLevel == null) return null

  const level = Number(rawLevel)
  return level === 0 || level === 1 || level === 2 ? level : null
}

function isObject(value: JsonValue): value is Record<string, JsonValue> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

export function valueToText(value: JsonValue | undefined, joiner = '；'): string {
  if (value == null) return '—'
  if (typeof value === 'string') return value || '—'
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (Array.isArray(value)) {
    const text = value.map((item) => valueToText(item, joiner)).filter((item) => item && item !== '—').join(joiner)
    return text || '—'
  }
  if (isObject(value)) {
    const orderedItems = value.ordered_items
    if (Array.isArray(orderedItems)) {
      const orderedText = orderedItems
        .map((item) => {
          if (isObject(item)) {
            const type = item.type ? String(item.type) : ''
            const itemValue = item.value ? String(item.value) : ''
            return [type, itemValue].filter(Boolean).join(': ')
          }
          return valueToText(item, joiner)
        })
        .join(' | ')
      if (orderedText) return orderedText
    }
    const text = Object.entries(value)
      .filter(([key]) => key !== 'ordered_items' && key !== 'thickness_mm_context')
      .map(([key, item]) => {
        const itemText = valueToText(item, joiner)
        return itemText && itemText !== '—' ? `${key}: ${itemText}` : ''
      })
      .filter(Boolean)
      .join('；')
    return text || '—'
  }
  return '—'
}

// ===== 字段自然展示（复刻旧前端逻辑，不再用 KEY: value 对象形式）=====
function s(value: JsonValue | undefined): string {
  return value == null ? '' : String(value).trim()
}
function arr(value: JsonValue | undefined): string[] {
  return Array.isArray(value) ? value.map(s).filter(Boolean) : []
}
function asObj(value: JsonValue | undefined): Record<string, JsonValue> {
  return isObject(value as JsonValue) ? (value as Record<string, JsonValue>) : {}
}

export function getTypeCategory(result?: EncodingResult | null): string {
  return s(result?.route_info?.model_category as JsonValue | undefined)
}

function formatTypeValue(value: JsonValue | undefined): string {
  if (!isObject(value as JsonValue)) return s(value)
  const o = value as Record<string, JsonValue>
  const parts: string[] = []

  const pushUnique = (items: string[]) => {
    for (const item of items) {
      const text = item.trim()
      if (text && !parts.includes(text)) parts.push(text)
    }
  }

  pushUnique([s(o.FLANGE_STYLE)])
  const body = s(o.BODY)
  const geo = asObj(o.GEOMETRY)
  pushUnique([body, s(geo.ANGLE), s(geo.RADIUS)])
  pushUnique(arr(o.SEAL))
  pushUnique([...arr(o.CONN), ...arr(o.ENDS)])
  pushUnique(arr(o.MANU))
  return parts.join(';')
}

function formatMaterialItem(item: JsonValue): string {
  if (!isObject(item)) return s(item)
  const o = item as Record<string, JsonValue>
  const role = s(o.ROLE)
  const value = s(o.VALUE)
  const sreq = arr(o.SPECIAL_REQ)
  const body = sreq.length ? `${value}${sreq.join('')}` : value
  if (!role || role === 'MAIN') return body
  return body ? `${role}:${body}` : role
}

function formatStandardItem(item: JsonValue): string {
  if (!isObject(item)) return s(item)
  const o = item as Record<string, JsonValue>
  const main = [s(o.BODY), s(o.GRADE), s(o.APPENDIX), s(o.METHOD)].filter(Boolean).join('')
  const category = s(o.CATEGORY)
  return category ? `${main}（${category}）` : main
}

function formatPressureValue(value: JsonValue | undefined): string {
  // 直接是字符串/标量（如 "PN16"）时原样返回
  if (!isObject(value as JsonValue)) return s(value)
  const items = Array.isArray(asObj(value).items) ? (asObj(value).items as JsonValue[]) : []
  return items
    .map((item) => {
      if (!isObject(item)) return s(item)
      const o = item as Record<string, JsonValue>
      return [s(o.type), s(o.value)].filter(Boolean).join(': ')
    })
    .filter(Boolean)
    .join(' ; ')
}

function formatSizeThk(fieldType: string, value: JsonValue | undefined): string {
  if (!isObject(value as JsonValue)) return s(value)
  const o = value as Record<string, JsonValue>
  const order = fieldType === 'SIZE' ? ['DN', 'OD', 'INCH', 'LENGTH'] : ['MM', 'INCH', 'SCHEDULE', 'SERIES', 'BWG']
  const keys = [
    ...order.filter((k) => Object.prototype.hasOwnProperty.call(o, k)),
    ...Object.keys(o).filter((k) => !order.includes(k) && !k.startsWith('_') && typeof o[k] !== 'object'),
  ]
  return keys
    .map((k) => {
      const vals = (Array.isArray(o[k]) ? (o[k] as JsonValue[]) : [o[k]])
        .map((x) => (isObject(x as JsonValue) ? '' : s(x)))
        .filter(Boolean)
        .join(' x ')
      return vals ? `${k}: ${vals}` : ''
    })
    .filter(Boolean)
    .join(' ; ')
}

// 单行文本
export function formatFieldValue(fieldType: string, value: JsonValue | undefined): string {
  if (value == null) return '—'
  if (fieldType === 'MATERIAL') {
    const items = Array.isArray(value) ? value : [value]
    return items.map(formatMaterialItem).filter(Boolean).join(' ; ') || '—'
  }
  if (fieldType === 'STANDARD') {
    const items = Array.isArray(value) ? value : [value]
    return items.map(formatStandardItem).filter(Boolean).join(' ; ') || '—'
  }
  if (fieldType === 'PRESSURE') return formatPressureValue(value) || '—'
  if (fieldType === 'TYPE') return formatTypeValue(value) || '—'
  if (fieldType === 'SIZE' || fieldType === 'THICKNESS') return formatSizeThk(fieldType, value) || '—'
  return valueToText(value)
}

// 多行（规范/材质多条时一行一条）
export function formatFieldLines(fieldType: string, value: JsonValue | undefined): string[] {
  if (value == null) return ['—']
  if (fieldType === 'STANDARD' || fieldType === 'MATERIAL') {
    const items = Array.isArray(value) ? value : [value]
    const fmt = fieldType === 'STANDARD' ? formatStandardItem : formatMaterialItem
    const lines = items.map(fmt).filter(Boolean)
    return lines.length > 0 ? lines : ['—']
  }
  return [formatFieldValue(fieldType, value)]
}

export function formatFieldStage1Lines(field: FieldPayload | undefined, fieldType: string): string[] {
  return field ? formatFieldLines(fieldType, field.stage1_raw?.value) : ['—']
}

export function formatFieldStage2Lines(field: FieldPayload | undefined, fieldType: string): string[] {
  return field ? formatFieldLines(fieldType, field.stage2_input?.value) : ['—']
}

export function formatFieldCode(field?: FieldPayload) {
  return field?.stage2_output?.code || '—'
}

export function getDifficultyLabel(result?: EncodingResult) {
  const level = getDifficultyLevel(result)
  return level == null ? '待定' : DIFFICULTY_LABELS[level]
}

export function getDifficultyVariant(result?: EncodingResult) {
  const level = getDifficultyLevel(result)
  if (level === 0) return 'danger' as const
  if (level === 1) return 'caution' as const
  if (level === 2) return 'success' as const
  return 'neutral' as const
}

export function getRouteReason(result?: EncodingResult) {
  const routingReason = result?.routing?.reason_text?.trim()
  if (routingReason) return routingReason

  const finalLevel = getDifficultyLevel(result)
  const stage1ReasonText = result?.difficulty_split?.reason_text?.trim()
  const stage1Reasons = Array.isArray(result?.difficulty_split?.reasons)
    ? result!.difficulty_split!.reasons.map((item) => String(item).trim()).filter(Boolean)
    : []
  const stage1Reason = stage1ReasonText || stage1Reasons.join(' | ')
  const stage1Error = Array.isArray(result?.errors) && result.errors.length > 0 ? result.errors[0] : ''

  if (finalLevel === 0) {
    if (stage1Reason) return stage1Reason
    if (stage1Error) return stage1Error
    return '未提供一阶段分流原因'
  }

  const secondPassChecks = Array.isArray(result?.second_pass?.failed_checks) ? result!.second_pass!.failed_checks : []
  if (secondPassChecks.length > 0) {
    const reasonParts = secondPassChecks
      .map((item) => {
        const field = String(item?.field ?? '').trim()
        const reason = String(item?.reason ?? '').trim()
        if (!reason) return ''
        return field ? `${field}: ${reason}` : reason
      })
      .filter(Boolean)
    if (reasonParts.length > 0) return reasonParts.join(' | ')
  }

  if (finalLevel === 1 || result?.second_pass?.final_level != null) {
    return '未提供二次分流原因'
  }

  if (finalLevel === 2) return '无需二次分流原因'
  if (stage1Reason) return stage1Reason
  if (stage1Error) return stage1Error
  return '未提供原因说明'
}

export function getRoutingStageText(result?: EncodingResult) {
  const decisionStage = result?.routing?.decision_stage ?? result?.second_pass?.decision_stage
  if (decisionStage === 'stage1') {
    const stage1Level = result?.routing?.stage1_level ?? result?.second_pass?.stage1_level ?? result?.second_pass?.stage1_difficulty
    return stage1Level === 0 ? '一阶段已判困难，未进入二次分流' : '一阶段直接判定'
  }
  if (decisionStage === 'second_pass') return '一阶段通过后触发二次分流'
  if (decisionStage === 'project_frequency') return '批量完成后叠加项目频次复核'
  if (result?.routing?.final_level == null && result?.second_pass?.final_level == null) return '仅一阶段判定'
  return '分流已完成'
}

export function summarizeItemText(text: string, max = 36) {
  return text.length > max ? `${text.slice(0, max)}…` : text
}
