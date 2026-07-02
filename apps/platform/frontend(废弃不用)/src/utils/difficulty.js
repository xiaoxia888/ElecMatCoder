const DIFF_HARD = 0
const DIFF_EASY = 1
const DIFF_SECOND_EASY = 2

const TEXT_TO_LEVEL = {
  困难: DIFF_HARD,
  简单: DIFF_EASY,
  二次简单: DIFF_SECOND_EASY,
  '0': DIFF_HARD,
  '1': DIFF_EASY,
  '2': DIFF_SECOND_EASY
}

const LEVEL_TO_TEXT = {
  [DIFF_HARD]: '困难',
  [DIFF_EASY]: '简单',
  [DIFF_SECOND_EASY]: '二次简单'
}

export function normalizeDifficultyLevel(value) {
  if (value === null || value === undefined || value === '') return null
  if (typeof value === 'number' && Number.isInteger(value)) {
    return Object.prototype.hasOwnProperty.call(LEVEL_TO_TEXT, value) ? value : null
  }
  const text = String(value).trim()
  if (!text) return null
  if (Object.prototype.hasOwnProperty.call(TEXT_TO_LEVEL, text)) {
    return TEXT_TO_LEVEL[text]
  }
  const parsed = Number.parseInt(text, 10)
  if (Number.isNaN(parsed)) return null
  return Object.prototype.hasOwnProperty.call(LEVEL_TO_TEXT, parsed) ? parsed : null
}

export function hasDifficultyLevel(value) {
  return normalizeDifficultyLevel(value) !== null
}

export function getDifficultyLabel(value) {
  const level = normalizeDifficultyLevel(value)
  return level === null ? '' : LEVEL_TO_TEXT[level]
}

export function buildSecondPassFailureReason(secondPass) {
  if (!secondPass || typeof secondPass !== 'object') return ''

  const results = secondPass.results && typeof secondPass.results === 'object'
    ? secondPass.results
    : {}
  const skipped = secondPass.skipped_fields && typeof secondPass.skipped_fields === 'object'
    ? secondPass.skipped_fields
    : {}

  const labelMap = {
    SIZE: '尺寸',
    THICKNESS: '壁厚',
    PRESSURE: '磅级',
    MATERIAL: '材质',
    TYPE: '种类',
    STANDARD: '规范'
  }
  const order = ['SIZE', 'THICKNESS', 'PRESSURE', 'MATERIAL', 'TYPE', 'STANDARD']

  const failedReasons = order
    .map(field => {
      const payload = results[field]
      if (!payload || typeof payload !== 'object' || payload.passed !== false) return ''
      const reason = String(payload.reason || '').trim()
      if (!reason) return ''
      return `${labelMap[field] || field}: ${reason}`
    })
    .filter(Boolean)

  if (failedReasons.length > 0) {
    return failedReasons.join('；')
  }

  return order
    .map(field => {
      const reason = String(skipped[field] || '').trim()
      if (!reason) return ''
      return `${labelMap[field] || field}: ${reason}`
    })
    .filter(Boolean)
    .join('；')
}

export function buildDifficultyReason(result) {
  const difficulty = result?.difficulty_split || {}
  const level = normalizeDifficultyLevel(difficulty.difficulty)
  if (level !== null && level !== DIFF_EASY) {
    return String(difficulty.reason_text || '').trim()
  }
  return buildSecondPassFailureReason(result?.second_pass || null)
}

export function getDisplayDifficultyLevel(result) {
  const raw = result?.second_pass?.final_level ?? result?.difficulty_split?.difficulty
  return getDifficultyLabel(raw)
}

export {
  DIFF_HARD,
  DIFF_EASY,
  DIFF_SECOND_EASY
}
