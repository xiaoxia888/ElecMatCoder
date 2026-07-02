<template>
  <div class="review-detail-view">
    <aside class="sidebar">
      <div class="sidebar-section summary-section">
        <router-link to="/review" class="back-link">← 返回任务列表</router-link>
        <div class="task-title">审核详情</div>
        <div class="summary-meta">
          <div class="meta-row meta-row-full">
            <span class="meta-label">任务编号</span>
            <span class="meta-value">{{ detailData?.taskCode || '--' }}</span>
          </div>
          <div class="meta-row">
            <span class="meta-label">审核人</span>
            <span class="meta-value">{{ detailData?.reviewer || '--' }}</span>
          </div>
          <div class="meta-row">
            <span class="meta-label">审核日期</span>
            <span class="meta-value">{{ formatDate(detailData?.reviewDate) }}</span>
          </div>
        </div>

        <div class="search-section" v-if="items.length > 0">
          <div class="summary-divider"></div>
          <div class="section-header compact">
            <span class="section-title">材料定位</span>
          </div>
          <div class="jump-search">
            <input
              class="jump-search-input"
              v-model="jumpKeyword"
              type="text"
              placeholder="输入材料描述模糊查找"
              @focus="showSearchDropdown = searchResults.length > 0"
              @input="handleSearchInput"
            />
            <button v-if="jumpKeyword" type="button" class="clear-btn" @click="clearSearch">×</button>
            <div v-if="showSearchDropdown && searchResults.length > 0" class="search-dropdown">
              <div
                v-for="item in searchResults"
                :key="item.id"
                class="search-option"
                @mouseenter="scheduleTooltip('search-option', item.index)"
                @mouseleave="hideTooltip"
                @mousedown.prevent="selectSearchResult(item)"
              >
                <span class="search-index">#{{ item.index + 1 }}</span>
                <span class="search-text">{{ item.description }}</span>
                <div v-if="activeTooltip.type === 'search-option' && activeTooltip.index === item.index" class="search-option-tooltip">
                  {{ item.description || item.name || '--' }}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="sidebar-section list-section" v-if="items.length > 0">
        <div class="section-header">
          <span class="section-title">材料列表</span>
          <span class="section-count">共 {{ items.length }} 条</span>
        </div>
        <div class="data-list" ref="listRef">
          <div
            v-for="(item, index) in items"
            :key="item.id || index"
            class="list-item"
            :class="{ active: currentIndex === index, corrected: isItemCorrected(index, item) }"
            :data-index="index"
            @click="selectItem(index)"
            @mouseenter="scheduleTooltip('list-item', index)"
            @mouseleave="hideTooltip"
          >
            <span class="item-num">#{{ index + 1 }}</span>
            <span class="item-text">{{ item.description || item.name || '--' }}</span>
            <span class="item-badge corrected" v-if="isItemCorrected(index, item)">改</span>
            <div v-if="activeTooltip.type === 'list-item' && activeTooltip.index === index" class="item-tooltip">
              {{ item.description || item.name || '--' }}
            </div>
          </div>
        </div>
      </div>
    </aside>

    <section class="content">
      <div v-if="loading" class="state-card">加载中...</div>
      <div v-else-if="!currentItem" class="state-card">暂无材料详情</div>
      <template v-else>
        <div class="source-card">
          <div class="card-header">
            <span class="card-title">原始描述</span>
            <div class="card-index-wrapper">
              <span class="card-index-hash">#</span>
              <input
                type="text"
                class="page-input"
                :value="currentIndex + 1"
                @keydown.enter="handlePageJump"
                @blur="handlePageJump"
              />
              <span class="card-index-total">/ {{ items.length }}</span>
            </div>
          </div>
          <div class="source-text">{{ currentItem.description || currentItem.name || '--' }}</div>
        </div>

        <div class="result-card">
          <div class="result-header">
            <div class="code-display">
              <span class="code-label">编码结果</span>
              <div class="code-box success">
                {{ currentItem.code || '--' }}
              </div>
            </div>
            <div class="status-info">
              <span class="status-tag" :class="{ warning: hasCorrectedCode, success: !hasCorrectedCode }">
                {{ hasCorrectedCode ? '已有修正' : '原始结果' }}
              </span>
            </div>
          </div>
          <div class="corrected-row">
            <span class="corrected-label">修正后编码</span>
            <span class="corrected-value" :class="{ empty: !hasCorrectedCode }">{{ correctedCodeText }}</span>
          </div>
          <div class="fields-section">
            <div class="fields-header">
              <span class="fields-title">字段分解</span>
              <span class="fields-hint">双击右侧编码可修正</span>
            </div>
            <div class="fields-list">
              <div v-for="field in fieldRows" :key="field.key" class="field-group">
                <div class="field-item">
                  <div class="field-main">
                    <span class="type-tag" :class="getFieldTypeClass(field.key)">{{ field.label }}</span>
                    <span class="field-original" :title="field.raw || '--'">{{ field.raw || '--' }}</span>
                    <span
                      class="field-code"
                      :class="{ corrected: field.isCorrected }"
                      :title="field.displayCode || '--'"
                      @dblclick="openFieldEditor(field)"
                    >
                      {{ field.displayCode || '--' }}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="write-panel">
          <div class="panel-header">
            <span class="panel-title">写入审核结果</span>
          </div>
          <div class="panel-actions">
            <button
              type="button"
              class="btn btn-danger btn-sm"
              :disabled="writingCorrections || !hasPendingCorrectionChanges"
              @click="writeCorrectionsToH3yun"
            >
              {{ writingCorrections ? '写入中...' : '写入氚云' }}
            </button>
          </div>
        </div>
      </template>
    </section>

    <div v-if="editorVisible && editingField" class="editor-overlay" @click.self="closeFieldEditor">
      <div class="editor-dialog">
        <div class="editor-header">
          <div class="editor-title">修正{{ editingField.label }}编码</div>
          <button type="button" class="editor-close" @click="closeFieldEditor">×</button>
        </div>
        <div class="editor-body">
          <div class="editor-row">
            <span class="editor-label">原始内容</span>
            <div class="editor-value">{{ editingField.raw || '--' }}</div>
          </div>
          <div class="editor-row">
              <span class="editor-label">模型生成编码</span>
              <div class="editor-value code">{{ editingField.code || '--' }}</div>
          </div>
          <div class="editor-row">
            <span class="editor-label">修正编码</span>
            <input
              ref="editorInputRef"
              v-model="editingValue"
              type="text"
              class="editor-input"
              placeholder="请输入修正后的标签编码"
              @keydown.enter="applyFieldEdit"
            />
          </div>
        </div>
        <div class="editor-actions">
          <button type="button" class="editor-btn editor-btn-light" @click="closeFieldEditor">取消</button>
          <button type="button" class="editor-btn editor-btn-primary" @click="applyFieldEdit">确定</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick, watch, inject } from 'vue'
import { useRoute } from 'vue-router'
import { getTaskObjectDetail, writeTaskCorrections } from '../api/h3yun'

const route = useRoute()
const showToast = inject('showToast')

const loading = ref(false)
const detailData = ref(null)
const currentIndex = ref(0)
const jumpKeyword = ref('')
const showSearchDropdown = ref(false)
const listRef = ref(null)
const editorInputRef = ref(null)
const activeTooltip = ref({ type: '', index: -1 })
const localFieldEdits = ref({})
const editorVisible = ref(false)
const editingField = ref(null)
const editingValue = ref('')
const writingCorrections = ref(false)

const TOOLTIP_DELAY_MS = 500
let tooltipTimer = null
const fieldDefinitions = [
  { key: 'type', label: '种类', rawKey: 'typeRaw', codeKey: 'typeCode', correctedKey: 'correctedType' },
  { key: 'size', label: '尺寸', rawKey: 'sizeRaw', codeKey: 'sizeCode', correctedKey: 'correctedSize' },
  { key: 'thickness', label: '壁厚', rawKey: 'thicknessRaw', codeKey: 'thicknessCode', correctedKey: 'correctedThickness' },
  { key: 'pressure', label: '磅级', rawKey: 'pressureRaw', codeKey: 'pressureCode', correctedKey: 'correctedPressure' },
  { key: 'material', label: '材质', rawKey: 'materialRaw', codeKey: 'materialCode', correctedKey: 'correctedMaterial' },
  { key: 'standard', label: '规范', rawKey: 'standardRaw', codeKey: 'standardCode', correctedKey: 'correctedStandard' },
]

const items = computed(() => detailData.value?.items || [])
const currentItem = computed(() => items.value[currentIndex.value] || null)

const searchResults = computed(() => {
  const keyword = jumpKeyword.value.trim().toLowerCase()
  if (!keyword) return []
  return items.value
    .map((item, index) => ({ ...item, index }))
    .filter(item => String(item.description || item.name || '').toLowerCase().includes(keyword))
    .slice(0, 30)
})

const fieldRows = computed(() => buildFieldRows(currentItem.value || {}, currentIndex.value))

const correctedCodeText = computed(() => {
  const item = currentItem.value || {}
  return getEffectiveCorrectedCode(currentIndex.value, item, fieldRows.value)
})

const hasCorrectedCode = computed(() => correctedCodeText.value !== '--')
const pendingCorrectionItems = computed(() => {
  return items.value
    .map((item, index) => buildCorrectionPayload(index, item))
    .filter(payload => isCorrectionPayloadChanged(payload))
})
const hasPendingCorrectionChanges = computed(() => pendingCorrectionItems.value.length > 0)

function handleKeydown(event) {
  const tagName = event.target?.tagName
  if (tagName === 'INPUT' || tagName === 'TEXTAREA' || tagName === 'SELECT') return
  if (editorVisible.value) return
  if (!items.value.length) return

  if (event.key === 'ArrowLeft' && currentIndex.value > 0) {
    event.preventDefault()
    selectItem(currentIndex.value - 1)
  } else if (event.key === 'ArrowRight' && currentIndex.value < items.value.length - 1) {
    event.preventDefault()
    selectItem(currentIndex.value + 1)
  }
}

function normalizeFieldValue(value) {
  const text = String(value ?? '').trim()
  return text ? text : null
}

function normalizeComparableValue(value) {
  return String(value ?? '').trim()
}

function getLocalFieldEdit(fieldKey, itemIndex = currentIndex.value) {
  const itemEdits = localFieldEdits.value[itemIndex] || {}
  return Object.prototype.hasOwnProperty.call(itemEdits, fieldKey) ? itemEdits[fieldKey] : undefined
}

function hasLocalFieldEdit(fieldKey, itemIndex = currentIndex.value) {
  const itemEdits = localFieldEdits.value[itemIndex] || {}
  return Object.prototype.hasOwnProperty.call(itemEdits, fieldKey)
}

function buildFieldRows(item, itemIndex) {
  return fieldDefinitions.map(field => {
    const originalCode = normalizeFieldValue(item[field.codeKey])
    const initialCorrectedCode = normalizeFieldValue(item[field.correctedKey])
    const editedCode = getLocalFieldEdit(field.key, itemIndex)
    const displayCode = hasLocalFieldEdit(field.key, itemIndex)
      ? String(editedCode ?? '').trim()
      : (initialCorrectedCode ?? originalCode)

    return {
      key: field.key,
      label: field.label,
      raw: item[field.rawKey],
      code: originalCode,
      initialCorrectedCode,
      displayCode,
      isCorrected: displayCode !== originalCode
    }
  })
}

function getJoinedFieldCode(rows) {
  return rows
    .map(field => normalizeFieldValue(field.displayCode))
    .filter(Boolean)
    .join('')
}

function getPersistedCorrectedFieldValue(index, item, field) {
  const originalCode = normalizeComparableValue(item ? item[field.codeKey] : '')
  if (hasLocalFieldEdit(field.key, index)) {
    const localValue = normalizeComparableValue(getLocalFieldEdit(field.key, index))
    return localValue === originalCode ? '' : localValue
  }

  const backendValue = normalizeComparableValue(item ? item[field.correctedKey] : '')
  return backendValue === originalCode ? '' : backendValue
}

function getEffectiveCorrectedCode(index, item, rows = buildFieldRows(item || {}, index)) {
  const originalFinalCode = normalizeFieldValue(item?.code)
  const backendCorrectedCode = normalizeFieldValue(item?.correctedCode)
  const hasLocalEdit = Object.keys(localFieldEdits.value[index] || {}).length > 0

  if (hasLocalEdit) {
    const joinedCode = getJoinedFieldCode(rows)
    return joinedCode && joinedCode !== originalFinalCode ? joinedCode : '--'
  }

  return backendCorrectedCode && backendCorrectedCode !== originalFinalCode ? backendCorrectedCode : '--'
}

function isItemCorrected(index, item) {
  return getEffectiveCorrectedCode(index, item) !== '--'
}

function buildCorrectionPayload(index, item) {
  const rows = buildFieldRows(item || {}, index)
  const correctedCodeValue = getEffectiveCorrectedCode(index, item, rows)
  const payload = {
    id: item?.id || '',
    correctedCode: correctedCodeValue === '--' ? '' : correctedCodeValue
  }

  fieldDefinitions.forEach(field => {
    const payloadKey = field.correctedKey.replace(/^corrected/, 'corrected')
    payload[payloadKey] = getPersistedCorrectedFieldValue(index, item, field)
  })

  return payload
}

function isCorrectionPayloadChanged(payload) {
  const item = items.value.find(entry => (entry?.id || '') === payload.id)
  if (!item) return false

  if (normalizeComparableValue(item.correctedCode) !== normalizeComparableValue(payload.correctedCode)) {
    return true
  }

  return fieldDefinitions.some(field => {
    const key = field.correctedKey.replace(/^corrected/, 'corrected')
    return normalizeComparableValue(item[field.correctedKey]) !== normalizeComparableValue(payload[key])
  })
}

function getFieldTypeClass(type) {
  const map = {
    type: 'type-blue',
    size: 'type-orange',
    thickness: 'type-pink',
    pressure: 'type-red',
    material: 'type-green',
    standard: 'type-gray'
  }
  return map[type] || ''
}

function formatDate(value) {
  if (!value) return '--'
  return String(value).replace(/\//g, '-')
}

function handleSearchInput() {
  showSearchDropdown.value = searchResults.value.length > 0
}

function clearSearch() {
  jumpKeyword.value = ''
  showSearchDropdown.value = false
  hideTooltip()
}

function selectItem(index) {
  closeFieldEditor()
  currentIndex.value = index
  scrollToItem(index)
}

function handlePageJump(event) {
  const raw = String(event.target.value || '').trim()
  const num = Number(raw)
  if (!Number.isFinite(num)) {
    event.target.value = currentIndex.value + 1
    return
  }
  const nextIndex = Math.max(0, Math.min(items.value.length - 1, Math.trunc(num) - 1))
  currentIndex.value = nextIndex
  event.target.value = nextIndex + 1
  scrollToItem(nextIndex)
}

function selectSearchResult(item) {
  jumpKeyword.value = item.description || item.name || ''
  showSearchDropdown.value = false
  currentIndex.value = item.index
  scrollToItem(item.index)
  hideTooltip()
}

function scrollToItem(index) {
  nextTick(() => {
    const container = listRef.value
    if (!container) return
    const target = container.querySelector(`.list-item[data-index="${index}"]`)
    target?.scrollIntoView({ block: 'nearest' })
  })
}

function handleClickOutside(event) {
  if (!event.target.closest('.jump-search')) {
    showSearchDropdown.value = false
  }
}

function clearTooltipTimer() {
  if (tooltipTimer) {
    clearTimeout(tooltipTimer)
    tooltipTimer = null
  }
}

function scheduleTooltip(type, index = -1) {
  clearTooltipTimer()
  activeTooltip.value = { type: '', index: -1 }
  tooltipTimer = setTimeout(() => {
    activeTooltip.value = { type, index }
  }, TOOLTIP_DELAY_MS)
}

function hideTooltip() {
  clearTooltipTimer()
  activeTooltip.value = { type: '', index: -1 }
}

function openFieldEditor(field) {
  editingField.value = { ...field }
  editingValue.value = field.displayCode || ''
  editorVisible.value = true
  nextTick(() => {
    editorInputRef.value?.focus()
    editorInputRef.value?.select()
  })
}

function closeFieldEditor() {
  editorVisible.value = false
  editingField.value = null
  editingValue.value = ''
}

function applyFieldEdit() {
  if (!editingField.value) return
  const itemEdits = {
    ...(localFieldEdits.value[currentIndex.value] || {})
  }
  itemEdits[editingField.value.key] = String(editingValue.value || '').trim()
  localFieldEdits.value = {
    ...localFieldEdits.value,
    [currentIndex.value]: itemEdits
  }
  closeFieldEditor()
}

async function writeCorrectionsToH3yun() {
  if (!detailData.value?.id) {
    showToast?.('缺少任务ID，无法写入氚云', 'error')
    return
  }
  if (!hasPendingCorrectionChanges.value) {
    showToast?.('没有需要写入的修正数据', 'warning')
    return
  }

  writingCorrections.value = true
  try {
    const changedItems = pendingCorrectionItems.value
    await writeTaskCorrections(detailData.value.id, changedItems)

    const changedPayloadMap = {}
    changedItems.forEach(item => {
      changedPayloadMap[item.id] = item
    })

    detailData.value = {
      ...detailData.value,
      items: items.value.map(item => {
        const payload = changedPayloadMap[item.id]
        if (!payload) return item
        return {
          ...item,
          correctedCode: payload.correctedCode || '',
          correctedType: payload.correctedType || '',
          correctedSize: payload.correctedSize || '',
          correctedThickness: payload.correctedThickness || '',
          correctedPressure: payload.correctedPressure || '',
          correctedMaterial: payload.correctedMaterial || '',
          correctedStandard: payload.correctedStandard || '',
        }
      })
    }

    localFieldEdits.value = {}
    showToast?.(`成功写入氚云 ${changedItems.length} 条修正数据`, 'success')
  } catch (error) {
    console.error('写入氚云失败:', error)
    showToast?.(`写入失败: ${error.message}`, 'error')
  } finally {
    writingCorrections.value = false
  }
}

async function loadDetail() {
  const id = route.params.id
  if (!id) return

  loading.value = true
  try {
    const result = await getTaskObjectDetail(id)
    detailData.value = result.data || null
    localFieldEdits.value = {}
    currentIndex.value = 0
    jumpKeyword.value = ''
    showSearchDropdown.value = false
    closeFieldEditor()
  } catch (error) {
    console.error('加载审核详情失败:', error)
    showToast?.(`加载失败: ${error.message}`, 'error')
  } finally {
    loading.value = false
  }
}

watch(() => route.params.id, () => {
  loadDetail()
})

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
  document.addEventListener('keydown', handleKeydown)
  loadDetail()
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
  document.removeEventListener('keydown', handleKeydown)
  clearTooltipTimer()
})
</script>

<style scoped>
.review-detail-view {
  display: flex;
  height: 100%;
  background: var(--bg-secondary);
  min-height: 0;
}

.sidebar {
  width: 300px;
  min-width: 300px;
  background: var(--bg-primary);
  border-right: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  flex-shrink: 0;
  min-height: 0;
}

.sidebar-section {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-light);
  overflow: hidden;
}

.summary-section {
  padding: 12px 16px;
  overflow: visible;
  position: relative;
  z-index: 5;
}

.back-link {
  color: var(--primary);
  text-decoration: none;
  font-size: 14px;
}

.task-title {
  margin-top: 8px;
  font-size: 20px;
  font-weight: 600;
  color: var(--text-primary);
}

.summary-meta {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px 14px;
  margin-top: 12px;
}

.meta-row-full {
  grid-column: 1 / -1;
}

.meta-row {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}

.meta-label {
  color: var(--text-secondary);
  font-size: 12px;
}

.meta-value {
  color: var(--text-primary);
  font-size: 13px;
  text-align: left;
  word-break: break-all;
}

.summary-divider {
  height: 1px;
  background: var(--border-light);
  margin: 12px 0 0;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.section-header.compact {
  padding: 10px 0 0;
  margin-bottom: 0;
}

.section-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.section-count {
  color: var(--text-muted);
  font-size: 12px;
}

.search-section {
  overflow: visible;
}

.jump-search {
  position: relative;
  padding: 10px 0 0;
}

.jump-search-input {
  width: 100%;
  height: 32px;
  padding: 0 36px 0 12px;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  background: var(--bg-primary);
  color: var(--text-primary);
  font-size: 12px;
  outline: none;
}

.jump-search-input:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.12);
}

.clear-btn {
  position: absolute;
  top: 16px;
  right: 10px;
  border: none;
  background: transparent;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 16px;
}

.search-dropdown {
  position: absolute;
  left: 0;
  right: 0;
  top: 48px;
  max-height: 260px;
  overflow-y: auto;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  background: var(--bg-primary);
  box-shadow: var(--shadow-md);
  z-index: 20;
}

.search-option {
  display: flex;
  gap: 8px;
  align-items: center;
  padding: 10px 12px;
  cursor: pointer;
}

.search-option-tooltip,
.item-tooltip {
  padding: 10px 12px;
  border-radius: 8px;
  background: rgba(40, 40, 40, 0.92);
  color: #fff;
  font-size: 12px;
  line-height: 1.5;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
  white-space: normal;
  word-break: break-word;
  pointer-events: none;
}

.search-option:hover {
  background: var(--bg-tertiary);
}

.search-option {
  position: relative;
}

.search-option-tooltip {
  position: absolute;
  left: 8px;
  right: 8px;
  top: calc(100% + 2px);
  z-index: 26;
}

.search-index {
  color: var(--text-muted);
  font-size: 12px;
  flex-shrink: 0;
}

.search-text {
  font-size: 13px;
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.list-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
  overflow: hidden;
  border-bottom: none;
  padding-bottom: 0;
}

.data-list {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
  overflow-y: auto;
  overflow-x: hidden;
  min-height: 0;
  margin: 0;
  padding: 0 8px 0 0;
  scrollbar-gutter: stable;
}

.list-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.15s;
  position: relative;
  border-left: 3px solid transparent;
  flex-shrink: 0;
}

.list-item:hover {
  background: var(--bg-tertiary);
}

.list-item.active {
  background: var(--primary-light);
  border-left-color: var(--primary);
}

.list-item.corrected {
  border-left-color: var(--danger);
}

.list-item.corrected.active {
  border-left-color: var(--danger);
}

.item-num {
  min-width: 24px;
  color: var(--text-muted);
  font-size: 10px;
}

.item-text {
  flex: 1;
  font-size: 12px;
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.item-tooltip {
  position: absolute;
  left: 10px;
  right: 10px;
  top: calc(100% + 4px);
  z-index: 30;
}

.item-badge {
  font-size: 9px;
  font-weight: 600;
  padding: 1px 5px;
  border-radius: 2px;
}

.item-badge.corrected {
  background: var(--danger-light);
  color: var(--danger);
}

.content {
  flex: 1;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  overflow-y: auto;
  min-height: 0;
  min-width: 0;
}

.state-card,
.source-card,
.result-card {
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  overflow: hidden;
}

.state-card {
  min-height: 240px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
}

.source-card {
  padding: 14px 16px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.card-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
}

.card-index-wrapper {
  display: flex;
  align-items: center;
  font-size: 11px;
  color: var(--text-muted);
  border: 1px solid var(--border-color);
  border-radius: 4px;
  padding: 2px 6px;
}

.card-index-hash,
.card-index-total {
  color: var(--text-muted);
}

.page-input {
  width: 40px;
  border: none;
  background: transparent;
  font-size: 11px;
  color: var(--primary);
  font-weight: 500;
  text-align: center;
  outline: none;
  padding: 2px 0;
}

.page-input:focus {
  background: var(--bg-tertiary);
  border-radius: 2px;
}

.source-text {
  color: var(--text-primary);
  line-height: 1.7;
  word-break: break-word;
}

.result-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 14px;
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

.status-tag {
  font-size: 10px;
  font-weight: 600;
  padding: 3px 8px;
  border-radius: 3px;
  background: var(--success-light);
  color: var(--success);
}

.status-tag.warning {
  background: var(--danger-light);
  color: var(--danger);
}

.corrected-row {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 16px 0;
}

.corrected-label {
  font-size: 12px;
  color: var(--text-secondary);
  white-space: nowrap;
}

.corrected-value {
  font-size: 13px;
  font-weight: 600;
  color: var(--danger);
  font-family: 'SF Mono', Monaco, 'Courier New', monospace;
  word-break: break-all;
}

.corrected-value.empty {
  color: var(--text-muted);
  font-weight: 500;
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

.type-blue {
  background: #e3f2fd;
  color: #1565c0;
}

.type-orange {
  background: #fff3e0;
  color: #ef6c00;
}

.type-pink {
  background: #fce4ec;
  color: #c2185b;
}

.type-red {
  background: #ffebee;
  color: #c62828;
}

.type-green {
  background: #e8f5e9;
  color: #2e7d32;
}

.type-gray {
  background: #eceff1;
  color: #546e7a;
}

.field-original,
.field-code {
  min-width: 0;
  word-break: break-all;
}

.field-original {
  color: var(--text-primary);
  flex: 1;
}

.field-code {
  font-family: 'SF Mono', Monaco, monospace;
  font-weight: 600;
  color: var(--primary);
  margin-left: auto;
  text-align: right;
  cursor: pointer;
}

.field-code.corrected {
  color: var(--danger);
}

.write-panel {
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
  overflow: hidden;
}

.panel-header {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-light);
}

.panel-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}

.panel-actions {
  display: flex;
  justify-content: flex-end;
  padding: 12px 16px;
}

.editor-overlay {
  position: fixed;
  inset: 0;
  z-index: 200;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background: rgba(15, 23, 42, 0.28);
}

.editor-dialog {
  width: min(520px, 100%);
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  box-shadow: 0 24px 60px rgba(15, 23, 42, 0.18);
  overflow: hidden;
}

.editor-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 16px;
  border-bottom: 1px solid var(--border-light);
}

.editor-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
}

.editor-close {
  border: none;
  background: transparent;
  color: var(--text-muted);
  font-size: 20px;
  line-height: 1;
  cursor: pointer;
}

.editor-body {
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 16px;
}

.editor-row {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.editor-label {
  font-size: 12px;
  color: var(--text-secondary);
}

.editor-value {
  padding: 10px 12px;
  border-radius: 8px;
  background: var(--bg-secondary);
  color: var(--text-primary);
  word-break: break-word;
}

.editor-value.code {
  font-family: 'SF Mono', Monaco, 'Courier New', monospace;
}

.editor-input {
  width: 100%;
  height: 40px;
  padding: 0 12px;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  background: var(--bg-primary);
  color: var(--text-primary);
  outline: none;
}

.editor-input:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
}

.editor-actions {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 0 16px 16px;
}

.editor-btn {
  height: 36px;
  padding: 0 14px;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
}

.editor-btn-light {
  background: var(--bg-primary);
  color: var(--text-primary);
}

.editor-btn-primary {
  background: var(--primary);
  border-color: var(--primary);
  color: #fff;
}

@media (max-width: 1200px) {
  .review-detail-view {
    flex-direction: column;
  }

  .sidebar {
    width: 100%;
    border-right: none;
    border-bottom: 1px solid var(--border-color);
  }

  .summary-meta {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
