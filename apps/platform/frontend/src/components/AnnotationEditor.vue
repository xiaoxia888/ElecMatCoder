<template>
  <div class="annotation-editor" tabindex="0" @keydown="handleKeydown" ref="editorRef">
    <!-- 工具栏 -->
    <div class="editor-toolbar">
      <div class="toolbar-left">
        <button class="btn btn-ghost btn-sm" @click="$emit('navigate', -1)" :disabled="currentIndex <= 0">
          ← 上一条
        </button>
        <span class="nav-info">{{ currentIndex + 1 }} / {{ total }}</span>
        <button class="btn btn-ghost btn-sm" @click="$emit('navigate', 1)" :disabled="currentIndex >= total - 1">
          下一条 →
        </button>
      </div>
      <div class="toolbar-right">
        <select class="select" style="width: 160px" v-model="localModel" @change="$emit('model-change', localModel)">
          <option v-for="m in models" :key="m.value" :value="m.value">{{ m.label }}</option>
        </select>
        <button class="btn btn-primary" @click="$emit('tokenize')" :disabled="isLoading || !text">
          <span v-if="isLoading" class="spinner"></span>
          <span v-else>分词</span>
        </button>
        <button 
          class="btn btn-ghost" 
          @click="$emit('clear-tokens')" 
          :disabled="isLoading || localTokens.length === 0"
          title="清除当前记录的分词结果和标签"
        >
          清除标签
        </button>
      </div>
    </div>

    <!-- 原文展示（支持拖拽选择依据） -->
    <div class="source-section">
      <div class="section-label">
        原文
        <span v-if="isSelectingEvidence" class="selecting-hint">请在原文中拖拽选择依据内容</span>
      </div>
      <div 
        class="source-text" 
        :class="{ 'selecting-mode': isSelectingEvidence }"
        @mouseup="handleTextSelection"
      >{{ text || '请选择一条数据' }}</div>
    </div>

    <!-- 材料大类（始终显示） -->
    <div class="type-section">
      <div class="section-label">材料大类（点击后在原文中拖拽选择依据）</div>
      <div class="type-buttons">
        <button 
          v-for="type in typeOptions" 
          :key="type"
          class="type-btn"
          :class="{ active: localTypeClass === type, selecting: isSelectingEvidence && pendingTypeClass === type }"
          @click="startSelectEvidence(type)"
        >
          {{ type }}
        </button>
      </div>
      <div v-if="localTypeEvidence" class="type-evidence">
        依据: <span>{{ localTypeEvidence }}</span>
        <button class="btn btn-ghost btn-sm" @click="clearEvidence">×</button>
      </div>
    </div>

    <!-- 标注区域（始终显示） -->
    <div class="tokens-section">
      <div class="section-label">
        分词结果
        <span class="label-hint">双击编辑 | 右键菜单 | 0-9标签 | M/]合并后 | [合并前</span>
      </div>
      <div class="tokens-container">
        <template v-if="localTokens.length > 0">
          <div 
            v-for="(token, index) in localTokens" 
            :key="index"
            class="token"
            :class="[`tag-${token.tag}`, { 'is-space': isSpace(token.word), 'selected': selectedIndex === index, 'editing': editingIndex === index }]"
            @click="selectToken(index)"
            @dblclick="startEdit(index)"
            @contextmenu.prevent="showContextMenu($event, index)"
          >
            <!-- 词语部分（编辑模式显示输入框，否则显示文字） -->
            <input 
              v-if="editingIndex === index"
              ref="editInputRef"
              v-model="editingValue"
              class="token-edit-input"
              @blur="finishEdit"
              @keydown.enter="finishEdit"
              @keydown.escape="cancelEdit"
              @click.stop
            />
            <span v-else class="token-word">{{ displayWord(token.word) }}</span>
            <!-- 标签始终显示 -->
            <span class="token-tag">{{ labelMap[token.tag] || token.tag }}</span>
          </div>
        </template>
        <div v-else class="empty-tokens">
          点击「分词」按钮开始处理
        </div>
      </div>
    </div>

    <!-- 标签图例（始终显示） -->
    <div class="legend">
      <span v-for="(label, idx) in labels" :key="label" class="legend-item" :class="`tag-${label}`">
        <kbd>{{ idx }}</kbd> {{ labelMap[label] }}
      </span>
    </div>

    <!-- 右键菜单 -->
    <div 
      v-if="contextMenu.visible" 
      class="context-menu"
      :style="{ left: contextMenu.x + 'px', top: contextMenu.y + 'px' }"
    >
      <div class="context-menu-item" @click="insertTokenBefore">在前面插入</div>
      <div class="context-menu-item" @click="insertTokenAfter">在后面插入</div>
      <div class="context-menu-item" @click="contextMergeNext" :class="{ disabled: contextMenu.tokenIndex >= localTokens.length - 1 }">合并后一个 [M]</div>
      <div class="context-menu-item" @click="contextMergePrev" :class="{ disabled: contextMenu.tokenIndex <= 0 }">合并前一个 [[]</div>
      <div class="context-menu-item danger" @click="deleteToken">删除当前</div>
    </div>

    <!-- 点击其他地方关闭菜单 -->
    <div v-if="contextMenu.visible" class="context-menu-overlay" @click="closeContextMenu"></div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted, onUnmounted, nextTick } from 'vue'

const props = defineProps({
  text: String,
  tokens: { type: Array, default: () => [] },
  tokenizerModel: String,
  isLoading: Boolean,
  currentIndex: Number,
  total: Number,
  typeClass: String,
  typeEvidence: String,
  platform: { type: String, default: 'pipe' }
})

const emit = defineEmits(['tokenize', 'update-tokens', 'model-change', 'navigate', 'update-type-class', 'update-type-evidence', 'clear-tokens'])

const labels = ['O', 'TYPE', 'MATERIAL', 'SIZE', 'THICKNESS', 'PRESSURE', 'STANDARD', 'STANDARD_GRADE', 'CONN', 'MANU']

// 标签中英文映射
const labelMap = {
  'TYPE': '种类',
  'MATERIAL': '材质',
  'SIZE': '尺寸',
  'THICKNESS': '壁厚',
  'PRESSURE': '磅级',
  'STANDARD': '规范',
  'STANDARD_GRADE': '规范等级',
  'CONN': '连接',
  'MANU': '工艺',
  'O': '其他'
}

const typeOptions = ['管子', '管件', '阀门', '法兰', '垫片', '螺栓']
const models = [
  { value: 'deepseek-chat', label: 'DeepSeek API' },
  { value: 'ner_pipe', label: 'NER模型' },
  { value: 'qwen3:8b', label: 'Qwen3 8B' }
]

const editorRef = ref(null)
const editInputRef = ref(null)
const localModel = ref(props.tokenizerModel)
const localTokens = ref([])
const localTypeClass = ref('')
const localTypeEvidence = ref('')

// 选中状态
const selectedIndex = ref(-1)

// 编辑状态
const editingIndex = ref(-1)
const editingValue = ref('')

// 右键菜单
const contextMenu = ref({
  visible: false,
  x: 0,
  y: 0,
  tokenIndex: -1
})

// 选择依据模式
const isSelectingEvidence = ref(false)
const pendingTypeClass = ref('')

watch(() => props.tokens, (newTokens, oldTokens) => {
  localTokens.value = JSON.parse(JSON.stringify(newTokens))
  // 只有当 tokens 完全替换时才重置选中状态（如重新分词）
  // 如果只是标签变化（长度相同），保持选中状态
  const isFullReplace = !oldTokens || oldTokens.length !== newTokens.length
  if (isFullReplace) {
    selectedIndex.value = -1
    editingIndex.value = -1
  }
}, { immediate: true, deep: true })

watch(() => props.typeClass, (v) => { localTypeClass.value = v }, { immediate: true })
watch(() => props.typeEvidence, (v) => { localTypeEvidence.value = v }, { immediate: true })

// 全局键盘事件监听
function handleGlobalKeydown(e) {
  // 如果焦点在输入框内，不处理
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
    return
  }
  
  // 如果正在编辑词块，不处理
  if (editingIndex.value >= 0) return
  
  // 左右键切换记录
  if (e.key === 'ArrowLeft') {
    e.preventDefault()
    emit('navigate', -1)
    return
  }
  if (e.key === 'ArrowRight') {
    e.preventDefault()
    emit('navigate', 1)
    return
  }
  
  // Tab 键切换词块
  if (e.key === 'Tab' && localTokens.value.length > 0) {
    e.preventDefault()
    if (selectedIndex.value < 0) {
      selectedIndex.value = 0
    } else if (e.shiftKey) {
      selectedIndex.value = selectedIndex.value > 0 ? selectedIndex.value - 1 : localTokens.value.length - 1
    } else {
      selectedIndex.value = selectedIndex.value < localTokens.value.length - 1 ? selectedIndex.value + 1 : 0
    }
    return
  }
  
  // 数字键 0-9 选择标签
  if (selectedIndex.value >= 0 && e.key >= '0' && e.key <= '9') {
    e.preventDefault()
    const labelIdx = parseInt(e.key)
    if (labelIdx < labels.length) {
      localTokens.value[selectedIndex.value].tag = labels[labelIdx]
      emit('update-tokens', localTokens.value)
    }
    return
  }
  
  // M 键或 ] 键：合并当前词块和后一个词块
  if (selectedIndex.value >= 0 && (e.key === 'm' || e.key === 'M' || e.key === ']')) {
    e.preventDefault()
    mergeWithNext()
    return
  }
  
  // [ 键：合并当前词块和前一个词块
  if (selectedIndex.value >= 0 && e.key === '[') {
    e.preventDefault()
    mergeWithPrev()
    return
  }
}

// 编辑器内键盘事件（保留以防止事件冒泡问题）
function handleKeydown(e) {
  // 键盘事件已移至全局处理
}

// Token 相关
function isSpace(word) {
  return !word.trim()
}

function displayWord(word) {
  if (!word.trim()) return '·'
  return word
}

function selectToken(index) {
  selectedIndex.value = index
  // 聚焦编辑器以接收键盘事件
  editorRef.value?.focus()
}

function startEdit(index) {
  editingIndex.value = index
  editingValue.value = localTokens.value[index].word
  nextTick(() => {
    const inputs = editInputRef.value
    if (inputs && inputs.length > 0) {
      inputs[0].focus()
      inputs[0].select()
    }
  })
}

function finishEdit() {
  if (editingIndex.value >= 0 && editingValue.value) {
    localTokens.value[editingIndex.value].word = editingValue.value
    emit('update-tokens', localTokens.value)
  }
  editingIndex.value = -1
  editingValue.value = ''
}

function cancelEdit() {
  editingIndex.value = -1
  editingValue.value = ''
}

// 右键菜单
function showContextMenu(e, index) {
  contextMenu.value = {
    visible: true,
    x: e.clientX,
    y: e.clientY,
    tokenIndex: index
  }
}

function closeContextMenu() {
  contextMenu.value.visible = false
}

function insertTokenBefore() {
  const idx = contextMenu.value.tokenIndex
  localTokens.value.splice(idx, 0, { word: '新词', tag: 'O' })
  emit('update-tokens', localTokens.value)
  closeContextMenu()
  // 自动进入编辑模式
  nextTick(() => startEdit(idx))
}

function insertTokenAfter() {
  const idx = contextMenu.value.tokenIndex
  localTokens.value.splice(idx + 1, 0, { word: '新词', tag: 'O' })
  emit('update-tokens', localTokens.value)
  closeContextMenu()
  // 自动进入编辑模式
  nextTick(() => startEdit(idx + 1))
}

function deleteToken() {
  const idx = contextMenu.value.tokenIndex
  localTokens.value.splice(idx, 1)
  emit('update-tokens', localTokens.value)
  closeContextMenu()
  selectedIndex.value = -1
}

// 合并当前词块和后一个词块
function mergeWithNext() {
  const idx = selectedIndex.value
  if (idx < 0 || idx >= localTokens.value.length - 1) return
  
  const current = localTokens.value[idx]
  const next = localTokens.value[idx + 1]
  
  // 合并词语，保留当前词块的标签
  current.word = current.word + next.word
  
  // 删除后一个词块
  localTokens.value.splice(idx + 1, 1)
  
  emit('update-tokens', localTokens.value)
}

// 合并当前词块和前一个词块
function mergeWithPrev() {
  const idx = selectedIndex.value
  if (idx <= 0) return
  
  const prev = localTokens.value[idx - 1]
  const current = localTokens.value[idx]
  
  // 合并到前一个词块，保留前一个词块的标签
  prev.word = prev.word + current.word
  
  // 删除当前词块
  localTokens.value.splice(idx, 1)
  
  // 选中合并后的词块
  selectedIndex.value = idx - 1
  
  emit('update-tokens', localTokens.value)
}

// 右键菜单合并操作
function contextMergeNext() {
  const idx = contextMenu.value.tokenIndex
  if (idx >= localTokens.value.length - 1) return
  
  selectedIndex.value = idx
  mergeWithNext()
  closeContextMenu()
}

function contextMergePrev() {
  const idx = contextMenu.value.tokenIndex
  if (idx <= 0) return
  
  selectedIndex.value = idx
  mergeWithPrev()
  closeContextMenu()
}

// 材料大类依据选择
function startSelectEvidence(type) {
  // 如果已经是选择模式且是同一个类型，取消选择模式
  if (isSelectingEvidence.value && pendingTypeClass.value === type) {
    isSelectingEvidence.value = false
    pendingTypeClass.value = ''
    return
  }
  
  isSelectingEvidence.value = true
  pendingTypeClass.value = type
}

function handleTextSelection() {
  if (!isSelectingEvidence.value) return
  
  const selection = window.getSelection()
  const selectedText = selection.toString().trim()
  
  if (selectedText) {
    localTypeClass.value = pendingTypeClass.value
    localTypeEvidence.value = selectedText
    emit('update-type-class', pendingTypeClass.value)
    emit('update-type-evidence', selectedText)
  }
  
  isSelectingEvidence.value = false
  pendingTypeClass.value = ''
}

function clearEvidence() {
  localTypeEvidence.value = ''
  localTypeClass.value = ''
  emit('update-type-evidence', '')
  emit('update-type-class', '')
}

// 点击外部关闭菜单
function handleClickOutside(e) {
  if (contextMenu.value.visible && !e.target.closest('.context-menu')) {
    closeContextMenu()
  }
}

onMounted(() => {
  document.addEventListener('click', handleClickOutside)
  // 全局监听左右键
  document.addEventListener('keydown', handleGlobalKeydown)
  // 自动聚焦以接收键盘事件
  editorRef.value?.focus()
})

onUnmounted(() => {
  document.removeEventListener('click', handleClickOutside)
  document.removeEventListener('keydown', handleGlobalKeydown)
})
</script>

<style scoped>
.annotation-editor {
  background: var(--bg-primary);
  border-radius: 8px;
  border: 1px solid var(--border-color);
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 20px;
  outline: none;
}

.editor-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 16px;
  border-bottom: 1px solid var(--border-light);
}

.toolbar-left, .toolbar-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.nav-info {
  font-size: 13px;
  color: var(--text-secondary);
  min-width: 80px;
  text-align: center;
}

.section-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 8px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.label-hint {
  font-weight: 400;
  font-size: 11px;
  color: var(--text-muted);
  text-transform: none;
}

.selecting-hint {
  font-weight: 400;
  font-size: 11px;
  color: var(--primary);
  text-transform: none;
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.source-text {
  padding: 12px 16px;
  background: var(--bg-secondary);
  border-radius: 6px;
  font-size: 14px;
  line-height: 1.6;
  color: var(--text-primary);
  user-select: text;
}

.source-text.selecting-mode {
  cursor: text;
  background: var(--bg-tertiary);
  border: 2px dashed var(--primary);
}

.type-section {
  background: var(--bg-secondary);
  padding: 16px;
  border-radius: 6px;
}

.type-buttons {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.type-btn {
  padding: 6px 16px;
  border: 1px solid var(--border-color);
  background: var(--bg-primary);
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.2s;
}

.type-btn:hover {
  border-color: var(--primary);
}

.type-btn.active {
  background: var(--primary);
  color: white;
  border-color: var(--primary);
}

.type-btn.selecting {
  border-color: var(--primary);
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.3);
  animation: pulse 1s infinite;
}

.type-evidence {
  margin-top: 12px;
  font-size: 13px;
  color: var(--text-secondary);
  display: flex;
  align-items: center;
  gap: 8px;
}

.type-evidence span {
  color: var(--primary);
  font-weight: 500;
}

.tokens-container {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  padding: 16px;
  background: var(--bg-secondary);
  border-radius: 6px;
  min-height: 100px;
}

.empty-tokens {
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
  font-size: 14px;
}

.token {
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 6px 10px;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.15s;
  user-select: none;
  border: 2px solid transparent;
  position: relative;
}

.token:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-sm);
}

.token.selected {
  border-color: var(--primary);
  box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
}

.token.is-space {
  min-width: 24px;
}

.token-edit-input {
  width: 100%;
  min-width: 50px;
  max-width: 150px;
  padding: 2px 4px;
  border: 1px solid var(--primary);
  border-radius: 3px;
  font-size: 14px;
  font-weight: 500;
  outline: none;
  background: white;
  text-align: center;
  margin-bottom: 2px;
}

.token-word {
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 2px;
}

.token.editing {
  border-color: var(--primary);
}

.token-tag {
  font-size: 10px;
  opacity: 0.8;
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding-top: 12px;
  border-top: 1px solid var(--border-light);
}

.legend-item {
  padding: 2px 8px;
  border-radius: 3px;
  font-size: 11px;
  display: flex;
  align-items: center;
  gap: 4px;
}

.legend-item kbd {
  font-family: monospace;
  font-size: 10px;
  padding: 1px 4px;
  background: rgba(0,0,0,0.1);
  border-radius: 2px;
}

/* 右键菜单 */
.context-menu {
  position: fixed;
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 6px;
  box-shadow: var(--shadow-lg);
  z-index: 1000;
  min-width: 120px;
  overflow: hidden;
}

.context-menu-item {
  padding: 8px 16px;
  cursor: pointer;
  font-size: 13px;
  transition: background 0.15s;
}

.context-menu-item:hover {
  background: var(--bg-secondary);
}

.context-menu-item.danger {
  color: var(--danger);
}

.context-menu-item.danger:hover {
  background: rgba(239, 68, 68, 0.1);
}

.context-menu-item.disabled {
  color: var(--text-muted);
  cursor: not-allowed;
}

.context-menu-item.disabled:hover {
  background: transparent;
}

.context-menu-overlay {
  position: fixed;
  inset: 0;
  z-index: 999;
}
</style>
