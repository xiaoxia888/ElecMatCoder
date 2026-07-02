<template>
  <div class="data-import">
    <div class="import-section">
      <!-- 上传区域 -->
      <div 
        class="upload-dropzone"
        :class="{ dragover: isDragover }"
        @dragover.prevent="isDragover = true"
        @dragleave="isDragover = false"
        @drop.prevent="handleDrop"
        @click="triggerFileInput"
      >
        <div class="upload-dropzone-icon">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
            <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
          </svg>
        </div>
        <div class="upload-dropzone-text">
          拖拽 Excel 文件到此处，或 <em>点击上传</em>
        </div>
        <div class="upload-dropzone-hint">支持 .xlsx / .xls 格式</div>
      </div>
      <input 
        ref="fileInput"
        type="file" 
        accept=".xlsx,.xls"
        style="display: none"
        @change="handleFileSelect"
      >

      <!-- 文件信息 -->
      <div v-if="fileName" class="file-info">
        <span class="file-name">{{ fileName }}</span>
        <button class="btn btn-sm btn-ghost" @click="clearFile">×</button>
      </div>

      <!-- 列选择 -->
      <div v-if="columns.length > 0" class="column-select">
        <div class="form-group">
          <label class="form-label">选择列</label>
          <select 
            class="select" 
            style="width: 100%"
            v-model="selectedColumn"
          >
            <option value="" disabled>请选择要处理的列</option>
            <option v-for="col in columns" :key="col" :value="col">
              {{ col }}
            </option>
          </select>
        </div>
      </div>

      <!-- 确认导入 -->
      <button 
        v-if="selectedColumn"
        class="btn btn-primary" 
        style="width: 100%; margin-top: 12px"
        @click="confirmImport"
      >
        确认导入
      </button>
    </div>

    <!-- 或者加载进度文件（仅标注平台显示） -->
    <template v-if="showProgress">
      <div class="import-divider">
        <span>或</span>
      </div>

      <div class="import-section">
        <button class="btn btn-secondary" style="width: 100%" @click="loadProgress">
          加载进度文件
        </button>
        <input 
          ref="progressInput"
          type="file" 
          accept=".json"
          style="display: none"
          @change="handleProgressLoad"
        >
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref, inject } from 'vue'

const props = defineProps({
  showProgress: { type: Boolean, default: true }  // 是否显示加载进度按钮
})

const emit = defineEmits(['data-loaded', 'progress-loaded'])
const showToast = inject('showToast')

const fileInput = ref(null)
const progressInput = ref(null)
const isDragover = ref(false)
const fileName = ref('')
const columns = ref([])
const selectedColumn = ref('')
const fileData = ref([])

function triggerFileInput() {
  fileInput.value?.click()
}

function handleDrop(e) {
  isDragover.value = false
  const file = e.dataTransfer.files[0]
  if (file) {
    processFile(file)
  }
}

function handleFileSelect(e) {
  const file = e.target.files[0]
  if (file) {
    processFile(file)
  }
}

async function processFile(file) {
  if (!file.name.match(/\.(xlsx|xls)$/i)) {
    showToast('请上传 Excel 文件', 'error')
    return
  }

  fileName.value = file.name
  
  try {
    // 动态导入 xlsx
    const XLSX = await import('xlsx')
    const data = await file.arrayBuffer()
    const workbook = XLSX.read(data)
    const firstSheet = workbook.Sheets[workbook.SheetNames[0]]
    const jsonData = XLSX.utils.sheet_to_json(firstSheet)
    
    if (jsonData.length === 0) {
      showToast('文件为空', 'error')
      return
    }

    fileData.value = jsonData
    columns.value = Object.keys(jsonData[0])
    
    // 自动选择包含"描述"的列
    const descCol = columns.value.find(c => c.includes('描述'))
    if (descCol) {
      selectedColumn.value = descCol
    }

    showToast(`已读取 ${jsonData.length} 条数据`, 'success')
  } catch (err) {
    showToast('文件解析失败: ' + err.message, 'error')
  }
}

function clearFile() {
  fileName.value = ''
  columns.value = []
  selectedColumn.value = ''
  fileData.value = []
  if (fileInput.value) {
    fileInput.value.value = ''
  }
}

function confirmImport() {
  if (!selectedColumn.value || fileData.value.length === 0) return
  
  emit('data-loaded', {
    data: fileData.value,
    column: selectedColumn.value
  })
}

function loadProgress() {
  progressInput.value?.click()
}

function handleProgressLoad(e) {
  const file = e.target.files[0]
  if (!file) return

  const reader = new FileReader()
  reader.onload = (event) => {
    try {
      const data = JSON.parse(event.target.result)
      emit('progress-loaded', data)
    } catch (err) {
      showToast('进度文件格式错误', 'error')
    }
  }
  reader.readAsText(file)
}
</script>

<style scoped>
.data-import {
  padding: 16px;
}

.import-section {
  margin-bottom: 16px;
}

.file-info {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 12px;
  background: var(--bg-tertiary);
  border-radius: 4px;
  margin-top: 12px;
}

.file-name {
  font-size: 13px;
  color: var(--text-primary);
}

.column-select {
  margin-top: 16px;
}

.import-divider {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 20px 0;
  color: var(--text-muted);
  font-size: 12px;
}

.import-divider::before,
.import-divider::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border-color);
}
</style>
