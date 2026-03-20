<template>
  <div class="annotation-view">
    <!-- 左侧边栏 -->
    <aside class="sidebar">
      <DataImport 
        @data-loaded="handleDataLoaded"
        @progress-loaded="handleProgressLoaded"
      />
      
      <div class="sidebar-section" v-if="dataList.length > 0">
        <div class="section-header">
          <span class="section-title">数据列表</span>
          <button class="btn btn-sm btn-primary" @click="handleBatchTokenize" :disabled="isBatchTokenizing">
            {{ isBatchTokenizing ? '处理中...' : '批量分词' }}
          </button>
        </div>
        
        <div v-if="isBatchTokenizing" class="progress-bar" style="margin-bottom: 12px">
          <div class="progress-fill" :style="{ width: batchProgress + '%' }"></div>
        </div>
        
        <DataList 
          :items="dataList"
          :current-index="currentIndex"
          :annotations="annotations"
          @select="currentIndex = $event"
        />
      </div>
    </aside>

    <!-- 右侧内容区 -->
    <section class="content">
      <AnnotationEditor
        v-if="currentIndex >= 0"
        :text="currentText"
        :tokens="currentTokens"
        :tokenizer-model="tokenizerModel"
        :is-loading="isTokenizing"
        :current-index="currentIndex"
        :total="dataList.length"
        :type-class="currentTypeClass"
        :type-evidence="currentTypeEvidence"
        @tokenize="handleTokenize"
        @update-tokens="handleUpdateTokens"
        @model-change="tokenizerModel = $event"
        @navigate="handleNavigate"
        @update-type-class="handleUpdateTypeClass"
        @update-type-evidence="handleUpdateTypeEvidence"
        @clear-tokens="handleClearTokens"
      />
      
      <div v-else class="empty-state">
        <div class="empty-state-text">请导入数据并选择一条记录开始标注</div>
      </div>
      
      <ExportPanel
        v-if="dataList.length > 0"
        :annotations="annotations"
        :data-list="dataList"
        @save-progress="handleSaveProgress"
      />
    </section>
  </div>
</template>

<script setup>
import { ref, computed, inject } from 'vue'
import axios from 'axios'
import DataImport from '../components/DataImport.vue'
import DataList from '../components/DataList.vue'
import AnnotationEditor from '../components/AnnotationEditor.vue'
import ExportPanel from '../components/ExportPanel.vue'

const showToast = inject('showToast')

// 状态（独立的数据）
const dataList = ref([])
const currentIndex = ref(-1)
const annotations = ref({})
const tokenizerModel = ref('ner_pipe')
const isTokenizing = ref(false)
const isBatchTokenizing = ref(false)
const batchProgress = ref(0)

// 计算属性
const currentText = computed(() => {
  return dataList.value[currentIndex.value]?.text || ''
})

const currentTokens = computed(() => {
  return annotations.value[currentIndex.value]?.tokens || []
})

const currentTypeClass = computed(() => {
  return annotations.value[currentIndex.value]?.typeClass || ''
})

const currentTypeEvidence = computed(() => {
  return annotations.value[currentIndex.value]?.typeEvidence || ''
})

// 数据加载
function handleDataLoaded({ data, column }) {
  dataList.value = data.map((row, index) => ({
    index,
    text: row[column] || ''
  })).filter(item => item.text)
  
  currentIndex.value = -1
  annotations.value = {}
  
  showToast(`成功导入 ${dataList.value.length} 条数据`, 'success')
  
  if (dataList.value.length > 0) {
    currentIndex.value = 0
  }
  
}

function handleProgressLoaded(data) {
  if (data.dataList) dataList.value = data.dataList
  if (data.annotations) annotations.value = data.annotations
  currentIndex.value = data.currentIndex ?? 0
  tokenizerModel.value = data.tokenizerModel ?? 'deepseek-chat'
  
  showToast(`进度已加载（${Object.keys(data.annotations || {}).length}条已标注）`, 'success')
}

// 分词
async function handleTokenize() {
  if (currentIndex.value < 0) return

  const text = dataList.value[currentIndex.value].text
  isTokenizing.value = true
  
  try {
    const response = await axios.post('/api/tokenize', {
      text,
      preprocess: true,
      model: tokenizerModel.value,
      platform: 'pipe'
    })

    if (response.data.success) {
      const autoEvidence = response.data.tokens?.find(t => t.tag === 'TYPE')?.word || ''
      
      annotations.value[currentIndex.value] = {
        ...annotations.value[currentIndex.value],
        text: response.data.processed_text,
        tokens: response.data.tokens,
        typeClass: response.data.type_class || '',
        typeEvidence: response.data.type_evidence || autoEvidence
      }
      showToast('分词完成', 'success')
    }
  } catch (err) {
    showToast(`分词失败: ${err.message}`, 'error')
  } finally {
    isTokenizing.value = false
  }
}

function handleUpdateTokens(tokens) {
  if (currentIndex.value >= 0 && annotations.value[currentIndex.value]) {
    annotations.value[currentIndex.value].tokens = tokens
  }
}

function handleUpdateTypeClass(typeClass) {
  if (currentIndex.value >= 0) {
    if (!annotations.value[currentIndex.value]) {
      annotations.value[currentIndex.value] = { text: currentText.value, tokens: [], typeClass: '', typeEvidence: '' }
    }
    annotations.value[currentIndex.value].typeClass = typeClass
  }
}

function handleUpdateTypeEvidence(typeEvidence) {
  if (currentIndex.value >= 0) {
    if (!annotations.value[currentIndex.value]) {
      annotations.value[currentIndex.value] = { text: currentText.value, tokens: [], typeClass: '', typeEvidence: '' }
    }
    annotations.value[currentIndex.value].typeEvidence = typeEvidence
  }
}

function handleNavigate(delta) {
  const newIndex = currentIndex.value + delta
  if (newIndex >= 0 && newIndex < dataList.value.length) {
    currentIndex.value = newIndex
  }
}

// 批量分词（跳过已分词的）
async function handleBatchTokenize() {
  if (dataList.value.length === 0) return

  // 筛选出未分词的记录
  const needTokenize = dataList.value
    .map((item, idx) => ({ ...item, originalIndex: idx }))
    .filter(item => {
      const ann = annotations.value[item.originalIndex]
      return !ann || !ann.tokens || ann.tokens.length === 0
    })
  
  if (needTokenize.length === 0) {
    showToast('所有记录已分词，无需重复处理', 'info')
    return
  }

  isBatchTokenizing.value = true
  batchProgress.value = 0
  
  const total = needTokenize.length
  const batchSize = 10
  
  for (let i = 0; i < total; i += batchSize) {
    const batch = needTokenize.slice(i, Math.min(i + batchSize, total))
    const texts = batch.map(item => item.text)
    
    try {
      const response = await axios.post('/api/tokenize/batch', {
        texts,
        preprocess: true,
        model: tokenizerModel.value,
        platform: 'pipe'
      })
      
      if (response.data.results) {
        response.data.results.forEach((result, idx) => {
          const actualIndex = batch[idx].originalIndex
          if (result.success) {
            const autoEvidence = result.tokens?.find(t => t.tag === 'TYPE')?.word || ''
            
            annotations.value[actualIndex] = {
              ...(annotations.value[actualIndex] || {}),
              text: result.processed_text,
              tokens: result.tokens,
              typeClass: result.type_class || '',
              typeEvidence: result.type_evidence || autoEvidence
            }
          }
        })
      }
    } catch (err) {
      console.error('批量分词错误:', err)
    }
    
    batchProgress.value = Math.round(((i + batch.length) / total) * 100)
  }
  
  isBatchTokenizing.value = false
  showToast(`批量分词完成，处理了 ${total} 条记录`, 'success')
  
  if (currentIndex.value < 0 && dataList.value.length > 0) {
    currentIndex.value = 0
  }
}

// 清除当前记录的标签
function handleClearTokens() {
  if (currentIndex.value >= 0 && annotations.value[currentIndex.value]) {
    annotations.value[currentIndex.value].tokens = []
    annotations.value[currentIndex.value].typeClass = ''
    annotations.value[currentIndex.value].typeEvidence = ''
    showToast('已清除当前记录的标签', 'success')
  }
}

// 保存进度
function handleSaveProgress() {
  const progressData = {
    version: '1.0',
    savedAt: new Date().toISOString(),
    dataList: dataList.value,
    annotations: annotations.value,
    currentIndex: currentIndex.value,
    tokenizerModel: tokenizerModel.value
  }
  
  const blob = new Blob([JSON.stringify(progressData, null, 2)], { type: 'application/json' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `progress_${new Date().toISOString().slice(0, 10)}.json`
  a.click()
  URL.revokeObjectURL(url)
  
  showToast('进度已保存', 'success')
}
</script>

<style scoped>
.annotation-view {
  display: flex;
  height: 100%;
  overflow: hidden;
}

.sidebar {
  width: 320px;
  background: var(--bg-primary);
  border-right: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.sidebar-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding: 16px;
  min-height: 0;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.section-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.content {
  flex: 1;
  padding: 24px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  overflow-y: auto;
  min-height: 0;
  max-width: 100%;
}

.content > :deep(.annotation-editor) {
  flex: 1;
}

.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

.empty-state-icon {
  font-size: 48px;
  margin-bottom: 16px;
  opacity: 0.5;
}
</style>
