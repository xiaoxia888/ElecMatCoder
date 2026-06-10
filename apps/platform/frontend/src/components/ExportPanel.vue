<template>
  <div class="export-panel">
    <div class="panel-header">
      <span class="panel-title">导出数据</span>
    </div>
    
    <div class="export-options">
      <!-- 标注数据导出 -->
      <div class="export-group">
        <div class="group-title">标注数据</div>
        <div class="export-buttons">
          <button class="btn btn-secondary btn-sm" @click="exportBIO" :disabled="!hasAnnotations">
            导出 BIO
          </button>
          <button class="btn btn-secondary btn-sm" @click="exportJSONL" :disabled="!hasAnnotations">
            导出 JSONL
          </button>
        </div>
      </div>
      
      <!-- 编码数据导出 -->
      <div class="export-group" v-if="hasEncodings">
        <div class="group-title">编码数据</div>
        <div class="export-buttons">
          <button class="btn btn-secondary btn-sm" @click="exportEncodingCSV">
            导出 CSV
          </button>
          <button class="btn btn-secondary btn-sm" @click="exportEncodingExcel">
            导出 Excel
          </button>
        </div>
      </div>
      
      <!-- 进度保存 -->
      <div class="export-group">
        <div class="group-title">进度管理</div>
        <div class="export-buttons">
          <button class="btn btn-primary btn-sm" @click="$emit('save-progress')">
            保存进度
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed, inject } from 'vue'
import { getDifficultyLabel } from '../utils/difficulty'

const props = defineProps({
  annotations: { type: Object, default: () => ({}) },
  encodings: { type: Object, default: () => ({}) },
  dataList: { type: Array, default: () => [] }
})

const emit = defineEmits(['save-progress'])
const showToast = inject('showToast')

const hasAnnotations = computed(() => Object.keys(props.annotations).length > 0)
const hasEncodings = computed(() => Object.keys(props.encodings).length > 0)

function exportBIO() {
  let bio = ''
  
  Object.entries(props.annotations).forEach(([index, ann]) => {
    if (!ann.tokens || ann.tokens.length === 0) return
    
    ann.tokens.forEach(token => {
      const word = token.word
      const tag = token.tag || 'O'
      
      if (tag === 'O') {
        for (const char of word) {
          bio += `${char}\t${tag}\n`
        }
      } else {
        for (let i = 0; i < word.length; i++) {
          const prefix = i === 0 ? 'B-' : 'I-'
          bio += `${word[i]}\t${prefix}${tag}\n`
        }
      }
    })
    bio += '\n'
  })
  
  downloadFile(bio, `annotations_${getDateStr()}.bio`, 'text/plain')
  showToast('BIO文件导出成功', 'success')
}

function exportJSONL() {
  let jsonl = ''
  
  Object.entries(props.annotations).forEach(([index, ann]) => {
    if (!ann.tokens || ann.tokens.length === 0) return
    
    const nerLabels = []
    ann.tokens.forEach(token => {
      const tag = token.tag || 'O'
      for (let i = 0; i < token.word.length; i++) {
        if (tag === 'O') {
          nerLabels.push('O')
        } else {
          nerLabels.push(i === 0 ? `B-${tag}` : `I-${tag}`)
        }
      }
    })
    
    const item = {
      text: ann.text || props.dataList[index]?.text || '',
      ner_labels: nerLabels,
      type_class: ann.typeClass || '',
      type_entity: ann.typeEvidence || ''
    }
    
    jsonl += JSON.stringify(item) + '\n'
  })
  
  downloadFile(jsonl, `annotations_${getDateStr()}.jsonl`, 'application/jsonl')
  showToast('JSONL文件导出成功', 'success')
}

function exportEncodingCSV() {
  const headers = ['序号', '原始描述', '编码结果', '是否需审核', '总置信度', '最低相似度', '分流难度', '二次分流最终难度', 'TYPE', 'MANU', 'CONN', 'SIZE', 'THICKNESS', 'PRESSURE', 'MATERIAL', 'STANDARD']
  let csv = headers.join(',') + '\n'
  
  Object.entries(props.encodings).forEach(([index, enc]) => {
    const fields = enc.fields || {}
    const row = [
      parseInt(index) + 1,
      `"${(enc.original_text || '').replace(/"/g, '""')}"`,
      enc.final_code || '',
      enc.need_review ? '是' : '否',
      enc.confidence ? (enc.confidence * 100).toFixed(2) + '%' : '',
      enc.min_similarity ? (enc.min_similarity * 100).toFixed(2) + '%' : '',
      getDifficultyLabel(enc.difficulty_split?.difficulty),
      getDifficultyLabel(enc.second_pass?.final_level),
      fields.TYPE?.code || '',
      fields.MANU?.code || '',
      fields.CONN?.code || '',
      fields.SIZE?.code || '',
      fields.THICKNESS?.code || '',
      fields.PRESSURE?.code || '',
      fields.MATERIAL?.code || '',
      fields.STANDARD?.code || ''
    ]
    csv += row.join(',') + '\n'
  })
  
  downloadFile('\ufeff' + csv, `encodings_${getDateStr()}.csv`, 'text/csv;charset=utf-8')
  showToast('CSV文件导出成功', 'success')
}

function exportEncodingExcel() {
  // 简单实现：导出CSV，用户可以用Excel打开
  exportEncodingCSV()
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
}

.panel-header {
  margin-bottom: 16px;
}

.panel-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}

.export-options {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.export-group {
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border-light);
}

.export-group:last-child {
  border-bottom: none;
  padding-bottom: 0;
}

.group-title {
  font-size: 12px;
  color: var(--text-secondary);
  margin-bottom: 8px;
}

.export-buttons {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}
</style>
