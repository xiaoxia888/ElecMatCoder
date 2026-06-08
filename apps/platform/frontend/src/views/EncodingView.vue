<template>
  <div class="encoding-view">
    <!-- 左侧边栏 -->
    <aside class="sidebar">
      <!-- 数据导入（可折叠） -->
      <div class="sidebar-section import-section" :class="{ collapsed: dataList.length > 0 && isImportCollapsed }">
        <div class="section-header clickable" @click="isImportCollapsed = !isImportCollapsed" v-if="dataList.length > 0">
          <span class="section-title">数据导入</span>
          <span class="collapse-icon">{{ isImportCollapsed ? '▼' : '▲' }}</span>
        </div>
        <div class="section-title" v-else>数据导入</div>
        
        <div class="import-content" v-show="!isImportCollapsed || dataList.length === 0">
          <DataImport 
            :show-progress="false"
            @data-loaded="handleDataLoaded"
          />
        </div>
        
        <!-- 已导入提示 -->
        <div v-if="dataList.length > 0 && isImportCollapsed" class="import-summary">
          已导入 {{ dataList.length }} 条数据
        </div>
      </div>
      
      <!-- 编码操作 -->
      <div class="sidebar-section operation-section" v-if="dataList.length > 0">
        <button 
          class="btn"
          :class="batchActionButtonClass"
          style="width: 100%"
          :disabled="(isEncoding && !activeBatchJobId) || isStopSubmitting"
          @click="handlePrimaryBatchAction"
        >
          {{ batchActionText }}
        </button>

        <div class="concurrency-row">
          <span class="concurrency-label">并发数</span>
          <input
            type="number"
            class="concurrency-input"
            :min="1"
            :max="16"
            :value="maxConcurrent"
            @change="handleMaxConcurrentChange"
            :disabled="isEncoding"
          />
          <span class="concurrency-hint">默认 {{ defaultMaxConcurrent }}</span>
        </div>
        
        <div v-if="isEncoding" class="progress-bar">
          <div class="progress-fill" :style="{ width: encodeProgress + '%' }"></div>
        </div>
        
        <!-- 统计信息 -->
        <div v-if="Object.keys(encodings).length > 0" class="stats-row">
          <span class="stat">总 <b>{{ Object.keys(encodings).length }}</b></span>
          <span class="stat success">成功 <b>{{ successCount }}</b></span>
          <span class="stat warning">待审 <b>{{ reviewCount }}</b></span>
        </div>
      </div>

      <div class="sidebar-section task-section" v-if="runningBatchJobs.length > 0">
        <div class="section-header">
          <span class="section-title">运行任务</span>
        </div>
        <div class="task-list">
          <div
            v-for="job in runningBatchJobs"
            :key="job.job_id"
            class="task-item"
            :class="{ active: activeBatchJobId === job.job_id }"
            @click="openBatchJob(job.job_id)"
          >
            <div class="task-item-main">
              <span class="task-item-title">任务 {{ shortJobId(job.job_id) }}</span>
              <span class="task-item-status">{{ taskStatusText(job.status) }}</span>
            </div>
            <div class="task-item-sub">
              <span>{{ job.processed || 0 }}/{{ job.total || 0 }}</span>
              <span v-if="job.queue_position">排队第 {{ job.queue_position }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- 数据列表（主要区域） -->
      <div class="sidebar-section list-section" v-if="dataList.length > 0">
        <div class="section-header">
          <span class="section-title">数据列表</span>
          <div class="filter-tabs">
            <button 
              class="filter-tab" 
              :class="{ active: filter === 'all' }"
              @click="filter = 'all'"
            >全部</button>
            <button 
              class="filter-tab" 
              :class="{ active: filter === 'review' }"
              @click="filter = 'review'"
            >待审</button>
          </div>
        </div>
        
        <div class="data-list" ref="dataListRef">
          <div 
            v-for="item in filteredDataList"
            :key="item.index"
            class="list-item"
            :class="{ 
              active: currentIndex === item.index,
              corrected: getItemStatus(item.index) === 'corrected',
              success: getItemStatus(item.index) === 'success',
              warning: getItemStatus(item.index) === 'review'
            }"
            :data-index="item.index"
            @click="handleSelectItem(item.index)"
          >
            <span class="item-num">#{{ item.index + 1 }}</span>
            <span class="item-text" :title="item.text">{{ item.text }}</span>
            <span
              v-if="getItemDifficulty(item.index)"
              class="item-difficulty"
              :class="getItemDifficultyClass(item.index)"
            >
              {{ getItemDifficulty(item.index) }}
            </span>
            <span class="item-badge corrected" v-if="getItemStatus(item.index) === 'corrected'">修正</span>
            <span class="item-badge" v-else-if="getItemStatus(item.index) === 'success'">OK</span>
            <span class="item-badge review" v-else-if="getItemStatus(item.index) === 'review'">!</span>
          </div>
        </div>
      </div>
    </aside>

    <!-- 右侧内容区 -->
    <section class="content">
      <template v-if="currentIndex >= 0">
        <!-- 原始描述卡片 -->
        <div class="source-card">
          <div class="card-header">
            <span class="card-title">原始描述</span>
            <div class="card-actions">
              <button 
                v-if="currentEncoding"
                class="btn btn-sm btn-outline"
                :disabled="isEncodingSingle"
                @click="handleReEncode"
              >
                {{ isEncodingSingle ? '编码中...' : '重新编码' }}
              </button>
              <div class="card-index-wrapper">
                <span class="card-index-hash">#</span>
                <input 
                  type="text"
                  class="page-input"
                  :value="currentIndex + 1"
                  @keydown.enter="handlePageJump($event)"
                  @blur="handlePageJump($event)"
                />
                <span class="card-index-total">/ {{ dataList.length }}</span>
              </div>
            </div>
          </div>
          <div class="source-text">{{ currentText }}</div>
          <div v-if="showProcessedText" class="processed-text-row">
            <span class="processed-text-label">格式化描述</span>
            <div class="processed-text">{{ currentProcessedText }}</div>
          </div>
        </div>
        
        <!-- 编码结果 -->
        <template v-if="currentEncoding">
          <EncodingResult 
            :result="currentEncoding"
            @select-candidate="handleSelectCandidate"
            @edit-field="handleEditField"
          />
        </template>
        <div v-else class="waiting-card">
          <button 
            class="btn btn-encode-single" 
            :disabled="isEncodingSingle"
            @click="handleEncodeSingle"
          >
            {{ isEncodingSingle ? '识别中...' : '识别当条' }}
          </button>
          <div class="waiting-hint">或点击左侧「一键编码」批量处理</div>
        </div>
        
        <!-- 导出面板 -->
        <EncodingExport
          v-if="Object.keys(encodings).length > 0"
          :encodings="encodings"
          :data-list="dataList"
        />

        <div
          v-if="showStopConfirmDialog"
          class="dialog-overlay"
          @click.self="closeStopConfirmDialog"
        >
          <div class="dialog-content stop-dialog">
            <div class="dialog-header">
              <span class="dialog-title">停止批量编码</span>
              <button
                v-if="!isStopSubmitting"
                class="dialog-close"
                @click="closeStopConfirmDialog"
              >&times;</button>
            </div>
            <div class="dialog-body">
              <div v-if="!isStopSubmitting" class="stop-dialog-text">
                当前任务 <b>{{ shortJobId(activeBatchJobId) }}</b> 正在{{ activeBatchJobStatus === 'queued' ? '排队' : '运行' }}。
                确认后将停止该任务。
              </div>
              <div v-else class="stop-dialog-loading">
                <span class="spinner"></span>
                <span>停止请求已提交，正在等待任务停止…</span>
              </div>
            </div>
            <div class="dialog-footer">
              <button
                v-if="!isStopSubmitting"
                class="btn btn-secondary btn-sm"
                @click="closeStopConfirmDialog"
              >取消</button>
              <button
                v-if="!isStopSubmitting"
                class="btn btn-danger btn-sm"
                @click="confirmCancelBatchJob"
              >确认停止</button>
            </div>
          </div>
        </div>

        <div v-if="showEditDialog" class="dialog-overlay" @click.self="closeEditDialog">
          <div class="dialog-content edit-dialog">
            <div class="dialog-header">
              <span class="dialog-title">{{ editDialogTitle }}</span>
              <button class="dialog-close" @click="closeEditDialog">&times;</button>
            </div>

            <div class="dialog-body">
              <template v-if="editDialogMode === 'single'">
                <div class="dialog-meta-grid">
                  <div class="form-group">
                    <label>原始内容</label>
                    <input class="form-input" :value="singleEditForm.originalContent" readonly />
                  </div>
                  <div class="form-group">
                    <label>原始编码</label>
                    <input class="form-input form-input-static" :value="singleEditForm.originalCode" readonly tabindex="-1" />
                  </div>
                </div>
                <div class="form-group">
                  <label>修改内容</label>
                  <input class="form-input" v-model="singleEditForm.modifiedContent" />
                </div>
                <div class="form-group">
                  <label>修改编码</label>
                  <input class="form-input" v-model="singleEditForm.modifiedCode" />
                </div>
              </template>

              <template v-else-if="editDialogMode === 'type'">
                <div class="dialog-meta-grid">
                  <div class="form-group">
                    <label>模型原始结构</label>
                    <input class="form-input" :value="typeEditForm.modelOriginalContent" readonly />
                  </div>
                  <div class="form-group">
                    <label>原始编码</label>
                    <input class="form-input form-input-static" :value="typeEditForm.originalCode" readonly tabindex="-1" />
                  </div>
                </div>
                <div class="form-group">
                  <label>一阶段最终结构</label>
                  <input class="form-input" :value="typeEditForm.stage1FinalContent" readonly />
                </div>
                <div class="form-group">
                  <label>实际编码输入</label>
                  <input class="form-input" :value="typeEditForm.encodingInput" readonly />
                </div>
                <div class="form-group">
                  <label>修改内容</label>
                  <input class="form-input" v-model="typeEditForm.modifiedContent" />
                </div>
                <div class="form-group">
                  <label>修改编码</label>
                  <input class="form-input" v-model="typeEditForm.modifiedCode" />
                </div>
              </template>

              <template v-else-if="editDialogMode === 'size'">
                <div class="dialog-meta-grid">
                  <div class="form-group">
                    <label>原始内容</label>
                    <input class="form-input" :value="sizeEditForm.originalContent" readonly />
                  </div>
                  <div class="form-group">
                    <label>原始编码</label>
                    <input class="form-input form-input-static" :value="sizeEditForm.originalCode" readonly tabindex="-1" />
                  </div>
                </div>
                <div class="structured-edit-grid structured-edit-grid-3">
                  <div class="form-group">
                    <label>DN（公称直径）</label>
                    <input class="form-input" v-model="sizeEditForm.dn" />
                    <div class="field-example">例如：`DN300 x DN150`</div>
                  </div>
                  <div class="form-group">
                    <label>OD（外径）</label>
                    <input class="form-input" v-model="sizeEditForm.od" />
                    <div class="field-example">例如：`323.9 x 168.3`</div>
                  </div>
                  <div class="form-group">
                    <label>INCH（英寸尺寸）</label>
                    <input class="form-input" v-model="sizeEditForm.inch" />
                    <div class="field-example">例如：`12 x 6` 或 `12&quot; x 6&quot;`</div>
                  </div>
                </div>
                <div class="form-hint">多个值请用 `x` 连接，例如 `DN300 x DN150`。</div>
                <div class="form-group">
                  <label>尺寸编码</label>
                  <input class="form-input" v-model="sizeEditForm.modifiedCode" />
                </div>
              </template>

              <template v-else-if="editDialogMode === 'thickness'">
                <div class="dialog-meta-grid">
                  <div class="form-group">
                    <label>原始内容</label>
                    <input class="form-input" :value="thicknessEditForm.originalContent" readonly />
                  </div>
                  <div class="form-group">
                    <label>原始编码</label>
                    <input class="form-input form-input-static" :value="thicknessEditForm.originalCode" readonly tabindex="-1" />
                  </div>
                </div>
                <div class="structured-edit-grid structured-edit-grid-3">
                  <div class="form-group">
                    <label>MM（毫米壁厚）</label>
                    <input class="form-input" v-model="thicknessEditForm.mm" />
                    <div class="field-example">例如：`3.2 x 4.5`</div>
                  </div>
                  <div class="form-group">
                    <label>INCH（英寸壁厚）</label>
                    <input class="form-input" v-model="thicknessEditForm.inch" />
                    <div class="field-example">例如：`0.5&quot; x 0.25&quot;`</div>
                  </div>
                  <div class="form-group">
                    <label>SCH（管表号）</label>
                    <input class="form-input" v-model="thicknessEditForm.schedule" />
                    <div class="field-example">例如：`SCH80 x SCH40`</div>
                  </div>
                  <div class="form-group">
                    <label>SERIES（壁厚系列）</label>
                    <input class="form-input" v-model="thicknessEditForm.series" />
                    <div class="field-example">例如：`XS x XXS`</div>
                  </div>
                  <div class="form-group">
                    <label>BWG（线规）</label>
                    <input class="form-input" v-model="thicknessEditForm.bwg" />
                    <div class="field-example">例如：`BWG12 x BWG10`</div>
                  </div>
                </div>
                <div class="form-hint">多个值请用 `x` 连接，例如 `SCH80 x SCH40`。</div>
                <div class="form-group">
                  <label>壁厚编码</label>
                  <input class="form-input" v-model="thicknessEditForm.modifiedCode" />
                </div>
              </template>

              <template v-else-if="editDialogMode === 'standard'">
                <div class="dialog-meta-grid">
                  <div class="form-group">
                    <label>原始内容</label>
                    <input class="form-input" :value="standardEditForm.originalContent" readonly />
                  </div>
                  <div class="form-group">
                    <label>原始编码</label>
                    <input class="form-input form-input-static" :value="standardEditForm.originalCode" readonly tabindex="-1" />
                  </div>
                </div>
                <div class="multi-edit-header">
                  <span>规范项</span>
                  <button class="btn btn-secondary btn-sm" @click="addStandardEditItem">新增规范项</button>
                </div>
                <div class="multi-edit-list">
                  <div v-for="(item, idx) in standardEditItems" :key="item.id" class="multi-edit-row">
                    <div class="multi-edit-row-header">
                      <span class="multi-edit-row-title">规范项 {{ idx + 1 }}</span>
                      <div class="multi-edit-row-actions">
                        <button class="btn btn-xs" :disabled="idx === 0" @click="moveStandardEditItem(idx, -1)">上移</button>
                        <button class="btn btn-xs" :disabled="idx === standardEditItems.length - 1" @click="moveStandardEditItem(idx, 1)">下移</button>
                        <button class="btn btn-xs btn-danger" @click="removeStandardEditItem(idx)">删除</button>
                      </div>
                    </div>
                    <div class="structured-edit-grid structured-edit-grid-4">
                      <div class="form-group">
                        <label>规范主体</label>
                        <input class="form-input" v-model="item.subject" />
                        <div class="field-example">例如：`GB/T3091`</div>
                      </div>
                      <div class="form-group">
                        <label>规范等级</label>
                        <input class="form-input" v-model="item.grade" />
                        <div class="field-example">例如：`CL1`、`Series I`、`Type E`</div>
                      </div>
                      <div class="form-group">
                        <label>规范方法</label>
                        <input class="form-input" v-model="item.method" />
                        <div class="field-example">例如：`Method E`、`Design A`</div>
                      </div>
                      <div class="form-group">
                        <label>规范附录</label>
                        <input class="form-input" v-model="item.appendix" />
                        <div class="field-example">例如：`Appendix A`、`附录B`</div>
                      </div>
                    </div>
                    <div class="form-group">
                      <label>规范编码</label>
                      <input class="form-input" v-model="item.code" />
                      <div class="field-example">例如：`GBT3091`</div>
                    </div>
                  </div>
                </div>
              </template>

            </div>

            <div class="dialog-footer">
              <button class="btn btn-secondary btn-sm" @click="closeEditDialog">取消</button>
              <button class="btn btn-primary btn-sm" @click="confirmEditDialog">确认</button>
            </div>
          </div>
        </div>
      </template>
      
      <div v-else class="empty-state">
        <svg class="empty-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
          <path d="M9 17H5a2 2 0 01-2-2V5a2 2 0 012-2h4m6 14h4a2 2 0 002-2V5a2 2 0 00-2-2h-4M9 12l3 3 3-3M12 15V3"/>
        </svg>
        <div class="empty-text">请先导入 Excel 数据</div>
        <div class="empty-hint">支持 .xlsx / .xls 格式</div>
      </div>
    </section>
  </div>
</template>

<script setup>
import { ref, computed, inject, onMounted, onUnmounted } from 'vue'
import axios from 'axios'
import DataImport from '../components/DataImport.vue'
import EncodingResult from '../components/EncodingResult.vue'
import EncodingExport from '../components/EncodingExport.vue'

const showToast = inject('showToast')

// 状态
const dataList = ref([])
const encodings = ref({})
const currentIndex = ref(-1)
const isEncoding = ref(false)
const isEncodingSingle = ref(false)
const encodeProgress = ref(0)
const activeBatchJobId = ref('')
const activeBatchJobStatus = ref('')
const activeBatchJobMeta = ref(null)
const runningBatchJobs = ref([])
const filter = ref('all')
const isImportCollapsed = ref(true)
const dataListRef = ref(null)
const defaultMaxConcurrent = ref(3)
const maxConcurrent = ref(3)
const showStopConfirmDialog = ref(false)
const isStopSubmitting = ref(false)
const showEditDialog = ref(false)
const editDialogMode = ref('single')
const editDialogType = ref('')
const editDialogIndex = ref(null)
const singleEditForm = ref({
  originalContent: '',
  originalCode: '',
  modifiedContent: '',
  modifiedCode: ''
})
const typeEditForm = ref({
  modelOriginalContent: '',
  stage1FinalContent: '',
  encodingInput: '',
  originalCode: '',
  modifiedContent: '',
  modifiedCode: ''
})
const sizeEditForm = ref({
  originalContent: '',
  originalCode: '',
  dn: '',
  od: '',
  inch: '',
  modifiedCode: ''
})
const thicknessEditForm = ref({
  originalContent: '',
  originalCode: '',
  mm: '',
  inch: '',
  schedule: '',
  series: '',
  bwg: '',
  modifiedCode: ''
})
const standardEditForm = ref({
  originalContent: '',
  originalCode: ''
})
const standardEditItems = ref([])
let batchJobEventSource = null
let editItemId = 0
const BATCH_JOB_KEEP_SECONDS = 60

const codeFieldOrder = ['TYPE', 'ENDS', 'SEAL', 'MANU', 'CONN', 'SIZE', 'THICKNESS', 'PRESSURE', 'MATERIAL', 'STANDARD']
const multiValueEditableFields = new Set(['STANDARD'])

const editDialogTitle = computed(() => {
  const labels = {
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
  return `修正${labels[editDialogType.value] || editDialogType.value || '字段'}编码`
})

function cloneDeep(value) {
  return JSON.parse(JSON.stringify(value))
}

function uniqueNonEmpty(values = []) {
  const result = []
  const seen = new Set()
  for (const value of values) {
    if (!value || seen.has(value)) continue
    seen.add(value)
    result.push(value)
  }
  return result
}

function ensureEncodingOriginalSnapshot(result) {
  if (!result || result.original_final_code !== undefined) return
  result.original_final_code = result.final_code || ''
  result.original_need_review = !!result.need_review
}

function ensureFieldOriginalSnapshot(field) {
  if (!field || field.original_snapshot) return
  field.original_snapshot = cloneDeep({
    original_value: field.original_value || '',
    stage1_final_value: field.stage1_final_value || '',
    original_values: field.original_values || [],
    matched_name: field.matched_name || '',
    matched_names: field.matched_names || [],
    encoding_input: field.encoding_input || '',
    code: field.code || '',
    codes: field.codes || [],
    manual_form: field.manual_form || null,
    similarity: field.similarity,
    need_review: field.need_review,
    display: field.display || '',
    items: field.items || []
  })
}

function ensureFieldItemOriginalSnapshot(item) {
  if (!item || item.original_snapshot) return
  item.original_snapshot = cloneDeep({
    original: item.original || '',
    matched: item.matched || '',
    code: item.code || '',
    similarity: item.similarity,
    need_review: item.need_review,
    category: item.category || '',
    base_code: item.base_code || '',
    grade: item.grade || ''
  })
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

function formatStructuredFieldText(value, type) {
  const parsed = safeParseJson(value)
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return String(value || '')
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
    if (angle) parts.push(`ANGLE: ${angle}`)
    if (body) parts.push(`BODY: ${body}`)
    if (radius) parts.push(`RADIUS: ${radius}`)
    if (manu.length) parts.push(`MANU: ${manu.join(' x ')}`)
    if (conn.length) parts.push(`CONN: ${conn.join(' x ')}`)
    if (seal.length) parts.push(`SEAL: ${seal.join(' x ')}`)
    if (ends.length) parts.push(`ENDS: ${ends.join(' x ')}`)
    return parts.join(' ; ')
  }

  const subtypeOrderMap = {
    SIZE: ['DN', 'OD', 'INCH', 'LENGTH'],
    THICKNESS: ['MM', 'INCH', 'SCHEDULE', 'SERIES', 'BWG']
  }
  const order = subtypeOrderMap[type] || []
  const keys = [
    ...order.filter(key => Object.prototype.hasOwnProperty.call(parsed, key)),
    ...Object.keys(parsed).filter(key => !order.includes(key) && !String(key).startsWith('_'))
  ]

  const parts = keys
    .map(key => {
      const values = Array.isArray(parsed[key]) ? parsed[key] : [parsed[key]]
      const normalized = values
        .map(item => (item && typeof item === 'object') ? '' : String(item || '').trim())
        .filter(Boolean)
      if (!normalized.length) return ''
      return `${key}: ${normalized.join(' x ')}`
    })
    .filter(Boolean)

  return parts.join(' ; ')
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

function formatTypeStructuredText(value) {
  const parsed = safeParseJson(value)
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    return String(value || '')
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
  if (body) parts.push(`BODY: ${body}`)
  if (angle) parts.push(`ANGLE: ${angle}`)
  if (radius) parts.push(`RADIUS: ${radius}`)
  if (manu.length) parts.push(`MANU: ${manu.join(' x ')}`)
  if (conn.length) parts.push(`CONN: ${conn.join(' x ')}`)
  if (seal.length) parts.push(`SEAL: ${seal.join(' x ')}`)
  if (ends.length) parts.push(`ENDS: ${ends.join(' x ')}`)
  return parts.join(' ; ')
}

function formatItemFieldText(items = []) {
  return items
    .map(item => {
      const original = String(item?.original || '').trim()
      if (!original) return ''
      return item?.category ? `${original}(${item.category})` : original
    })
    .filter(Boolean)
    .join(' | ')
}

function getEditableFieldText(fieldLike, type) {
  if (fieldLike?.items && fieldLike.items.length > 0) {
    const itemText = formatItemFieldText(fieldLike.items)
    if (itemText) return itemText
  }
  if (type === 'TYPE') {
    return formatTypeSummary(safeParseJson(fieldLike?.original_value || ''))
  }
  return formatStructuredFieldText(fieldLike?.original_value || '', type)
}

function buildTypeEditForm(fieldLike, snapshot) {
  return {
    modelOriginalContent: formatTypeStructuredText(snapshot?.original_value || ''),
    stage1FinalContent: formatTypeStructuredText(snapshot?.stage1_final_value || ''),
    encodingInput: String(snapshot?.encoding_input || snapshot?.matched_name || '').trim(),
    originalCode: snapshot?.code || '',
    modifiedContent: String(fieldLike?.encoding_input || fieldLike?.matched_name || '').trim(),
    modifiedCode: fieldLike?.code || ''
  }
}

function createEmptyStandardEditItem() {
  return {
    id: ++editItemId,
    subject: '',
    grade: '',
    method: '',
    appendix: '',
    code: '',
    snapshot: {
      original: '',
      code: ''
    }
  }
}

function splitJoinedValues(text) {
  return String(text || '')
    .split(/\s*[xX×]\s*/)
    .map(item => item.trim())
    .filter(Boolean)
}

function joinValues(values = []) {
  return values.filter(Boolean).join(' x ')
}

function toStructuredInputText(value) {
  return joinValues(Array.isArray(value) ? value.map(item => String(item || '').trim()) : [String(value || '').trim()])
}

function getStructuredFormSource(field, type) {
  if (field?.manual_form && field.manual_form.type === type) {
    return field.manual_form.values || {}
  }
  const parsed = safeParseJson(field?.original_value || '')
  if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) {
    return parsed
  }
  return {}
}

function buildStructuredSummary(values, orderedKeys) {
  return orderedKeys
    .map(key => {
      const text = String(values[key] || '').trim()
      return text ? `${key}: ${text}` : ''
    })
    .filter(Boolean)
    .join(' ; ')
}

function buildSizeEditForm(fieldLike) {
  const values = getStructuredFormSource(fieldLike, 'SIZE')
  return {
    originalContent: getEditableFieldText(fieldLike, 'SIZE'),
    originalCode: fieldLike?.code || '',
    dn: toStructuredInputText(values.DN),
    od: toStructuredInputText(values.OD),
    inch: toStructuredInputText(values.INCH),
    modifiedCode: fieldLike?.code || ''
  }
}

function buildThicknessEditForm(fieldLike) {
  const values = getStructuredFormSource(fieldLike, 'THICKNESS')
  return {
    originalContent: getEditableFieldText(fieldLike, 'THICKNESS'),
    originalCode: fieldLike?.code || '',
    mm: toStructuredInputText(values.MM),
    inch: toStructuredInputText(values.INCH),
    schedule: toStructuredInputText(values.SCHEDULE),
    series: toStructuredInputText(values.SERIES),
    bwg: toStructuredInputText(values.BWG),
    modifiedCode: fieldLike?.code || ''
  }
}

function buildStandardEditItems(field) {
  const items = (field?.items || []).map(item => {
    ensureFieldItemOriginalSnapshot(item)
    return {
      id: ++editItemId,
      subject: item.standard_subject || item.original || item.base_code || '',
      grade: item.standard_grade || item.grade || '',
      method: item.standard_method || '',
      appendix: item.standard_appendix || '',
      code: item.code || '',
      snapshot: cloneDeep(item.original_snapshot || {
        original: item.original || '',
        code: item.code || ''
      })
    }
  })
  return items.length ? items : [createEmptyStandardEditItem()]
}

function composeStandardOriginal(item) {
  return [item.subject, item.grade, item.method, item.appendix]
    .map(part => String(part || '').trim())
    .filter(Boolean)
    .join(' ')
}

function recomputeFieldState(field, type) {
  const shouldUseItems = multiValueEditableFields.has(type) || (field.items && field.items.length > 0)
  if (shouldUseItems) {
    const items = (field.items || []).filter(item => item && (item.original || item.code || item.matched))
    field.items = items
    field.original_values = items.map(item => item.original || '').filter(Boolean)
    field.matched_names = items.map(item => item.matched || item.original || '').filter(Boolean)
    field.codes = uniqueNonEmpty(items.map(item => item.code || ''))
    field.original_value = field.original_values.join(' | ')
    field.matched_name = field.matched_names.join(' | ')
    field.code = field.codes.join('')
    field.similarity = items.length > 0
      ? Math.min(...items.map(item => Number.isFinite(item.similarity) ? item.similarity : 1))
      : 1
    field.need_review = items.some(item => item.need_review)
    if (field.manual_override && type === 'STANDARD') {
      field.display = ''
    }
    return
  }

  field.original_values = field.original_value ? [field.original_value] : []
  field.matched_names = field.matched_name ? [field.matched_name] : []
  field.codes = field.code ? [field.code] : []
}

function recomputeEncodingResult(result) {
  if (!result || !result.fields) return
  let finalCode = ''
  for (const fieldType of codeFieldOrder) {
    const field = result.fields[fieldType]
    if (!field) continue
    recomputeFieldState(field, fieldType)
    if (field.code) finalCode += field.code
  }
  result.final_code = finalCode
  result.need_review = Object.values(result.fields).some(field => field.need_review)
}

// 键盘导航（左右键切换记录）
function handleKeydown(e) {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
    return
  }
  
  if (e.key === 'ArrowLeft' && currentIndex.value > 0) {
    e.preventDefault()
    currentIndex.value--
  } else if (e.key === 'ArrowRight' && currentIndex.value < dataList.value.length - 1) {
    e.preventDefault()
    currentIndex.value++
  }
}

// 生命周期
onMounted(() => {
  document.addEventListener('keydown', handleKeydown)
  loadBatchConfig()
  loadRunningBatchJobs()
})

onUnmounted(() => {
  document.removeEventListener('keydown', handleKeydown)
  closeBatchJobStream()
})

async function loadBatchConfig() {
  try {
    const res = await axios.get('/api/config')
    const backendDefault = Number(res.data?.batch_processing?.max_concurrent || 3)
    defaultMaxConcurrent.value = Number.isFinite(backendDefault) ? Math.max(1, Math.min(16, backendDefault)) : 3

    const saved = Number(localStorage.getItem('encoding_max_concurrent'))
    if (Number.isFinite(saved) && saved >= 1) {
      maxConcurrent.value = Math.max(1, Math.min(16, Math.trunc(saved)))
    } else {
      maxConcurrent.value = defaultMaxConcurrent.value
    }
  } catch (e) {
    defaultMaxConcurrent.value = 3
    const saved = Number(localStorage.getItem('encoding_max_concurrent'))
    maxConcurrent.value = Number.isFinite(saved) && saved >= 1 ? Math.max(1, Math.min(16, Math.trunc(saved))) : 3
  }
}

function handleMaxConcurrentChange(event) {
  const v = Number(event.target.value)
  if (!Number.isFinite(v) || v < 1) {
    maxConcurrent.value = defaultMaxConcurrent.value
  } else {
    maxConcurrent.value = Math.max(1, Math.min(16, Math.trunc(v)))
  }
  localStorage.setItem('encoding_max_concurrent', String(maxConcurrent.value))
  event.target.value = maxConcurrent.value
}

const currentText = computed(() => {
  return dataList.value[currentIndex.value]?.text || ''
})

const batchActionText = computed(() => {
  if (isEncoding.value) {
    if (isStopSubmitting.value) {
      return '停止中...'
    }
    return activeBatchJobId.value ? '■ 停止编码' : '创建任务中...'
  }
  return '一键编码'
})

const batchActionButtonClass = computed(() => {
  if (isEncoding.value) {
    return 'btn-danger batch-action-btn is-stopping'
  }
  return 'btn-primary batch-action-btn'
})

const currentEncoding = computed(() => {
  return encodings.value[currentIndex.value]
})

const currentProcessedText = computed(() => {
  return currentEncoding.value?.processed_text || ''
})

const showProcessedText = computed(() => {
  const processed = currentProcessedText.value
  const original = currentText.value
  return !!processed && processed !== original
})

const successCount = computed(() => {
  return Object.values(encodings.value).filter(e => e.success && !e.need_review).length
})

const reviewCount = computed(() => {
  return Object.values(encodings.value).filter(e => e.need_review).length
})

const filteredDataList = computed(() => {
  if (filter.value === 'all') {
    return dataList.value
  }
  return dataList.value.filter(item => {
    const enc = encodings.value[item.index]
    return enc && enc.need_review
  })
})

function getItemStatus(index) {
  const enc = encodings.value[index]
  if (!enc) return 'pending'
  if (hasManualCorrection(enc)) return 'corrected'
  if (enc.need_review) return 'review'
  if (enc.success) return 'success'
  return 'pending'
}

function getItemDifficulty(index) {
  const enc = encodings.value[index]
  return enc?.second_pass?.final_level || enc?.difficulty_split?.difficulty || ''
}

function getItemDifficultyClass(index) {
  const level = getItemDifficulty(index)
  if (level === '困难') return 'hard'
  if (level === '二次简单') return 'second-pass-easy'
  if (level === '简单') return 'second-pass-mid'
  return 'simple'
}

function hasManualCorrection(result) {
  if (!result?.fields) return false
  return Object.values(result.fields).some(field => {
    if (!field) return false
    if (field.manual_override) return true
    return (field.items || []).some(item => item?.manual_override)
  })
}

// 识别当条
async function handleEncodeSingle() {
  await encodeCurrentItem('识别完成', '识别失败')
}

// 重新编码当条
async function handleReEncode() {
  await encodeCurrentItem('重新编码完成', '重新编码失败')
}

// 编码当前条目
async function encodeCurrentItem(successMsg, failMsg) {
  if (currentIndex.value < 0) return
  
  const text = dataList.value[currentIndex.value]?.text
  if (!text) return
  
  isEncodingSingle.value = true
  
  try {
    // 1. NER 提取实体
    const predictRes = await axios.post('/api/pipe/predict', {
      text,
      preprocess: true
    })
    
    if (!predictRes.data.success) {
      encodings.value[currentIndex.value] = {
        original_text: text,
        processed_text: predictRes.data.processed_text || text,
        final_code: '',
        success: false,
        need_review: true,
        errors: ['NER识别失败: ' + (predictRes.data.error || '未知错误')],
        fields: {}
      }
      showToast(failMsg, 'error')
      return
    }

    const routeInfo = predictRes.data.route_info || null
    if (routeInfo && routeInfo.encoding_enabled === false) {
      encodings.value[currentIndex.value] = {
        original_text: text,
        processed_text: predictRes.data.processed_text || text,
        final_code: '',
        success: true,
        need_review: false,
        skipped_encoding: true,
        skip_reason: routeInfo.skip_encoding_reason || '',
        errors: [],
        fields: {},
        route_info: routeInfo,
        stage1_output: {
          ...(predictRes.data.model_output || {}),
          _STRUCTURAL_PROMPT: predictRes.data.structural_prompt_output || null
        },
        stage1_raw_response: [
          predictRes.data.model_raw_response || '',
          predictRes.data.structural_prompt_raw_response
            ? `\n\n[STRUCTURAL_PROMPT_RAW]\n${predictRes.data.structural_prompt_raw_response}`
            : ''
        ].join('')
      }
      showToast(routeInfo.skip_encoding_reason || successMsg, 'success')
      return
    }
    
    // 2. 实体编码
    const encodeRes = await axios.post('/api/pipe/encode', {
      entities: predictRes.data.entities,
      extract_confidence: predictRes.data.extract_confidence,
      extract_confidence_v2: predictRes.data.extract_confidence_v2,
      text,
      project_name: dataList.value[currentIndex.value]?.projectName || ''
    })

    encodings.value[currentIndex.value] = {
      ...encodeRes.data,
      processed_text: predictRes.data.processed_text || text,
      extract_confidence_v2: encodeRes.data.extract_confidence_v2 || predictRes.data.extract_confidence_v2 || {},
      route_info: predictRes.data.route_info || null,
      stage1_output: {
        ...(predictRes.data.model_output || {}),
        _STRUCTURAL_PROMPT: predictRes.data.structural_prompt_output || null
      },
      stage1_raw_response: [
        predictRes.data.model_raw_response || '',
        predictRes.data.structural_prompt_raw_response
          ? `\n\n[STRUCTURAL_PROMPT_RAW]\n${predictRes.data.structural_prompt_raw_response}`
          : ''
      ].join('')
    }
    showToast(successMsg, 'success')
    
  } catch (e) {
    console.error('编码失败:', e)
    showToast(failMsg + ': ' + (e.message || '网络错误'), 'error')
  } finally {
    isEncodingSingle.value = false
  }
}

// 数据导入处理
function handleDataLoaded({ data, column }) {
  dataList.value = data.map((row, index) => ({
    index,
    text: row[column] || '',
    projectName: resolveProjectName(row),
    rawRow: row
  })).filter(item => item.text)
  
  currentIndex.value = -1
  encodings.value = {}
  isImportCollapsed.value = true
  
  showToast(`成功导入 ${dataList.value.length} 条数据`, 'success')
  
  if (dataList.value.length > 0) {
    currentIndex.value = 0
  }
}

function resolveProjectName(row) {
  const projectKeys = ['项目名称', '子表.项目名称', 'project', 'PROJECT']
  for (const key of projectKeys) {
    const value = row?.[key]
    if (value != null && String(value).trim() !== '') {
      return String(value).trim()
    }
  }
  return ''
}

function buildEncodingEntry(encodeData, predictMeta) {
  return {
    ...encodeData,
    processed_text: encodeData?.processed_text || predictMeta?.processed_text || encodeData?.original_text || '',
    extract_confidence_v2: encodeData?.extract_confidence_v2 || predictMeta?.extract_confidence_v2 || {},
    route_info: encodeData?.route_info || predictMeta?.route_info || null,
    stage1_output: encodeData?.stage1_output || predictMeta?.stage1_output || null,
    stage1_raw_response: encodeData?.stage1_raw_response || predictMeta?.stage1_raw_response || ''
  }
}

function shortJobId(jobId) {
  const text = String(jobId || '')
  return text ? text.slice(0, 8) : '—'
}

function taskStatusText(status) {
  const value = String(status || '').trim()
  if (value === 'queued') return '排队中'
  if (value === 'running') return '运行中'
  if (value === 'cancelling') return '停止中'
  if (value === 'finished') return '已完成'
  if (value === 'cancelled') return '已停止'
  if (value === 'failed') return '失败'
  return value || '未知'
}

function syncRunningBatchJob(job) {
  if (!job || typeof job !== 'object' || !job.job_id) return
  const next = { ...job }
  const index = runningBatchJobs.value.findIndex(item => item.job_id === next.job_id)
  const isActive = ['queued', 'running', 'cancelling'].includes(String(next.status || ''))
  if (!isActive) {
    if (index >= 0) {
      runningBatchJobs.value.splice(index, 1)
    }
    return
  }
  if (index >= 0) {
    runningBatchJobs.value.splice(index, 1, next)
  } else {
    runningBatchJobs.value.push(next)
    runningBatchJobs.value.sort((a, b) => Number(a.created_at || 0) - Number(b.created_at || 0))
  }
}

function closeBatchJobStream() {
  if (batchJobEventSource) {
    batchJobEventSource.close()
    batchJobEventSource = null
  }
}

function clearActiveBatchJob() {
  activeBatchJobId.value = ''
  activeBatchJobStatus.value = ''
  activeBatchJobMeta.value = null
  closeBatchJobStream()
}

function applyBatchJobSnapshot(job) {
  if (!job || typeof job !== 'object') return
  const items = Array.isArray(job.items) ? job.items : []
  if (items.length > 0) {
    dataList.value = items.map((item, idx) => ({
      index: Number.isFinite(Number(item.index)) ? Number(item.index) : idx,
      text: item.text || '',
      projectName: item.project_name || '',
      rawRow: null
    })).sort((a, b) => a.index - b.index)
    if (currentIndex.value < 0 && dataList.value.length > 0) {
      currentIndex.value = 0
    }
  }

  const nextEncodings = {}
  const results = job.results && typeof job.results === 'object' ? job.results : {}
  Object.entries(results).forEach(([index, result]) => {
    const numericIndex = Number(index)
    if (Number.isFinite(numericIndex)) {
      nextEncodings[numericIndex] = buildEncodingEntry(result || {}, null)
    }
  })
  encodings.value = nextEncodings

  activeBatchJobId.value = String(job.job_id || '')
  activeBatchJobStatus.value = String(job.status || '')
  activeBatchJobMeta.value = { ...job }
  encodeProgress.value = job.total ? Math.round((Number(job.processed || 0) / Number(job.total || 1)) * 100) : 0
  isEncoding.value = ['queued', 'running', 'cancelling'].includes(activeBatchJobStatus.value)
  syncRunningBatchJob(job)
}

function isBatchJobDetailAvailable(jobId) {
  if (!jobId || activeBatchJobId.value !== jobId) return false
  const status = String(activeBatchJobStatus.value || '')
  if (['queued', 'running', 'cancelling'].includes(status)) {
    return true
  }
  if (!['finished', 'cancelled', 'failed'].includes(status)) {
    return false
  }
  const finishedAt = Number(activeBatchJobMeta.value?.finished_at || 0)
  if (!finishedAt) {
    return false
  }
  return (Date.now() / 1000) - finishedAt < BATCH_JOB_KEEP_SECONDS
}

async function loadRunningBatchJobs() {
  try {
    const res = await axios.get('/api/pipe/encode/batch/jobs')
    const jobs = Array.isArray(res.data?.jobs) ? res.data.jobs : []
    runningBatchJobs.value = jobs
  } catch (err) {
    runningBatchJobs.value = []
  }
}

function handleBatchJobEvent(event) {
  if (!event || typeof event !== 'object') return
  if (event.type === 'snapshot') {
    const snapshot = event.snapshot || {}
    syncRunningBatchJob(snapshot)
    applyBatchJobSnapshot(snapshot)
    return
  }

  const snapshot = event.snapshot && typeof event.snapshot === 'object' ? event.snapshot : null
  const index = Number(event.index)
  if (Number.isFinite(index) && event.result) {
    encodings.value[index] = {
      ...(encodings.value[index] || {}),
      ...buildEncodingEntry(event.result || {}, null),
    }
  }
  if (snapshot) {
    activeBatchJobStatus.value = String(snapshot.status || '')
    encodeProgress.value = snapshot.total ? Math.round((Number(snapshot.processed || 0) / Number(snapshot.total || 1)) * 100) : 0
    isEncoding.value = ['queued', 'running', 'cancelling'].includes(activeBatchJobStatus.value)
    syncRunningBatchJob(snapshot)
  }
  if (event.type === 'end' || event.type === 'cancelled' || event.type === 'failed') {
    if (snapshot) {
      applyBatchJobSnapshot(snapshot)
    }
    isEncoding.value = false
    isStopSubmitting.value = false
    showStopConfirmDialog.value = false
    if (event.type === 'cancelled') {
      showToast('批量编码已停止', 'success')
    } else if (event.type === 'end') {
      showToast(`编码完成: ${successCount.value} 成功, ${reviewCount.value} 待审核`, 'success')
    } else if (event.type === 'failed') {
      showToast(`批量编码失败: ${event.error || '未知错误'}`, 'error')
    }
    closeBatchJobStream()
  }
}

function subscribeBatchJob(jobId) {
  closeBatchJobStream()
  batchJobEventSource = new EventSource(`/api/pipe/encode/batch/jobs/${jobId}/stream`)
  batchJobEventSource.onmessage = (messageEvent) => {
    try {
      const event = JSON.parse(messageEvent.data)
      handleBatchJobEvent(event)
    } catch (err) {
      console.error('解析批量任务事件失败:', err)
    }
  }
  batchJobEventSource.onerror = () => {
    if (!activeBatchJobId.value) {
      closeBatchJobStream()
    }
  }
}

async function openBatchJob(jobId) {
  if (!jobId) return
  try {
    const res = await axios.get(`/api/pipe/encode/batch/jobs/${jobId}`)
    const job = res.data?.job
    if (!job) {
      return
    }
    applyBatchJobSnapshot(job)
    if (['queued', 'running', 'cancelling'].includes(String(job.status || ''))) {
      subscribeBatchJob(jobId)
    }
  } catch (err) {
    showToast(`加载任务失败: ${err.message || '网络错误'}`, 'error')
  }
}

function handleSelectItem(index) {
  currentIndex.value = index
  if (activeBatchJobId.value && isBatchJobDetailAvailable(activeBatchJobId.value)) {
    loadBatchJobItemDetail(activeBatchJobId.value, index)
  }
}

async function loadBatchJobItemDetail(jobId, itemIndex) {
  if (!jobId || !Number.isFinite(Number(itemIndex))) return
  try {
    const res = await axios.get(`/api/pipe/encode/batch/jobs/${jobId}/items/${itemIndex}`)
    const detail = res.data || {}
    const result = detail.result && typeof detail.result === 'object' ? detail.result : null
    if (result) {
      encodings.value[itemIndex] = {
        ...(encodings.value[itemIndex] || {}),
        ...buildEncodingEntry(result, null),
      }
    }
  } catch (err) {
    const statusCode = Number(err?.response?.status || 0)
    if ((statusCode === 404 || statusCode === 410) && activeBatchJobId.value === jobId) {
      activeBatchJobId.value = ''
      activeBatchJobStatus.value = ''
      activeBatchJobMeta.value = null
    }
  }
}

async function createBatchJob(items) {
  const res = await axios.post('/api/pipe/encode/batch/jobs', {
    items,
    max_concurrent: maxConcurrent.value
  })
  return res.data?.job || null
}

async function handleCancelBatchJob() {
  showStopConfirmDialog.value = true
}

function closeStopConfirmDialog() {
  if (isStopSubmitting.value) return
  showStopConfirmDialog.value = false
}

async function confirmCancelBatchJob() {
  if (!activeBatchJobId.value) return
  try {
    isStopSubmitting.value = true
    const res = await axios.post(`/api/pipe/encode/batch/jobs/${activeBatchJobId.value}/cancel`)
    const job = res.data?.job
    activeBatchJobStatus.value = String(job?.status || 'cancelling')
    if (job) {
      syncRunningBatchJob(job)
    }
  } catch (err) {
    isStopSubmitting.value = false
    showToast(`停止失败: ${err.message || '网络错误'}`, 'error')
  }
}

function handlePrimaryBatchAction() {
  if (isEncoding.value) {
    handleCancelBatchJob()
    return
  }
  handleBatchEncode()
}

// 一键编码
async function handleBatchEncode() {
  if (dataList.value.length === 0) return

  isEncoding.value = true
  encodeProgress.value = 0

  const batchItems = dataList.value.map((item, index) => ({
    client_index: index,
    text: item.text,
    project_name: item.projectName || '',
    preprocess: true
  }))

  try {
    const job = await createBatchJob(batchItems)
    if (!job?.job_id) {
      throw new Error('创建批量任务失败')
    }
    applyBatchJobSnapshot(job)
    subscribeBatchJob(job.job_id)
  } catch (err) {
    showToast(`编码失败: ${err.message}`, 'error')
    isEncoding.value = false
  } finally {
    if (!activeBatchJobId.value) {
      isEncoding.value = false
    }
  }
}

// 页码跳转
function handlePageJump(event) {
  const input = event.target
  const value = parseInt(input.value, 10)
  
  if (!isNaN(value) && value >= 1 && value <= dataList.value.length) {
    currentIndex.value = value - 1
    scrollToCurrentItem()
  } else {
    // 恢复当前值
    input.value = currentIndex.value + 1
  }
  
  // 失去焦点
  if (event.type === 'keydown') {
    input.blur()
  }
}

// 滚动列表到当前项
function scrollToCurrentItem() {
  if (!dataListRef.value) return
  
  const targetItem = dataListRef.value.querySelector(`[data-index="${currentIndex.value}"]`)
  if (targetItem) {
    targetItem.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }
}

function handleEditField({ type, index = null }) {
  if (!currentEncoding.value || !currentEncoding.value.fields?.[type]) return

  const field = currentEncoding.value.fields[type]
  editDialogType.value = type
  editDialogIndex.value = index

  const snapshot = field.original_snapshot || {
    original_value: field.original_value || '',
    stage1_final_value: field.stage1_final_value || '',
    encoding_input: field.encoding_input || '',
    code: field.code || '',
    items: cloneDeep(field.items || []),
    manual_form: cloneDeep(field.manual_form || null)
  }

  if (type === 'TYPE') {
    editDialogMode.value = 'type'
    typeEditForm.value = buildTypeEditForm(field, snapshot)
  } else if (type === 'SIZE') {
    editDialogMode.value = 'size'
    sizeEditForm.value = buildSizeEditForm(field)
    sizeEditForm.value.originalContent = getEditableFieldText(snapshot, type)
    sizeEditForm.value.originalCode = snapshot.code || ''
  } else if (type === 'THICKNESS') {
    editDialogMode.value = 'thickness'
    thicknessEditForm.value = buildThicknessEditForm(field)
    thicknessEditForm.value.originalContent = getEditableFieldText(snapshot, type)
    thicknessEditForm.value.originalCode = snapshot.code || ''
  } else if (type === 'STANDARD') {
    editDialogMode.value = 'standard'
    standardEditForm.value = {
      originalContent: getEditableFieldText(snapshot, type),
      originalCode: snapshot.code || ''
    }
    standardEditItems.value = buildStandardEditItems(field)
  } else {
    editDialogMode.value = 'single'
    singleEditForm.value = {
      originalContent: getEditableFieldText(snapshot, type),
      originalCode: snapshot.code || '',
      modifiedContent: getEditableFieldText(field, type),
      modifiedCode: field.code || ''
    }
  }

  showEditDialog.value = true
}

function closeEditDialog() {
  showEditDialog.value = false
  editDialogMode.value = 'single'
  editDialogType.value = ''
  editDialogIndex.value = null
  typeEditForm.value = {
    modelOriginalContent: '',
    stage1FinalContent: '',
    encodingInput: '',
    originalCode: '',
    modifiedContent: '',
    modifiedCode: ''
  }
  singleEditForm.value = {
    originalContent: '',
    originalCode: '',
    modifiedContent: '',
    modifiedCode: ''
  }
  sizeEditForm.value = {
    originalContent: '',
    originalCode: '',
    dn: '',
    od: '',
    inch: '',
    modifiedCode: ''
  }
  thicknessEditForm.value = {
    originalContent: '',
    originalCode: '',
    mm: '',
    inch: '',
    schedule: '',
    series: '',
    bwg: '',
    modifiedCode: ''
  }
  standardEditForm.value = {
    originalContent: '',
    originalCode: ''
  }
  standardEditItems.value = []
}

function addStandardEditItem() {
  standardEditItems.value.push(createEmptyStandardEditItem())
}

function removeStandardEditItem(index) {
  if (standardEditItems.value.length === 1) {
    standardEditItems.value.splice(0, 1, createEmptyStandardEditItem())
    return
  }
  standardEditItems.value.splice(index, 1)
}

function moveStandardEditItem(index, delta) {
  const target = index + delta
  if (target < 0 || target >= standardEditItems.value.length) return
  const [item] = standardEditItems.value.splice(index, 1)
  standardEditItems.value.splice(target, 0, item)
}

function confirmEditDialog() {
  if (!currentEncoding.value || !editDialogType.value) return

  const field = currentEncoding.value.fields?.[editDialogType.value]
  if (!field) return

  ensureEncodingOriginalSnapshot(currentEncoding.value)
  ensureFieldOriginalSnapshot(field)

  if (editDialogMode.value === 'size') {
    const manualValues = {
      DN: splitJoinedValues(sizeEditForm.value.dn),
      OD: splitJoinedValues(sizeEditForm.value.od),
      INCH: splitJoinedValues(sizeEditForm.value.inch)
    }
    field.original_value = buildStructuredSummary({
      DN: joinValues(manualValues.DN),
      OD: joinValues(manualValues.OD),
      INCH: joinValues(manualValues.INCH)
    }, ['DN', 'OD', 'INCH'])
    field.matched_name = field.original_value
    field.code = sizeEditForm.value.modifiedCode.trim()
    field.codes = field.code ? [field.code] : []
    field.original_values = field.original_value ? [field.original_value] : []
    field.matched_names = field.matched_name ? [field.matched_name] : []
    field.items = []
    field.manual_form = { type: 'SIZE', values: manualValues }
    field.similarity = 1
    field.need_review = false
    field.manual_override = true
  } else if (editDialogMode.value === 'type') {
    field.encoding_input = typeEditForm.value.modifiedContent.trim()
    field.matched_name = typeEditForm.value.modifiedContent.trim()
    field.code = typeEditForm.value.modifiedCode.trim()
    field.codes = field.code ? [field.code] : []
    field.matched_names = field.matched_name ? [field.matched_name] : []
    field.similarity = 1
    field.need_review = false
    field.manual_override = true
  } else if (editDialogMode.value === 'thickness') {
    const manualValues = {
      MM: splitJoinedValues(thicknessEditForm.value.mm),
      INCH: splitJoinedValues(thicknessEditForm.value.inch),
      SCHEDULE: splitJoinedValues(thicknessEditForm.value.schedule),
      SERIES: splitJoinedValues(thicknessEditForm.value.series),
      BWG: splitJoinedValues(thicknessEditForm.value.bwg)
    }
    field.original_value = buildStructuredSummary({
      MM: joinValues(manualValues.MM),
      INCH: joinValues(manualValues.INCH),
      SCHEDULE: joinValues(manualValues.SCHEDULE),
      SERIES: joinValues(manualValues.SERIES),
      BWG: joinValues(manualValues.BWG)
    }, ['MM', 'INCH', 'SCHEDULE', 'SERIES', 'BWG'])
    field.matched_name = field.original_value
    field.code = thicknessEditForm.value.modifiedCode.trim()
    field.codes = field.code ? [field.code] : []
    field.original_values = field.original_value ? [field.original_value] : []
    field.matched_names = field.matched_name ? [field.matched_name] : []
    field.items = []
    field.manual_form = { type: 'THICKNESS', values: manualValues }
    field.similarity = 1
    field.need_review = false
    field.manual_override = true
  } else if (editDialogMode.value === 'standard') {
    field.items = standardEditItems.value
      .map(item => {
        const subject = String(item.subject || '').trim()
        const grade = String(item.grade || '').trim()
        const method = String(item.method || '').trim()
        const appendix = String(item.appendix || '').trim()
        const code = String(item.code || '').trim()
        const original = composeStandardOriginal({ subject, grade, method, appendix })
        if (!original && !code) return null
        const snapshot = cloneDeep(item.snapshot || {
          original: '',
          code: ''
        })
        const isModified = original !== String(snapshot.original || '').trim() || code !== String(snapshot.code || '').trim()
        return {
          original,
          matched: original,
          code,
          similarity: 1,
          is_exact: true,
          need_review: false,
          candidates: [],
          category: '',
          base_code: subject,
          grade,
          standard_subject: subject,
          standard_grade: grade,
          standard_method: method,
          standard_appendix: appendix,
          manual_override: isModified,
          original_snapshot: snapshot
        }
      })
      .filter(Boolean)
    field.display = ''
    field.manual_form = null
    field.manual_override = field.items.some(item => item.manual_override)
  } else if (editDialogMode.value === 'single') {
    field.original_value = singleEditForm.value.modifiedContent.trim()
    field.matched_name = singleEditForm.value.modifiedContent.trim()
    field.code = singleEditForm.value.modifiedCode.trim()
    field.similarity = 1
    field.need_review = false
    field.manual_override = true
    field.manual_form = null
    field.original_values = field.original_value ? [field.original_value] : []
    field.matched_names = field.matched_name ? [field.matched_name] : []
    field.codes = field.code ? [field.code] : []
    field.items = []
    if (editDialogType.value === 'STANDARD') {
      field.display = ''
    }
  }

  recomputeEncodingResult(currentEncoding.value)
  closeEditDialog()
  showToast('修正已更新', 'success')
}

// 选择候选项
function handleSelectCandidate({ type, index, candidate }) {
  if (!currentEncoding.value || !currentEncoding.value.fields[type]) return
  
  const field = currentEncoding.value.fields[type]
  ensureEncodingOriginalSnapshot(currentEncoding.value)
  ensureFieldOriginalSnapshot(field)
  
  // 如果有 index，说明是更新 items 中的某一项
  if (typeof index === 'number' && field.items && field.items[index]) {
    ensureFieldItemOriginalSnapshot(field.items[index])
    // 更新该项
    field.items[index].matched = candidate.name
    field.items[index].code = candidate.code
    field.items[index].similarity = candidate.similarity
    field.items[index].need_review = false
    field.items[index].manual_override = true
    field.manual_override = true
  } else {
    // 单值字段，直接更新
    field.matched_name = candidate.name
    field.code = candidate.code
    field.similarity = candidate.similarity
    field.need_review = false
    field.manual_override = true
  }

  recomputeEncodingResult(currentEncoding.value)
  showToast('已更新', 'success')
}
</script>

<style scoped>
.encoding-view {
  display: flex;
  height: 100%;
  width: 100%;
  overflow: hidden;
}

.sidebar {
  width: 300px;
  min-width: 300px;
  background: var(--bg-primary);
  border-right: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.sidebar-section {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border-light);
}

.task-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.task-item {
  padding: 10px 12px;
  border: 1px solid var(--border-light);
  border-radius: 8px;
  cursor: pointer;
  background: var(--bg-secondary);
}

.task-item.active {
  border-color: var(--primary-color);
  background: rgba(59, 130, 246, 0.08);
}

.task-item-main,
.task-item-sub {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
}

.task-item-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
}

.task-item-status,
.task-item-sub {
  font-size: 12px;
  color: var(--text-muted);
}

/* 导入区域可折叠 */
.import-section.collapsed {
  padding: 8px 16px;
}

.import-section.collapsed .import-content {
  display: none;
}

.import-summary {
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 4px;
}

.section-header.clickable {
  cursor: pointer;
  user-select: none;
}

.collapse-icon {
  font-size: 10px;
  color: var(--text-muted);
}

/* 操作区域紧凑 */
.operation-section {
  padding: 12px 16px;
}

.batch-action-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 46px;
  font-weight: 600;
  letter-spacing: 0.2px;
  transition: background-color 0.22s ease, border-color 0.22s ease, color 0.22s ease, box-shadow 0.22s ease, transform 0.18s ease;
}

.batch-action-btn:hover:not(:disabled) {
  transform: translateY(-1px);
}

.batch-action-btn.is-stopping {
  box-shadow: 0 8px 18px rgba(220, 38, 38, 0.16);
}

.concurrency-row {
  margin-top: 8px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.concurrency-label {
  font-size: 12px;
  color: var(--text-secondary);
  white-space: nowrap;
}

.concurrency-input {
  width: 60px;
  height: 24px;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  background: var(--bg-primary);
  color: var(--text-primary);
  font-size: 12px;
  padding: 2px 6px;
}

.concurrency-hint {
  font-size: 11px;
  color: var(--text-muted);
}

.progress-bar {
  margin-top: 8px;
  height: 4px;
  background: var(--bg-tertiary);
  border-radius: 2px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: var(--primary);
  transition: width 0.2s;
}

.stats-row {
  display: flex;
  gap: 12px;
  margin-top: 10px;
  font-size: 12px;
  color: var(--text-secondary);
}

.stats-row .stat b {
  color: var(--text-primary);
  font-weight: 600;
}

.stats-row .stat.success b { color: var(--success); }
.stats-row .stat.warning b { color: #b06000; }

/* 数据列表占主要空间 */
.list-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  min-height: 0;
  border-bottom: none;
  padding-bottom: 0;
}

.section-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.filter-tabs {
  display: flex;
  background: var(--bg-tertiary);
  border-radius: 4px;
  padding: 2px;
}

.filter-tab {
  padding: 3px 10px;
  font-size: 11px;
  font-weight: 500;
  color: var(--text-secondary);
  background: none;
  border: none;
  border-radius: 3px;
  cursor: pointer;
}

.filter-tab.active {
  background: var(--bg-primary);
  color: var(--text-primary);
}

.data-list {
  flex: 1;
  overflow-y: auto;
  min-height: 0;
  margin: 0 -16px;
  padding: 0 16px;
}

.list-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 10px;
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.15s;
  margin-bottom: 2px;
  border-left: 3px solid transparent;
}

.list-item:hover {
  background: var(--bg-tertiary);
}

.list-item.active {
  background: var(--primary-light);
  border-left-color: var(--primary);
}

.list-item.corrected {
  border-left-color: #dc2626;
}

.list-item.success {
  border-left-color: var(--success);
}

.list-item.warning {
  border-left-color: var(--warning);
}

.item-num {
  font-size: 10px;
  color: var(--text-muted);
  min-width: 24px;
}

.item-text {
  flex: 1;
  font-size: 12px;
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.item-badge {
  font-size: 9px;
  font-weight: 600;
  padding: 1px 5px;
  border-radius: 2px;
  background: var(--success);
  color: #fff;
}

.item-badge.review {
  background: var(--warning);
  color: #000;
}

.item-badge.corrected {
  background: #dc2626;
  color: #fff;
}

.item-difficulty {
  font-size: 9px;
  font-weight: 600;
  padding: 1px 6px;
  border-radius: 999px;
  white-space: nowrap;
  border: 1px solid transparent;
}

.item-difficulty.simple {
  background: #e8f5e9;
  color: #2e7d32;
  border-color: #b7dfbc;
}

.item-difficulty.hard {
  background: #fff7ed;
  color: #c2410c;
  border-color: #fed7aa;
}

.item-second-pass.second-pass-easy {
  background: #ecfdf3;
  color: #027a48;
  border-color: #abefc6;
}

.item-second-pass.second-pass-mid {
  background: #fffaeb;
  color: #b54708;
  border-color: #fedf89;
}

.item-second-pass.second-pass-hard {
  background: #fef3f2;
  color: #b42318;
  border-color: #fecdca;
}

/* 右侧内容区 */
.content {
  flex: 1;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  overflow-y: auto;
  min-height: 0;
}

.source-card {
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 8px;
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

.card-actions {
  display: flex;
  align-items: center;
  gap: 10px;
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

.card-index-hash {
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

.card-index-total {
  color: var(--text-muted);
}

.btn-sm {
  padding: 4px 10px;
  font-size: 11px;
}

.btn-outline {
  background: transparent;
  border: 1px solid var(--primary);
  color: var(--primary);
  border-radius: 4px;
  cursor: pointer;
  transition: all 0.15s;
}

.btn-outline:hover {
  background: var(--primary);
  color: #fff;
}

.btn-outline:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.source-text {
  font-size: 14px;
  line-height: 1.5;
  color: var(--text-primary);
  font-family: 'SF Mono', Monaco, 'Courier New', monospace;
  word-break: break-all;
}

.processed-text-row {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--border-color);
}

.processed-text-label {
  display: block;
  margin-bottom: 6px;
  font-size: 12px;
  color: var(--text-secondary);
}

.processed-text {
  font-size: 13px;
  line-height: 1.5;
  color: var(--text-secondary);
  font-family: 'SF Mono', Monaco, 'Courier New', monospace;
  word-break: break-all;
}

.waiting-card {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  background: var(--bg-primary);
  border: 1px dashed var(--border-color);
  border-radius: 8px;
  min-height: 150px;
}

.btn-encode-single {
  padding: 12px 32px;
  font-size: 15px;
  font-weight: 500;
  color: #fff;
  background: linear-gradient(135deg, #3b82f6, #2563eb);
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-encode-single:hover {
  background: linear-gradient(135deg, #2563eb, #1d4ed8);
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
}

.btn-encode-single:disabled {
  opacity: 0.6;
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
}

.waiting-hint {
  color: var(--text-muted);
  font-size: 13px;
}

.empty-state {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: var(--text-muted);
}

.empty-icon {
  width: 48px;
  height: 48px;
  margin-bottom: 12px;
  opacity: 0.4;
}

.empty-text {
  font-size: 14px;
  font-weight: 500;
  margin-bottom: 4px;
}

.empty-hint {
  font-size: 12px;
  opacity: 0.7;
}

.dialog-overlay {
  position: fixed;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background: rgba(15, 23, 42, 0.28);
  z-index: 1000;
}

.dialog-content {
  background: var(--bg-primary);
  border: 1px solid var(--border-color);
  border-radius: 14px;
  width: min(780px, 100%);
  max-width: 92vw;
  max-height: 88vh;
  overflow: hidden;
  box-shadow: 0 24px 60px rgba(15, 23, 42, 0.18);
}

.dialog-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 16px;
  border-bottom: 1px solid var(--border-light);
}

.dialog-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
}

.dialog-close {
  background: none;
  border: none;
  font-size: 20px;
  line-height: 1;
  color: var(--text-muted);
  cursor: pointer;
}

.dialog-body {
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 16px 18px;
  max-height: calc(88vh - 132px);
  overflow-y: auto;
  background: linear-gradient(180deg, #fafcff 0%, #ffffff 100%);
}

.dialog-footer {
  display: flex;
  justify-content: flex-end;
  gap: 10px;
  padding: 0 18px 18px;
}

.stop-dialog {
  width: min(460px, 100%);
}

.stop-dialog-text {
  font-size: 14px;
  line-height: 1.7;
  color: var(--text-primary);
}

.stop-dialog-loading {
  display: flex;
  align-items: center;
  gap: 10px;
  min-height: 48px;
  font-size: 14px;
  color: var(--text-primary);
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.dialog-meta-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px 16px;
}

.form-group label {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-secondary);
}

.field-example {
  font-size: 11px;
  line-height: 1.3;
  color: #a16207;
  padding: 0 2px;
}

.form-hint {
  margin-top: 0;
  font-size: 11px;
  color: #92400e;
  background: #fff7ed;
  border: 1px solid #fed7aa;
  border-radius: 6px;
  padding: 6px 10px;
}

.form-input {
  width: 100%;
  min-height: 40px;
  padding: 10px 12px;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  background: var(--bg-primary);
  color: var(--text-primary);
  font-size: 14px;
  outline: none;
}

.form-input:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.12);
}

.form-input[readonly] {
  background: #f8fafc;
}

.form-input-static {
  user-select: none;
  pointer-events: none;
}

.multi-edit-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2px;
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}

.multi-edit-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.structured-edit-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px 14px;
}

.structured-edit-grid-3 {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.structured-edit-grid-4 {
  grid-template-columns: repeat(4, minmax(0, 1fr));
}

.multi-edit-row {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  padding: 12px;
  box-shadow: 0 6px 20px rgba(15, 23, 42, 0.04);
}

.multi-edit-row-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

@media (max-width: 960px) {
  .dialog-meta-grid,
  .structured-edit-grid-3,
  .structured-edit-grid-4 {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 720px) {
  .dialog-meta-grid,
  .structured-edit-grid,
  .structured-edit-grid-3,
  .structured-edit-grid-4 {
    grid-template-columns: minmax(0, 1fr);
  }
}

.multi-edit-row-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}

.multi-edit-row-actions {
  display: flex;
  gap: 6px;
}

.multi-edit-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}

.btn-xs {
  padding: 4px 8px;
  border-radius: 4px;
  border: 1px solid var(--border-color);
  background: var(--bg-primary);
  color: var(--text-primary);
  cursor: pointer;
  font-size: 12px;
}

.btn-xs:disabled {
  opacity: 0.45;
  cursor: not-allowed;
}

.btn-xs.btn-danger {
  color: #dc2626;
}
</style>
