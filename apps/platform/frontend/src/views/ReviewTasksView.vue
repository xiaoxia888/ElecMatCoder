<template>
  <div class="review-tasks-view">
    <div class="toolbar">
      <h2 class="page-title">代码审核 - 任务列表</h2>
      <div class="toolbar-right">
        <button class="btn btn-outline btn-sm" @click="resetFilters" v-if="hasActiveFilters">重置筛选</button>
        <button class="btn btn-outline btn-sm" @click="showFilters = !showFilters">
          筛选 {{ showFilters ? '↑' : '↓' }}
        </button>
        <button class="btn btn-outline btn-sm" @click="refreshTaskList" :disabled="loading">
          {{ loading ? '加载中...' : '刷新' }}
        </button>
      </div>
    </div>

    <div class="filter-bar" v-show="showFilters">
      <div class="filter-row">
        <div class="filter-item">
          <label>任务编号</label>
          <input v-model="filters.filterTaskCode" type="text" placeholder="请输入" @keyup.enter="applyFilters" />
        </div>
        <div class="filter-item">
          <label>审核人</label>
          <input v-model="filters.filterReviewer" type="text" placeholder="请输入" @keyup.enter="applyFilters" />
        </div>
      </div>
      <div class="filter-row">
        <div class="filter-item filter-date">
          <label>创建时间</label>
          <div class="date-range">
            <input v-model="filters.filterCreatedTimeStart" type="date" />
            <span>至</span>
            <input v-model="filters.filterCreatedTimeEnd" type="date" />
          </div>
        </div>
        <div class="filter-item filter-date">
          <label>反馈时间</label>
          <div class="date-range">
            <input v-model="filters.filterFeedbackTimeStart" type="date" />
            <span>至</span>
            <input v-model="filters.filterFeedbackTimeEnd" type="date" />
          </div>
        </div>
        <div class="filter-actions">
          <button class="btn btn-primary btn-sm" @click="applyFilters">筛选</button>
          <button class="btn btn-outline btn-sm" @click="resetFilters">重置</button>
        </div>
      </div>
    </div>

    <div class="table-container">
      <table class="data-table" v-if="tasks.length > 0">
        <thead>
          <tr>
            <th>任务编号</th>
            <th>编码数量</th>
            <th>审核人</th>
            <th>审核日期</th>
            <th>创建人</th>
            <th>创建时间</th>
            <th>状态</th>
            <th>操作</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="task in tasks" :key="task.taskCode">
            <td class="task-code">{{ task.taskCode || '--' }}</td>
            <td class="task-count">{{ task.count || 0 }} 条</td>
            <td>{{ task.reviewer || '--' }}</td>
            <td class="task-date">{{ formatDate(task.reviewDate) }}</td>
            <td>{{ task.creator || '--' }}</td>
            <td class="task-date">{{ formatDate(task.createdTime) }}</td>
            <td>
              <span class="status-badge" :class="getStatusClass(task.status)">
                {{ task.status || '--' }}
              </span>
            </td>
            <td class="task-action">
              <router-link
                :to="{ name: 'review-detail', params: { id: task.id } }"
                class="btn btn-sm btn-primary"
              >
                查看详情
              </router-link>
            </td>
          </tr>
        </tbody>
      </table>

      <div class="loading-state" v-if="loading">
        <p>加载中...</p>
      </div>

      <div class="empty-state" v-else-if="tasks.length === 0">
        <p>暂无任务数据</p>
      </div>
    </div>

    <div class="pagination" v-if="pagination.total > 0">
      <div class="pagination-left">
        <span class="total-info">共 {{ pagination.total }} 个任务</span>
      </div>
      <div class="pagination-center">
        <button class="page-btn" :disabled="pagination.pageIndex <= 1" @click="loadTaskList(pagination.pageIndex - 1)">上一页</button>
        <span class="page-info">第 {{ pagination.pageIndex }} 页 / 共 {{ pagination.totalPages }} 页</span>
        <button class="page-btn" :disabled="pagination.pageIndex >= pagination.totalPages" @click="loadTaskList(pagination.pageIndex + 1)">下一页</button>
      </div>
      <div class="pagination-right">
        <span class="page-size-label">每页</span>
        <select class="page-size-select" v-model="pagination.pageSize" @change="onPageSizeChange">
          <option :value="10">10</option>
          <option :value="20">20</option>
          <option :value="50">50</option>
          <option :value="100">100</option>
        </select>
        <span class="page-size-label">条</span>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, inject } from 'vue'
import { getTaskList } from '../api/h3yun'

const showToast = inject('showToast')

const tasks = ref([])
const loading = ref(false)
const showFilters = ref(false)

const pagination = reactive({
  pageIndex: 1,
  pageSize: 20,
  total: 0,
  totalPages: 0
})

const filters = reactive({
  filterTaskCode: '',
  filterReviewer: '',
  filterCreatedTimeStart: '',
  filterCreatedTimeEnd: '',
  filterFeedbackTimeStart: '',
  filterFeedbackTimeEnd: ''
})

const hasActiveFilters = computed(() => Object.values(filters).some(v => v !== ''))

function formatDate(dateStr) {
  if (!dateStr) return '--'
  return String(dateStr).replace('T', ' ').replace(/\//g, '-')
}

function getStatusClass(status) {
  if (status === '进行中') return 'status-processing'
  if (status === '已完成') return 'status-success'
  if (status === '已核对') return 'status-success'
  if (status === '待核对') return 'status-processing'
  if (status === '草稿') return 'status-draft'
  return 'status-default'
}

function buildNormalizedFilters() {
  const nextFilters = { ...filters }

  if (nextFilters.filterCreatedTimeStart && !nextFilters.filterCreatedTimeEnd) {
    nextFilters.filterCreatedTimeEnd = nextFilters.filterCreatedTimeStart
  } else if (!nextFilters.filterCreatedTimeStart && nextFilters.filterCreatedTimeEnd) {
    nextFilters.filterCreatedTimeStart = nextFilters.filterCreatedTimeEnd
  }

  if (nextFilters.filterFeedbackTimeStart && !nextFilters.filterFeedbackTimeEnd) {
    nextFilters.filterFeedbackTimeEnd = nextFilters.filterFeedbackTimeStart
  } else if (!nextFilters.filterFeedbackTimeStart && nextFilters.filterFeedbackTimeEnd) {
    nextFilters.filterFeedbackTimeStart = nextFilters.filterFeedbackTimeEnd
  }

  return nextFilters
}

async function loadTaskList(pageIndex = 1) {
  loading.value = true
  try {
    const result = await getTaskList({
      pageIndex,
      pageSize: pagination.pageSize,
      filters: buildNormalizedFilters()
    })
    tasks.value = result.data || []
    pagination.pageIndex = result.pageIndex || pageIndex
    pagination.total = result.total || 0
    pagination.totalPages = result.totalPages || 0
  } catch (error) {
    console.error('加载任务列表失败:', error)
    showToast?.(`加载失败: ${error.message}`, 'error')
  } finally {
    loading.value = false
  }
}

function applyFilters() {
  loadTaskList(1)
}

function resetFilters() {
  filters.filterTaskCode = ''
  filters.filterReviewer = ''
  filters.filterCreatedTimeStart = ''
  filters.filterCreatedTimeEnd = ''
  filters.filterFeedbackTimeStart = ''
  filters.filterFeedbackTimeEnd = ''
  loadTaskList(1)
}

function refreshTaskList() {
  loadTaskList(1)
}

function onPageSizeChange() {
  loadTaskList(1)
}

onMounted(() => {
  loadTaskList(1)
})
</script>

<style scoped>
.review-tasks-view {
  display: flex;
  flex-direction: column;
  height: 100%;
  background: var(--bg-secondary);
}

.toolbar {
  padding: 16px 24px;
  background: var(--bg-primary);
  border-bottom: 1px solid var(--border-color);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-shrink: 0;
}

.toolbar-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.page-title {
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.filter-bar {
  padding: 16px 24px;
  background: var(--bg-primary);
  border-bottom: 1px solid var(--border-color);
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.filter-row {
  display: flex;
  gap: 16px;
  align-items: flex-end;
  flex-wrap: wrap;
}

.filter-item {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 220px;
}

.filter-item label {
  font-size: 13px;
  color: var(--text-secondary);
}

.filter-item input {
  height: 36px;
  padding: 0 12px;
  border: 1px solid var(--border-color);
  border-radius: 6px;
  background: var(--bg-primary);
  color: var(--text-primary);
  font-size: 14px;
}

.filter-date {
  min-width: 360px;
}

.date-range {
  display: flex;
  align-items: center;
  gap: 8px;
}

.date-range input {
  flex: 1;
}

.date-range span {
  color: var(--text-secondary);
  font-size: 13px;
}

.filter-actions {
  display: flex;
  gap: 8px;
  margin-left: auto;
}

.table-container {
  flex: 1;
  overflow: auto;
  padding: 24px;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  background: var(--bg-primary);
  border-radius: 8px;
  overflow: hidden;
  box-shadow: var(--shadow-sm);
}

.data-table th,
.data-table td {
  padding: 14px 16px;
  text-align: left;
  border-bottom: 1px solid var(--border-color);
  vertical-align: middle;
}

.data-table th {
  background: var(--bg-tertiary);
  font-weight: 600;
  color: var(--text-secondary);
  font-size: 14px;
  white-space: nowrap;
}

.data-table td {
  font-size: 14px;
  color: var(--text-primary);
}

.data-table tbody tr:hover td {
  background: var(--bg-secondary);
}

.task-code {
  font-weight: 600;
  color: var(--primary);
}

.task-count {
  font-weight: 500;
}

.task-date {
  color: var(--text-tertiary);
  white-space: nowrap;
}

.status-badge {
  display: inline-flex;
  align-items: center;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 600;
}

.status-processing {
  background: rgba(37, 99, 235, 0.12);
  color: #2563eb;
}

.status-success {
  background: rgba(22, 163, 74, 0.12);
  color: #16a34a;
}

.status-draft {
  background: rgba(107, 114, 128, 0.12);
  color: #6b7280;
}

.status-default {
  background: rgba(148, 163, 184, 0.15);
  color: var(--text-secondary);
}

.pagination {
  padding: 12px 24px;
  background: var(--bg-primary);
  border-top: 1px solid var(--border-color);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-shrink: 0;
}

.pagination-left,
.pagination-right {
  flex: 1;
}

.pagination-right {
  display: flex;
  justify-content: flex-end;
  align-items: center;
}

.pagination-center {
  display: flex;
  align-items: center;
  gap: 12px;
}

.total-info,
.page-info,
.page-size-label {
  font-size: 13px;
  color: var(--text-secondary);
}

.page-btn {
  padding: 6px 12px;
  border: 1px solid var(--border-color);
  background: var(--bg-primary);
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
  color: var(--text-primary);
  transition: all 0.2s;
}

.page-btn:hover:not(:disabled) {
  border-color: var(--primary);
  color: var(--primary);
}

.page-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.page-size-select {
  padding: 4px 8px;
  border: 1px solid var(--border-color);
  border-radius: 4px;
  font-size: 13px;
  background: var(--bg-primary);
  color: var(--text-primary);
  cursor: pointer;
  margin: 0 4px;
}

.empty-state,
.loading-state {
  padding: 80px 40px;
  text-align: center;
  color: var(--text-tertiary);
  font-size: 14px;
}

.btn {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.2s;
  text-decoration: none;
  display: inline-block;
}

.btn-sm {
  padding: 6px 12px;
  font-size: 13px;
}

.btn-outline {
  background: transparent;
  border: 1px solid var(--border-color);
  color: var(--text-primary);
}

.btn-outline:hover:not(:disabled) {
  border-color: var(--primary);
  color: var(--primary);
}

.btn-primary {
  background: var(--primary);
  color: white;
}

.btn-primary:hover {
  opacity: 0.9;
}

.btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
</style>
