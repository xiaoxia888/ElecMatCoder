import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '@/lib/api'
import { clamp } from '@/lib/utils'
import { getDifficultyLabel } from '@/lib/formatters'
import { resolveProjectName } from '@/lib/import-export'
import type { BatchJobEvent, BatchJobSummary, EncodingResult, ImportedRow, TaskInfo } from '@/types/encoding'

function normalizeImportedRows(
  rows: Record<string, unknown>[],
  column: string,
  projectColumn?: string,
): ImportedRow[] {
  return rows
    .map((row, index) => ({
      index,
      text: String(row[column] ?? '').trim(),
      projectName: projectColumn
        ? String(row[projectColumn] ?? '').trim()
        : resolveProjectName(row),
      rawRow: row,
    }))
    .filter((item) => item.text)
}

function isBatchJobRunning(status: string | undefined) {
  return ['queued', 'running', 'cancelling'].includes(String(status || ''))
}

export function useEncodingWorkspace() {
  const [dataList, setDataList] = useState<ImportedRow[]>([])
  const [results, setResults] = useState<Record<number, EncodingResult>>({})
  const [taskName, setTaskName] = useState('')
  const [localDataList, setLocalDataList] = useState<ImportedRow[]>([])
  const [localResults, setLocalResults] = useState<Record<number, EncodingResult>>({})
  const [localTaskName, setLocalTaskName] = useState('')
  const [localCurrentIndex, setLocalCurrentIndex] = useState(-1)
  const [jobs, setJobs] = useState<BatchJobSummary[]>([])
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null)
  const [currentIndex, setCurrentIndex] = useState(-1)
  const [filter, setFilter] = useState<'all' | 'review' | 'hard'>('all')
  const [defaultMaxConcurrent, setDefaultMaxConcurrent] = useState(2)
  const [maxConcurrent, setMaxConcurrent] = useState(2)
  const [isEncodingSingle, setIsEncodingSingle] = useState(false)
  const [isEncodingBatch, setIsEncodingBatch] = useState(false)
  const [isStoppingBatch, setIsStoppingBatch] = useState(false)
  const [isTaskLoading, setIsTaskLoading] = useState(false)
  const [activeJob, setActiveJob] = useState<BatchJobSummary | null>(null)
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')
  const eventSourceRef = useRef<EventSource | null>(null)
  const activeTaskIdRef = useRef<string | null>(null)

  useEffect(() => {
    activeTaskIdRef.current = activeTaskId
  }, [activeTaskId])

  useEffect(() => {
    let mounted = true
    api
      .getConfig()
      .then((config) => {
        if (!mounted) return
        const nextDefault = clamp(Number(config.batch_processing?.max_concurrent || 2), 1, 16)
        setDefaultMaxConcurrent(nextDefault)
        const saved = Number(window.localStorage.getItem('encoding_max_concurrent'))
        setMaxConcurrent(Number.isFinite(saved) && saved >= 1 ? clamp(saved, 1, 16) : nextDefault)
      })
      .catch(() => {
        if (!mounted) return
        const saved = Number(window.localStorage.getItem('encoding_max_concurrent'))
        setMaxConcurrent(Number.isFinite(saved) && saved >= 1 ? clamp(saved, 1, 16) : 2)
      })
    // 仅拉取任务列表用于展示，不自动加载任何任务的数据（保持页面空白，需手动点击任务）
    api
      .listBatchJobs()
      .then((res) => {
        if (mounted) setJobs(res.jobs || [])
      })
      .catch(() => undefined)
    return () => {
      mounted = false
      eventSourceRef.current?.close()
    }
  }, [])

  const filteredDataList = useMemo(() => {
    if (filter === 'all') return dataList
    if (filter === 'review') {
      return dataList.filter((item) => results[item.index]?.need_review)
    }
    return dataList.filter((item) => getDifficultyLabel(results[item.index]) === '困难')
  }, [dataList, filter, results])

  const currentItem = currentIndex >= 0 ? dataList[currentIndex] || null : null
  const currentResult = currentItem ? results[currentItem.index] : null

  const stats = useMemo(() => {
    const values = Object.values(results)
    return {
      total: values.length,
      // 只要编码成功就算成功（含需审核）
      success: values.filter((item) => item.success).length,
      review: values.filter((item) => item.need_review).length,
    }
  }, [results])

  const progress = useMemo(() => {
    if (!activeJob?.total) return 0
    return Math.max(0, Math.min(100, Math.round((Number(activeJob.processed || 0) / Number(activeJob.total || 1)) * 100)))
  }, [activeJob])

  // 任务列表：服务端任务 + 本地已导入未提交的任务，点击后才加载数据
  const tasks = useMemo<TaskInfo[]>(() => {
    const list: TaskInfo[] = jobs.map((job) => {
      const total = Number(job.total || 0)
      const processed = Number(job.processed || 0)
      const st = String(job.status || '')
      const running = ['queued', 'running', 'cancelling'].includes(st)
      const status: TaskInfo['status'] = running ? 'running' : processed === 0 ? 'idle' : processed >= total && total > 0 ? 'done' : 'partial'
      return {
        id: job.job_id,
        name: `批量任务 ${job.job_id.slice(0, 6)}`,
        total,
        // 计数来自服务端汇总字段（列表接口不再下发完整 results）
        success: Number(job.success_count || 0),
        review: Number(job.review_count || 0),
        progress: total > 0 ? Math.round((processed / total) * 100) : 0,
        status,
        durationSeconds: Number.isFinite(Number(job.duration_seconds)) ? Number(job.duration_seconds) : null,
      }
    })

    if (localDataList.length > 0) {
      const localValues = Object.values(localResults)
      const localDone = localValues.length
      list.unshift({
        id: 'local',
        name: localTaskName || '当前导入',
        total: localDataList.length,
        success: localValues.filter((item) => item.success).length,
        review: localValues.filter((item) => item.need_review).length,
        progress: localDataList.length > 0 ? Math.round((localDone / localDataList.length) * 100) : 0,
        status: localDone === 0 ? 'idle' : localDone >= localDataList.length ? 'done' : 'partial',
        durationSeconds: null,
      })
    }

    const runningTaskId = activeJob?.job_id || null
    if (runningTaskId) {
      const idx = list.findIndex((t) => t.id === runningTaskId)
      const runningTotal = Number(activeJob?.total || 0)
      const runningProcessed = Number(activeJob?.processed || 0)
      const runningSuccess = Number(activeJob?.success_count || 0)
      const runningReview = Number(activeJob?.review_count || 0)
      const runningProgress = runningTotal > 0 ? Math.round((runningProcessed / runningTotal) * 100) : 0
      if (idx >= 0) {
        if (isEncodingBatch) {
          list[idx] = {
            ...list[idx],
            total: runningTotal || list[idx].total,
            success: runningSuccess,
            review: runningReview,
            progress: runningProgress,
            status: 'running',
            durationSeconds: Number.isFinite(Number(activeJob?.duration_seconds)) ? Number(activeJob?.duration_seconds) : null,
          }
        }
      } else if (activeTaskId === runningTaskId) {
        // 本地导入、尚未提交为后端任务
        const total = runningTotal || dataList.length
        const done = runningProcessed || stats.total
        list.unshift({
          id: runningTaskId,
          name: taskName || '当前导入',
          total,
          success: runningSuccess || stats.success,
          review: runningReview || stats.review,
          progress: isEncodingBatch ? runningProgress : total > 0 ? Math.round((done / total) * 100) : 0,
          status: isEncodingBatch ? 'running' : done === 0 ? 'idle' : done >= total && total > 0 ? 'done' : 'partial',
          durationSeconds: Number.isFinite(Number(activeJob?.duration_seconds)) ? Number(activeJob?.duration_seconds) : null,
        })
      }
    }
    // 是否展示中途停止/失败的任务由后端配置（batch_processing.show_terminated_jobs）决定，
    // 前端直接展示后端下发的任务，不再额外过滤
    return list
  }, [jobs, activeTaskId, activeJob?.job_id, dataList.length, localDataList.length, localResults, localTaskName, stats, isEncodingBatch, progress, taskName])

  function clearStream() {
    eventSourceRef.current?.close()
    eventSourceRef.current = null
  }

  function applyRunningJobSnapshot(job: BatchJobSummary) {
    const running = isBatchJobRunning(job.status)
    setActiveJob(job)
    setIsEncodingBatch(running)
    if (!running) {
      setIsStoppingBatch(false)
    }
  }

  function applyViewedJobSnapshot(job: BatchJobSummary) {
    const items = Array.isArray(job.items) ? job.items : []
    setDataList(
      items
        .map((item, idx) => ({
          index: Number.isFinite(Number(item.index)) ? Number(item.index) : idx,
          text: item.text || '',
          projectName: item.project_name || '',
          rawRow: {},
        }))
        .sort((a, b) => a.index - b.index),
    )

    const nextResults: Record<number, EncodingResult> = {}
    Object.entries(job.results || {}).forEach(([index, result]) => {
      const numeric = Number(index)
      if (Number.isFinite(numeric) && result) nextResults[numeric] = result
    })
    setResults(nextResults)
  }

  async function hydrateTaskFromServer(jobId: string, preferredIndex?: number) {
    const res = await api.getBatchJob(jobId)
    if (!res.job || activeTaskIdRef.current !== jobId) return
    applyViewedJobSnapshot(res.job)
    setCurrentIndex((prev) => {
      const total = res.job.items?.length ?? 0
      if (total <= 0) return -1
      const nextIndex = typeof preferredIndex === 'number' ? preferredIndex : prev
      return Number.isFinite(nextIndex) && nextIndex >= 0 && nextIndex < total ? nextIndex : 0
    })
  }

  function subscribeBatchJob(jobId: string) {
    clearStream()
    const source = new EventSource(`/api/pipe/encode/batch/jobs/${jobId}/stream`)
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as BatchJobEvent
      if (event.type === 'snapshot' && event.snapshot) {
        applyRunningJobSnapshot(event.snapshot)
        if (activeTaskIdRef.current === jobId && event.snapshot.results && Object.keys(event.snapshot.results).length > 0) {
          applyViewedJobSnapshot(event.snapshot)
        }
        return
      }
      if (typeof event.index === 'number' && event.result && activeTaskIdRef.current === jobId) {
        setResults((prev) => ({ ...prev, [event.index!]: event.result! }))
      }
      if (event.snapshot) {
        applyRunningJobSnapshot(event.snapshot)
      }
      if (event.type === 'end' || event.type === 'cancelled' || event.type === 'failed') {
        if (activeTaskIdRef.current === jobId) {
          void hydrateTaskFromServer(jobId)
        }
        setIsEncodingBatch(false)
        setIsStoppingBatch(false)
        if (eventSourceRef.current === source) {
          clearStream()
        }
        refreshJobs()
      }
    }
    eventSourceRef.current = source
  }

  function refreshJobs() {
    api
      .listBatchJobs()
      .then((res) => setJobs(res.jobs || []))
      .catch(() => undefined)
  }

  // 点击任务卡片：加载该任务的数据（运行中则订阅实时进度）
  async function loadTask(id: string) {
    if (id === 'local') {
      setIsTaskLoading(false)
      setActiveTaskId('local')
      setTaskName(localTaskName)
      setDataList(localDataList)
      setResults(localResults)
      setCurrentIndex(localCurrentIndex)
      return
    }
    if (id === activeTaskId && (dataList.length > 0 || Object.keys(results).length > 0)) return
    setIsTaskLoading(true)
    setActiveTaskId(id)
    try {
      const res = await api.getBatchJob(id)
      if (!res.job || activeTaskIdRef.current !== id) return
      applyViewedJobSnapshot(res.job)
      setCurrentIndex((res.job.items?.length ?? 0) > 0 ? 0 : -1)
      if (isBatchJobRunning(res.job.status)) {
        applyRunningJobSnapshot(res.job)
        if (activeJob?.job_id !== id || !eventSourceRef.current) {
          subscribeBatchJob(id)
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载任务失败')
    } finally {
      if (activeTaskIdRef.current === id) {
        setIsTaskLoading(false)
      }
    }
  }

  function importRows(
    rows: Record<string, unknown>[],
    column: string,
    fileName = '',
    projectColumn = '',
  ) {
    const normalized = normalizeImportedRows(rows, column, projectColumn || undefined)
    const nextIndex = normalized.length > 0 ? 0 : -1
    setLocalDataList(normalized)
    setLocalResults({})
    setLocalTaskName(fileName)
    setLocalCurrentIndex(nextIndex)
    setDataList(normalized)
    setResults({})
    setTaskName(fileName)
    setActiveTaskId('local')
    setCurrentIndex(nextIndex)
    setActiveJob(null)
    setNotice(`已导入 ${normalized.length} 条数据`)
    setError('')
    clearStream()
    setIsTaskLoading(false)
  }

  async function encodeCurrentItem() {
    if (!currentItem) return
    setIsEncodingSingle(true)
    setError('')
    try {
      const result = await api.encodeSingle({
        text: currentItem.text,
        preprocess: true,
        project_name: currentItem.projectName || '',
      })
      setResults((prev) => ({ ...prev, [currentItem.index]: result }))
      if (activeTaskId === 'local') {
        setLocalResults((prev) => ({ ...prev, [currentItem.index]: result }))
      }
      setNotice(result.need_review ? '当前样本已编码，结果进入待审核。' : '当前样本编码完成。')
    } catch (err) {
      setError(err instanceof Error ? err.message : '单条编码失败')
    } finally {
      setIsEncodingSingle(false)
    }
  }

  async function createBatchJob() {
    if (dataList.length === 0) return
    setIsEncodingBatch(true)
    setError('')
    // 新任务从零开始，清掉上一个任务可能残留的结果，避免计数/展示串台
    setResults({})
    try {
      const job = await api.createBatchJob({
        items: dataList.map((item) => ({
          client_index: item.index,
          text: item.text,
          project_name: item.projectName,
          preprocess: true,
        })),
        max_concurrent: maxConcurrent,
      })
      if (job.job?.job_id) {
        setLocalDataList([])
        setLocalResults({})
        setLocalTaskName('')
        setLocalCurrentIndex(-1)
        setActiveTaskId(job.job.job_id)
        applyRunningJobSnapshot(job.job)
        applyViewedJobSnapshot(job.job)
        subscribeBatchJob(job.job.job_id)
        refreshJobs()
        setNotice(`批量任务已创建，任务号 ${job.job.job_id.slice(0, 8)}。`)
      }
    } catch (err) {
      setIsEncodingBatch(false)
      setError(err instanceof Error ? err.message : '批量任务创建失败')
    }
  }

  async function cancelBatchJob() {
    if (!activeJob?.job_id) return false
    setIsStoppingBatch(true)
    setError('')
    try {
      const res = await api.cancelBatchJob(activeJob.job_id)
      applyRunningJobSnapshot(res.job)
      if (!isBatchJobRunning(res.job?.status)) {
        setIsEncodingBatch(false)
        setIsStoppingBatch(false)
        refreshJobs()
      }
      setNotice('已提交停止请求。')
      return true
    } catch (err) {
      setError(err instanceof Error ? err.message : '停止任务失败')
      setIsStoppingBatch(false)
      return false
    }
  }

  async function selectItem(index: number) {
    setCurrentIndex(index)
    if (activeTaskId === 'local') {
      setLocalCurrentIndex(index)
    }
    // 服务端任务：点击描述时按需查询该条结果（F12 可独立查看该条请求/响应），并写回缓存
    if (activeTaskId && activeTaskId !== 'local') {
      try {
        const detail = await api.getBatchJobItem(activeTaskId, index)
        if (detail.result) {
          setResults((prev) => ({ ...prev, [index]: detail.result! }))
        }
      } catch {
        // 单条查询失败不影响切换
      }
    }
  }

  function goRelative(delta: number) {
    if (filteredDataList.length === 0) return
    const pos = filteredDataList.findIndex((item) => item.index === currentIndex)
    const base = pos < 0 ? 0 : pos
    const nextPos = Math.max(0, Math.min(filteredDataList.length - 1, base + delta))
    void selectItem(filteredDataList[nextPos].index)
  }
  const goPrev = () => goRelative(-1)
  const goNext = () => goRelative(1)

  // 跳转到第 position 条（1 基，基于全部数据，超出范围自动夹紧）
  function goTo(position: number) {
    if (dataList.length === 0) return
    const clamped = Math.max(1, Math.min(dataList.length, Math.floor(position)))
    void selectItem(dataList[clamped - 1].index)
  }

  // 键盘左右方向键切换上一条/下一条
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      const target = event.target
      if (target instanceof HTMLElement && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return
      if (event.key === 'ArrowLeft') {
        event.preventDefault()
        goRelative(-1)
      } else if (event.key === 'ArrowRight') {
        event.preventDefault()
        goRelative(1)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filteredDataList, currentIndex, activeJob])

  function updateMaxConcurrent(value: number) {
    const nextValue = clamp(value, 1, 16)
    setMaxConcurrent(nextValue)
    window.localStorage.setItem('encoding_max_concurrent', String(nextValue))
  }

  function getItemStatus(index: number) {
    const result = results[index]
    if (!result) return 'pending'
    if (result.need_review) return 'review'
    if (result.success) return 'success'
    return 'pending'
  }

  function getItemDifficulty(index: number) {
    return getDifficultyLabel(results[index])
  }

  return {
    dataList,
    filteredDataList,
    currentIndex,
    currentItem,
    currentResult,
    results,
    filter,
    setFilter,
    stats,
    defaultMaxConcurrent,
    maxConcurrent,
    updateMaxConcurrent,
    isEncodingSingle,
    isEncodingBatch,
    isStoppingBatch,
    isTaskLoading,
    activeJob,
    progress,
    tasks,
    taskName,
    activeTaskId,
    selectTask: loadTask,
    notice,
    error,
    importRows,
    encodeCurrentItem,
    createBatchJob,
    cancelBatchJob,
    setCurrentIndex: selectItem,
    goPrev,
    goNext,
    goTo,
    getItemStatus,
    getItemDifficulty,
  }
}
