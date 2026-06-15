import { useEffect, useMemo, useRef, useState } from 'react'
import { api } from '@/lib/api'
import { clamp } from '@/lib/utils'
import { getDifficultyLabel } from '@/lib/formatters'
import { resolveProjectName } from '@/lib/import-export'
import type { BatchJobEvent, BatchJobSummary, EncodingResult, ImportedRow, TaskInfo } from '@/types/encoding'

function normalizeImportedRows(rows: Record<string, unknown>[], column: string): ImportedRow[] {
  return rows
    .map((row, index) => ({
      index,
      text: String(row[column] ?? '').trim(),
      projectName: resolveProjectName(row),
      rawRow: row,
    }))
    .filter((item) => item.text)
}

export function useEncodingWorkspace() {
  const [dataList, setDataList] = useState<ImportedRow[]>([])
  const [results, setResults] = useState<Record<number, EncodingResult>>({})
  const [taskName, setTaskName] = useState('')
  const [jobs, setJobs] = useState<BatchJobSummary[]>([])
  const [activeTaskId, setActiveTaskId] = useState<string | null>(null)
  const [currentIndex, setCurrentIndex] = useState(-1)
  const [filter, setFilter] = useState<'all' | 'review' | 'hard'>('all')
  const [defaultMaxConcurrent, setDefaultMaxConcurrent] = useState(2)
  const [maxConcurrent, setMaxConcurrent] = useState(2)
  const [isEncodingSingle, setIsEncodingSingle] = useState(false)
  const [isEncodingBatch, setIsEncodingBatch] = useState(false)
  const [isStoppingBatch, setIsStoppingBatch] = useState(false)
  const [activeJob, setActiveJob] = useState<BatchJobSummary | null>(null)
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')
  const eventSourceRef = useRef<EventSource | null>(null)

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
      }
    })

    if (activeTaskId) {
      const idx = list.findIndex((t) => t.id === activeTaskId)
      if (idx >= 0) {
        // 后端已有该任务：仅当本地正在运行该批量任务时，才用实时进度覆盖；
        // 查看已完成/历史任务时保留后端数据，避免切换瞬间本地结果清空导致进度条闪烁
        if (isEncodingBatch) {
          list[idx] = {
            ...list[idx],
            total: dataList.length || list[idx].total,
            success: stats.success,
            review: stats.review,
            progress,
            status: 'running',
          }
        }
      } else {
        // 本地导入、尚未提交为后端任务
        const total = dataList.length
        const done = stats.total
        list.unshift({
          id: activeTaskId,
          name: taskName || '当前导入',
          total,
          success: stats.success,
          review: stats.review,
          progress: isEncodingBatch ? progress : total > 0 ? Math.round((done / total) * 100) : 0,
          status: isEncodingBatch ? 'running' : done === 0 ? 'idle' : done >= total && total > 0 ? 'done' : 'partial',
        })
      }
    }
    // 是否展示中途停止/失败的任务由后端配置（batch_processing.show_terminated_jobs）决定，
    // 前端直接展示后端下发的任务，不再额外过滤
    return list
  }, [jobs, activeTaskId, dataList.length, stats, isEncodingBatch, progress, taskName])

  function clearStream() {
    eventSourceRef.current?.close()
    eventSourceRef.current = null
  }

  function applyJobSnapshot(job: BatchJobSummary) {
    setActiveJob(job)
    setIsEncodingBatch(['queued', 'running', 'cancelling'].includes(String(job.status || '')))

    const items = Array.isArray(job.items) ? job.items : []
    if (items.length > 0) {
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
    }

    const nextResults: Record<number, EncodingResult> = {}
    Object.entries(job.results || {}).forEach(([index, result]) => {
      const numeric = Number(index)
      if (Number.isFinite(numeric) && result) nextResults[numeric] = result
    })
    if (Object.keys(nextResults).length > 0) {
      setResults((prev) => ({ ...prev, ...nextResults }))
    }
  }

  function subscribeBatchJob(jobId: string) {
    clearStream()
    const source = new EventSource(`/api/pipe/encode/batch/jobs/${jobId}/stream`)
    source.onmessage = (message) => {
      const event = JSON.parse(message.data) as BatchJobEvent
      if (event.type === 'snapshot' && event.snapshot) {
        applyJobSnapshot(event.snapshot)
        return
      }
      if (typeof event.index === 'number' && event.result) {
        setResults((prev) => ({ ...prev, [event.index!]: event.result! }))
      }
      if (event.snapshot) {
        setActiveJob(event.snapshot)
        setIsEncodingBatch(['queued', 'running', 'cancelling'].includes(String(event.snapshot.status || '')))
      }
      if (event.type === 'end' || event.type === 'cancelled' || event.type === 'failed') {
        if (event.snapshot) applyJobSnapshot(event.snapshot)
        setIsEncodingBatch(false)
        setIsStoppingBatch(false)
        clearStream()
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
      setActiveTaskId('local')
      return
    }
    if (id === activeTaskId) return
    // 切换任务前：停掉上一个任务的实时流并清空数据，避免跨任务结果串台
    clearStream()
    setActiveTaskId(id)
    setResults({})
    setDataList([])
    setCurrentIndex(-1)
    try {
      const res = await api.getBatchJob(id)
      if (res.job) {
        applyJobSnapshot(res.job)
        setCurrentIndex((res.job.items?.length ?? 0) > 0 ? 0 : -1)
        if (['queued', 'running', 'cancelling'].includes(String(res.job.status || ''))) {
          subscribeBatchJob(id)
        } else {
          clearStream()
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载任务失败')
    }
  }

  function importRows(rows: Record<string, unknown>[], column: string, fileName = '') {
    const normalized = normalizeImportedRows(rows, column)
    setDataList(normalized)
    setResults({})
    setTaskName(fileName)
    setActiveTaskId('local')
    setCurrentIndex(normalized.length > 0 ? 0 : -1)
    setActiveJob(null)
    setNotice(`已导入 ${normalized.length} 条数据`)
    setError('')
    clearStream()
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
        setActiveTaskId(job.job.job_id)
        applyJobSnapshot(job.job)
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
    if (!activeJob?.job_id) return
    setIsStoppingBatch(true)
    setError('')
    try {
      const res = await api.cancelBatchJob(activeJob.job_id)
      setActiveJob(res.job)
      setNotice('已提交停止请求。')
    } catch (err) {
      setError(err instanceof Error ? err.message : '停止任务失败')
    } finally {
      setIsStoppingBatch(false)
    }
  }

  async function selectItem(index: number) {
    setCurrentIndex(index)
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
