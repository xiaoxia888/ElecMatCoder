import { Search } from 'lucide-react'
import { memo, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react'
import { getDifficultyLabel, summarizeItemText } from '@/lib/formatters'
import type { EncodingResult, ImportedRow } from '@/types/encoding'

interface DataListPanelProps {
  dataList: ImportedRow[]
  results: Record<number, EncodingResult>
  totalCount: number
  reviewCount: number
  hardCount: number
  listKey: string
  currentIndex: number
  filter: 'all' | 'review' | 'hard'
  onFilterChange: (value: 'all' | 'review' | 'hard') => void
  onSelect: (index: number) => void
}

const DIFF_STYLE: Record<string, string> = {
  困难: 'bg-dangerSoft text-danger',
  简单: 'bg-successSoft text-success',
}
const STATUS_STYLE: Record<string, string> = {
  review: 'bg-cautionSoft text-caution',
  success: 'bg-successSoft text-success',
  skipped: 'bg-canvas text-muted',
  failed: 'bg-dangerSoft text-danger',
}

const ROW_HEIGHT = 72
const OVERSCAN = 8

function getItemStatus(result?: EncodingResult) {
  if (!result) return 'pending'
  if (!result.success) return 'failed'
  if (result.skipped_encoding) return 'skipped'
  return result.need_review ? 'review' : 'success'
}

export const DataListPanel = memo(function DataListPanel(props: DataListPanelProps) {
  const { dataList, results, totalCount, reviewCount, hardCount, listKey, currentIndex, filter, onFilterChange, onSelect } = props
  const [keyword, setKeyword] = useState('')
  const deferredKeyword = useDeferredValue(keyword)
  const viewportRef = useRef<HTMLDivElement | null>(null)
  const animationFrameRef = useRef<number | null>(null)
  const latestScrollTopRef = useRef(0)
  const [scrollTop, setScrollTop] = useState(0)
  const [viewportHeight, setViewportHeight] = useState(0)

  const visibleItems = useMemo(() => {
    const q = deferredKeyword.trim().toLowerCase()
    if (!q) return dataList
    return dataList.filter((item) => item.text.toLowerCase().includes(q))
  }, [dataList, deferredKeyword])

  useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport) return
    const updateHeight = () => setViewportHeight(viewport.clientHeight)
    updateHeight()
    const observer = new ResizeObserver(updateHeight)
    observer.observe(viewport)
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport) return
    viewport.scrollTop = 0
    setScrollTop(0)
  }, [listKey, deferredKeyword, filter])

  const selectedPosition = useMemo(
    () => visibleItems.findIndex((item) => item.index === currentIndex),
    [currentIndex, visibleItems],
  )

  useEffect(() => {
    const viewport = viewportRef.current
    if (!viewport || selectedPosition < 0) return
    const rowTop = selectedPosition * ROW_HEIGHT
    const rowBottom = rowTop + ROW_HEIGHT
    if (rowTop < viewport.scrollTop) {
      viewport.scrollTop = rowTop
    } else if (rowBottom > viewport.scrollTop + viewport.clientHeight) {
      viewport.scrollTop = rowBottom - viewport.clientHeight
    }
  }, [selectedPosition])

  const startIndex = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN)
  const endIndex = Math.min(
    visibleItems.length,
    Math.ceil((scrollTop + viewportHeight) / ROW_HEIGHT) + OVERSCAN,
  )
  const renderedItems = visibleItems.slice(startIndex, endIndex)

  function handleScroll(event: React.UIEvent<HTMLDivElement>) {
    latestScrollTopRef.current = event.currentTarget.scrollTop
    if (animationFrameRef.current !== null) return
    animationFrameRef.current = window.requestAnimationFrame(() => {
      animationFrameRef.current = null
      setScrollTop(latestScrollTopRef.current)
    })
  }

  useEffect(() => () => {
    if (animationFrameRef.current !== null) window.cancelAnimationFrame(animationFrameRef.current)
  }, [])

  const filters: Array<{ key: 'all' | 'review' | 'hard'; label: string; active: string; idle: string }> = [
    { key: 'all', label: `全部(${totalCount})`, active: 'bg-accent text-white', idle: 'bg-canvas text-muted' },
    { key: 'review', label: `待审(${reviewCount})`, active: 'bg-caution text-white', idle: 'bg-cautionSoft text-caution' },
    { key: 'hard', label: `困难(${hardCount})`, active: 'bg-danger text-white', idle: 'bg-dangerSoft text-danger' },
  ]

  return (
    <section className="flex h-full min-h-0 flex-col rounded-xl border border-line bg-white p-4 shadow-panel">
      <h2 className="mb-3 text-[15px] font-bold text-ink">数据列表</h2>

      <div className="mb-3 flex items-center gap-2">
        {filters.map((f) => (
          <button
            key={f.key}
            type="button"
            onClick={() => onFilterChange(f.key)}
            className={`rounded-md px-2.5 py-1 text-[12px] font-medium transition ${filter === f.key ? f.active : f.idle}`}
          >
            {f.label}
          </button>
        ))}
      </div>

      <div className="relative mb-2">
        <input
          value={keyword}
          onChange={(event) => setKeyword(event.target.value)}
          placeholder="搜索描述或编码结果"
          className="h-10 w-full rounded-lg border border-line bg-[#fafcff] pl-3.5 pr-9 text-[13px] outline-none focus:border-accent"
        />
        <Search className="pointer-events-none absolute right-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
      </div>

      <div ref={viewportRef} onScroll={handleScroll} className="-mx-1 min-h-0 flex-1 overflow-y-auto px-1">
        <div className="relative" style={{ height: visibleItems.length * ROW_HEIGHT }}>
        {renderedItems.map((item, offset) => {
          const result = results[item.index]
          const status = getItemStatus(result)
          const difficulty = getDifficultyLabel(result)
          const selected = currentIndex === item.index
          // 未识别（pending）不显示状态；难度同理仅在已识别时显示
          const statusLabel = status === 'review'
            ? '待审'
            : status === 'success'
              ? '已通过'
              : status === 'skipped'
                ? '已跳过'
                : status === 'failed'
                  ? '运行失败'
                  : ''
          const showDifficulty = status !== 'pending' && difficulty && difficulty !== '待定'
          return (
            <button
              key={item.index}
              type="button"
              onClick={() => onSelect(item.index)}
              className={`absolute left-0 right-0 rounded-lg px-3 py-2.5 text-left transition ${selected ? 'bg-accentSoft' : 'hover:bg-[#f5f7fb]'}`}
              style={{ top: (startIndex + offset) * ROW_HEIGHT, height: ROW_HEIGHT - 2 }}
            >
              {selected && <span className="absolute bottom-2 left-0 top-2 w-[3px] rounded-full bg-accent" />}
              <div className="flex items-start gap-2">
                <span className="mt-0.5 shrink-0 text-[13px] text-muted">#{item.index + 1}</span>
                <span className="line-clamp-2 flex-1 text-[13px] leading-snug text-ink">{summarizeItemText(item.text, 64)}</span>
                <span className="flex shrink-0 flex-col items-end gap-1">
                  {showDifficulty && (
                    <span className={`rounded-md px-2 py-0.5 text-[12px] font-medium ${DIFF_STYLE[difficulty] ?? 'bg-cautionSoft text-caution'}`}>{difficulty}</span>
                  )}
                  {statusLabel && (
                    <span className={`rounded-md px-2 py-0.5 text-[12px] font-medium ${STATUS_STYLE[status] ?? 'bg-canvas text-muted'}`}>{statusLabel}</span>
                  )}
                </span>
              </div>
            </button>
          )
        })}
        </div>
      </div>
    </section>
  )
})
