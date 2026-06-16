import { Search } from 'lucide-react'
import { useMemo, useState } from 'react'
import { summarizeItemText } from '@/lib/formatters'
import type { ImportedRow } from '@/types/encoding'

interface DataListPanelProps {
  dataList: ImportedRow[]
  allItems: ImportedRow[]
  currentIndex: number
  filter: 'all' | 'review' | 'hard'
  onFilterChange: (value: 'all' | 'review' | 'hard') => void
  onSelect: (index: number) => void
  getItemStatus: (index: number) => string
  getItemDifficulty: (index: number) => string
}

const DIFF_STYLE: Record<string, string> = {
  困难: 'bg-dangerSoft text-danger',
  中等: 'bg-cautionSoft text-caution',
  简单: 'bg-successSoft text-success',
}
const STATUS_STYLE: Record<string, string> = {
  review: 'bg-cautionSoft text-caution',
  success: 'bg-successSoft text-success',
}

export function DataListPanel(props: DataListPanelProps) {
  const { dataList, allItems, currentIndex, filter, onFilterChange, onSelect, getItemStatus, getItemDifficulty } = props
  const [keyword, setKeyword] = useState('')

  const reviewCount = useMemo(() => allItems.filter((item) => getItemStatus(item.index) === 'review').length, [allItems, getItemStatus])
  const hardCount = useMemo(() => allItems.filter((item) => getItemDifficulty(item.index) === '困难').length, [allItems, getItemDifficulty])

  const visibleItems = useMemo(() => {
    const q = keyword.trim().toLowerCase()
    if (!q) return dataList
    return dataList.filter((item) => item.text.toLowerCase().includes(q))
  }, [dataList, keyword])

  const filters: Array<{ key: 'all' | 'review' | 'hard'; label: string; active: string; idle: string }> = [
    { key: 'all', label: `全部(${allItems.length})`, active: 'bg-accent text-white', idle: 'bg-canvas text-muted' },
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

      <div className="-mx-1 min-h-0 flex-1 space-y-0.5 overflow-y-auto px-1">
        {visibleItems.map((item) => {
          const status = getItemStatus(item.index)
          const difficulty = getItemDifficulty(item.index)
          const selected = currentIndex === item.index
          // 未识别（pending）不显示状态；难度同理仅在已识别时显示
          const statusLabel = status === 'review' ? '待审' : status === 'success' ? '已通过' : ''
          const showDifficulty = status !== 'pending' && difficulty && difficulty !== '待定'
          return (
            <button
              key={item.index}
              type="button"
              onClick={() => onSelect(item.index)}
              className={`relative w-full rounded-lg px-3 py-2.5 text-left transition ${selected ? 'bg-accentSoft' : 'hover:bg-[#f5f7fb]'}`}
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
    </section>
  )
}
