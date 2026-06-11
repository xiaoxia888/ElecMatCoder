import { ChevronLeft, ChevronRight } from 'lucide-react'
import type { EncodingResult, ImportedRow } from '@/types/encoding'

interface DescriptionCardProps {
  currentItem: ImportedRow | null
  currentResult: EncodingResult | null
  dataCount: number
  currentIndex: number
  isEncodingSingle: boolean
  onReencode: () => void
  onPrev: () => void
  onNext: () => void
}

export function DescriptionCard({ currentItem, currentResult, dataCount, currentIndex, isEncodingSingle, onReencode, onPrev, onNext }: DescriptionCardProps) {
  const rawText = currentItem?.text || ''
  const processed = currentResult?.processed_text || ''
  // 格式化描述与原始描述一致时不重复展示
  const processedText = processed && processed !== rawText ? processed : ''

  return (
    <section className="rounded-xl border border-line bg-white p-4 shadow-panel">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-[15px] font-bold text-ink">描述信息</h2>
        <div className="flex items-center gap-3">
          {dataCount > 0 && (
            <div className="flex items-center gap-1 text-[13px] text-muted">
              <button
                type="button"
                onClick={onPrev}
                disabled={!currentItem}
                className="grid h-7 w-7 place-items-center rounded-md border border-line transition hover:bg-canvas disabled:opacity-40"
                title="上一条 (←)"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              <span className="px-1 tabular-nums">
                #{currentIndex + 1} / {dataCount}
              </span>
              <button
                type="button"
                onClick={onNext}
                disabled={!currentItem}
                className="grid h-7 w-7 place-items-center rounded-md border border-line transition hover:bg-canvas disabled:opacity-40"
                title="下一条 (→)"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          )}
          <button
            type="button"
            onClick={onReencode}
            disabled={isEncodingSingle || !currentItem}
            className="rounded-lg border border-accent px-4 py-1.5 text-[13px] font-medium text-accent transition hover:bg-accentSoft/40 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isEncodingSingle ? '识别中…' : currentResult ? '重新编码' : '识别当前'}
          </button>
        </div>
      </div>
      <div className="space-y-3 text-[14px]">
        <div className="flex gap-8">
          <span className="w-20 shrink-0 text-muted">原始描述</span>
          <span className="break-all font-mono">{currentItem?.text || '请选择一条数据'}</span>
        </div>
        <div className="border-t border-[#eef1f6]" />
        <div className="flex gap-8">
          <span className="w-20 shrink-0 text-muted">格式化描述</span>
          <span className="break-all font-mono">{processedText || '—'}</span>
        </div>
      </div>
    </section>
  )
}
