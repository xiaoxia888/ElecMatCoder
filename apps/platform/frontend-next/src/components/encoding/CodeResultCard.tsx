import { getTypeCategory } from '@/lib/formatters'
import { formatPercent } from '@/lib/utils'
import type { EncodingResult } from '@/types/encoding'

interface CodeResultCardProps {
  result: EncodingResult | null
}

export function CodeResultCard({ result }: CodeResultCardProps) {
  const percent = Math.max(0, Math.min(100, Number(result?.confidence || 0) * 100))
  const category = getTypeCategory(result)

  return (
    <section className="rounded-xl border border-line bg-white p-4 shadow-panel">
      <div className="grid grid-cols-[1fr_280px_180px] gap-6">
        {/* 编码结果 */}
        <div>
          <h3 className="mb-3 text-[14px] text-muted">编码结果</h3>
          <div className="break-all font-mono text-[26px] font-bold tracking-tight text-ink">{result?.final_code || '—'}</div>
        </div>

        {/* 总置信度 */}
        <div className="border-l border-[#eef1f6] pl-6">
          <h3 className="mb-3 text-[14px] text-muted">总置信度</h3>
          <div className="mb-3 text-[26px] font-bold leading-none text-ink">{result ? formatPercent(result.confidence) : '—'}</div>
          <div className="h-1.5 w-full overflow-hidden rounded-full bg-[#eaeef5]">
            <div className="h-full rounded-full bg-accent" style={{ width: `${result ? percent : 0}%` }} />
          </div>
        </div>

        {/* 分类 */}
        <div className="border-l border-[#eef1f6] pl-6">
          <h3 className="mb-3 text-[14px] text-muted">分类</h3>
          <span
            className={`inline-block rounded-md px-2 py-1 text-[13px] font-medium ${
              category ? 'bg-[#eaf1ff] text-accent' : 'bg-[#f4f6fb] text-muted'
            }`}
          >
            {category || '—'}
          </span>
        </div>
      </div>
    </section>
  )
}
