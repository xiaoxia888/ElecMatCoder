import { getDifficultyLabel, getDifficultyVariant, getRouteReason, getRoutingStageText } from '@/lib/formatters'
import type { EncodingResult } from '@/types/encoding'

interface RoutingResultCardProps {
  result: EncodingResult | null
}

const DIFFICULTY_COLOR: Record<string, string> = {
  danger: 'text-danger',
  success: 'text-success',
  caution: 'text-caution',
  neutral: 'text-muted',
}

export function RoutingResultCard({ result }: RoutingResultCardProps) {
  const safeResult = result ?? undefined
  const difficultyColor = DIFFICULTY_COLOR[getDifficultyVariant(safeResult)] ?? 'text-ink'

  return (
    <section className="rounded-xl border border-line bg-white p-4 shadow-panel">
      <h2 className="mb-2.5 text-[15px] font-bold text-ink">分流结果</h2>
      <div className="grid grid-cols-3 gap-3">
        <div className="rounded-xl border border-line px-4 py-2.5">
          <div className="mb-1 text-[13px] text-muted">最终难度</div>
          <div className={`text-[18px] font-bold ${difficultyColor}`}>{getDifficultyLabel(safeResult)}</div>
        </div>
        <div className="rounded-xl border border-line px-4 py-2.5">
          <div className="mb-1 text-[13px] text-muted">判定阶段</div>
          <div className="text-[16px] font-bold text-ink">{getRoutingStageText(safeResult)}</div>
        </div>
        <div className="rounded-xl border border-line px-4 py-2.5">
          <div className="mb-1 text-[13px] text-muted">原因说明</div>
          <div className="text-[14px] leading-relaxed text-ink">{getRouteReason(safeResult)}</div>
        </div>
      </div>
    </section>
  )
}
