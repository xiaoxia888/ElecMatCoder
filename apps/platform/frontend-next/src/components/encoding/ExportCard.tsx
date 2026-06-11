import { Database, FileSpreadsheet, FileText } from 'lucide-react'
import { exportResultsToCsv, exportResultsToExcel, exportStage1Dataset } from '@/lib/import-export'
import type { EncodingResult, ImportedRow } from '@/types/encoding'

interface ExportCardProps {
  dataList: ImportedRow[]
  results: Record<number, EncodingResult>
}

export function ExportCard({ dataList, results }: ExportCardProps) {
  const hasResults = Object.keys(results).length > 0

  const btn =
    'flex w-full items-center justify-center gap-2 rounded-lg border border-line py-2.5 text-[14px] font-medium text-ink transition hover:bg-canvas disabled:cursor-not-allowed disabled:opacity-50'

  return (
    <section className="shrink-0 rounded-xl border border-line bg-white p-4 shadow-panel">
      <h2 className="text-[15px] font-bold text-ink">数据导出</h2>
      <p className="mb-3 mt-1 text-[12px] text-muted">导出当前任务编码结果及相关数据</p>
      <div className="space-y-2">
        <button type="button" className={btn} onClick={() => exportResultsToCsv(dataList, results)} disabled={!hasResults}>
          <FileText className="h-4 w-4 text-muted" />
          导出 CSV
        </button>
        <button type="button" className={btn} onClick={() => exportResultsToExcel(dataList, results)} disabled={!hasResults}>
          <FileSpreadsheet className="h-4 w-4 text-success" />
          导出 Excel
        </button>
        <button type="button" className={btn} onClick={() => exportStage1Dataset(dataList, results)} disabled={!hasResults}>
          <Database className="h-4 w-4 text-accent" />
          导出一阶段数据集
        </button>
      </div>
    </section>
  )
}
