import { formatFieldCode, formatFieldStage1Lines, formatFieldStage2Lines } from '@/lib/formatters'
import type { EncodingResult, FieldPayload } from '@/types/encoding'

const FIELD_ORDER = ['TYPE', 'SIZE', 'THICKNESS', 'PRESSURE', 'MATERIAL', 'STANDARD'] as const
const FIELD_LABELS: Record<(typeof FIELD_ORDER)[number], string> = {
  TYPE: '种类',
  SIZE: '尺寸',
  THICKNESS: '壁厚',
  PRESSURE: '磅级',
  MATERIAL: '材质',
  STANDARD: '规范',
}

interface FieldBreakdownCardProps {
  result: EncodingResult | null
}

function Cell({ lines, note }: { lines: string[]; note?: string }) {
  return (
    <td className="px-3 py-2.5 align-top text-muted">
      {lines.map((line, idx) => (
        <div key={idx} className={idx > 0 ? 'mt-1' : undefined}>
          {line}
        </div>
      ))}
      {note ? <div className="mt-1.5 text-[12px] leading-5 text-[#8a97ad]">{note}</div> : null}
    </td>
  )
}

function getThicknessStage2Note(field: FieldPayload | undefined, fieldType: string) {
  if (fieldType !== 'THICKNESS') return ''
  const notes = Array.isArray(field?.stage2_input?.notes) ? field!.stage2_input!.notes! : []
  return notes.length > 0 ? String(notes[0] ?? '').trim() : ''
}

function getEncodeSourceTag(field: FieldPayload | undefined) {
  const code = String(field?.stage2_output?.code ?? '').trim()
  if (!code) return ''
  const source = String(field?.encode_confidence_v2?.source ?? '').trim().toLowerCase()
  return source === 'llm_fallback' ? '模型' : ''
}

export function FieldBreakdownCard({ result }: FieldBreakdownCardProps) {
  return (
    <section className="flex h-full flex-col rounded-xl border border-line bg-white p-4 shadow-panel">
      <div className="min-h-0 flex-1 overflow-auto">
      <table className="w-full border-collapse text-[14px]">
        <thead>
          <tr className="bg-[#fafbfd] text-[13px] text-muted">
            <th className="w-[12%] rounded-l-lg px-3 py-2.5 text-left font-medium">字段</th>
            <th className="w-[31%] px-3 py-2.5 text-left font-medium">一阶段原始识别</th>
            <th className="w-[31%] px-3 py-2.5 text-left font-medium">二阶段实际输入</th>
            <th className="rounded-r-lg px-3 py-2.5 text-left font-medium">最终编码</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[#eef1f6]">
          {FIELD_ORDER.map((fieldType) => {
            const field = result?.fields?.[fieldType]
            const code = formatFieldCode(field)
            const sourceTag = getEncodeSourceTag(field)
            return (
              <tr key={fieldType} className="align-top">
                <td className="px-3 py-2.5 align-top font-medium text-ink">{FIELD_LABELS[fieldType]}</td>
                <Cell lines={formatFieldStage1Lines(field, fieldType)} />
                <Cell lines={formatFieldStage2Lines(field, fieldType)} note={getThicknessStage2Note(field, fieldType)} />
                <td className="px-3 py-2.5 align-top font-mono font-semibold text-accent">
                  <span>{code}</span>
                  {sourceTag ? (
                    <span className="ml-2 inline-flex rounded-md border border-[#d9e4ff] bg-[#edf3ff] px-2 py-0.5 font-sans text-[11px] font-medium leading-4 text-accent">
                      {sourceTag}
                    </span>
                  ) : null}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
      </div>
    </section>
  )
}
